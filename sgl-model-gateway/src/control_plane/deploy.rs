//! Declarative model deployment with swap-slot lifecycle.
//!
//! POST /v1/control/models {name, type, path, strategy}
//!   -> pick the least-loaded unit with capacity
//!   -> load (decode first, then prefill — P0: decode ~13s, prefill ~6s)
//!   -> ACTIVE
//!   -> pick the least-busy model already on that unit (replacee)
//!   -> DRAIN it (stop new routes + wait inflight == 0)
//!   -> unload the replacee (swap slot makes overwrite impossible)
//!
//! Engine-specific differences (lora vs base, pd double-end load order) are
//! isolated here — the rest of the router treats every model identically.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use tokio::time::sleep;

use super::{ChildUnit, ControlPlaneState, ModelDeployment};

#[derive(Clone, Debug, Deserialize)]
pub struct DeployRequest {
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String, // "lora" | "base"
    pub path: String,
    #[serde(default = "default_strategy")]
    pub strategy: String, // "any" | "all"
}

fn default_strategy() -> String {
    "any".to_string()
}

/// Number of seconds to wait for a drained model's inflight count to hit zero
/// before giving up (and failing the swap).
const DRAIN_TIMEOUT_SECS: u64 = 1800; // matches 2× HOP_TIMEOUT
const DRAIN_POLL_INTERVAL_SECS: u64 = 1;
/// Hard cap for the engine-side load phase (both engines in parallel).
/// GLM-5.2 LoRAs carry tens of thousands of target-module tensors and the
/// engines can take many minutes per load; 35min is generous but finite —
/// without it a hung engine load wedges the deployment in LOADING forever.
const LOAD_TIMEOUT_SECS: u64 = 2100;

// ---------------------------------------------------------------------------
// Engine adapter primitives
// ---------------------------------------------------------------------------

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(3600))
        // Below engine uvicorn keep-alive (5s default) — see jobs.rs note.
        .pool_idle_timeout(Duration::from_secs(4))
        .build()
        .expect("deploy client")
}

fn auth_header(unit: &ChildUnit, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match &unit.api_key {
        Some(k) => builder.header("Authorization", format!("Bearer {}", k)),
        None => builder,
    }
}

/// Resolve a model path that may be a remote download URL.
///
/// Supported forms:
///   - local directory           (existing weights on this host)
///   - `http(s)://.../x.tar.gz | x.tar.zst`
///     passed through to the engine: each PD node downloads + extracts its
///     own local copy (local bandwidth, parallel across nodes, no ssh
///     dependency, no intermediate hop through this router).
///
/// Returns the value to hand to `load_lora_adapter` (local dir or URL).
async fn resolve_model_path(_name: &str, path: &str) -> Result<String, String> {
    Ok(path.to_string())
}

/// Load a model on one engine endpoint (`/load_lora_adapter` for lora;
/// base models use the same endpoint family in this branch).
async fn load_on_endpoint(
    client: &reqwest::Client,
    unit: &ChildUnit,
    base: &str,
    name: &str,
    path: &str,
) -> bool {
    let url = format!("{}/load_lora_adapter", base.trim_end_matches('/'));
    let body = json!({ "lora_name": name, "lora_path": path });
    tracing::info!(url = %url, model = %name, "load_lora_adapter request sent (engine loads can take minutes for large LoRAs)");
    let mut builder = client.post(&url).json(&body);
    builder = auth_header(unit, builder);
    match builder.send().await {
        Ok(res) => {
            let status = res.status();
            if !status.is_success() {
                // Log the engine's response body — it carries success=false
                // plus error_message, which is the actual reason for the 4xx.
                let resp_body = res.text().await.unwrap_or_default();
                tracing::warn!(
                    url = %url,
                    status = %status,
                    model = %name,
                    path = %path,
                    resp_body = %resp_body,
                    "load_lora_adapter rejected by engine"
                );
            } else {
                tracing::info!(url = %url, model = %name, "load_lora_adapter accepted");
            }
            status.is_success()
        }
        Err(e) => {
            tracing::warn!(url = %url, model = %name, path = %path, error = %e, "load_lora_adapter failed");
            false
        }
    }
}

/// Unload a model from one engine endpoint. Uses a short timeout — a hung
/// engine unload must not block the whole deployment forever.
async fn unload_on_endpoint(
    client: &reqwest::Client,
    unit: &ChildUnit,
    base: &str,
    name: &str,
) -> bool {
    let url = format!("{}/unload_lora_adapter", base.trim_end_matches('/'));
    let body = json!({ "lora_name": name });
    let mut builder = client.post(&url).json(&body);
    builder = auth_header(unit, builder);
    match builder.send().await {
        Ok(res) => {
            let status = res.status();
            let ok = status.is_success();
            if !ok {
                let resp_body = res.text().await.unwrap_or_default();
                tracing::warn!(
                    url = %url,
                    status = %status,
                    model = %name,
                    resp_body = %resp_body,
                    "unload_lora_adapter rejected"
                );
            }
            ok
        }
        Err(e) => {
            tracing::warn!(url = %url, model = %name, error = %e, "unload_lora_adapter failed");
            false
        }
    }
}

/// Short-timeout client for unload operations (engine unload can hang).
fn unload_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        // Below engine uvicorn keep-alive (5s default) — see jobs.rs note.
        .pool_idle_timeout(Duration::from_secs(4))
        .build()
        .expect("unload client")
}

/// Load a model on a whole unit. For PD clusters, decode and prefill are
/// loaded **in parallel** (each engine downloads its own copy concurrently) —
/// total time ≈ max(decode, prefill) instead of the serial sum.
async fn load_model_on_unit(unit: &ChildUnit, name: &str, path: &str) -> bool {
    let client = client();
    let (d_ok, p_ok) = tokio::join!(
        async {
            match &unit.decode_url {
                Some(d) => load_on_endpoint(&client, unit, d, name, path).await,
                None => true,
            }
        },
        async {
            match &unit.prefill_url {
                Some(p) => load_on_endpoint(&client, unit, p, name, path).await,
                None => true,
            }
        },
    );
    d_ok && p_ok
}

async fn unload_model_on_unit(unit: &ChildUnit, name: &str) -> bool {
    let client = unload_client();
    let (d_ok, p_ok) = tokio::join!(
        async {
            match &unit.decode_url {
                Some(d) => unload_on_endpoint(&client, unit, d, name).await,
                None => true,
            }
        },
        async {
            match &unit.prefill_url {
                Some(p) => unload_on_endpoint(&client, unit, p, name).await,
                None => true,
            }
        },
    );
    d_ok && p_ok
}

/// Distribute downloaded weights to all cluster nodes when the router runs on
/// a different host than prefill/decode (engine `load_lora_adapter` needs the
/// weights locally on each node). Uses rsync over ssh.
///
/// NOTE: deprecated — remote URL weights are now downloaded by each engine
/// node locally (engine-side `resolve_lora_local_path`); no ssh needed.
#[allow(dead_code)]
async fn distribute_weights(unit: &ChildUnit, local_dir: &str) -> Result<(), String> {
    if unit.ssh_hosts.is_empty() {
        return Ok(());
    }
    let (base, name) = match local_dir.rsplit_once('/') {
        Some((b, n)) => (b.to_string(), n.to_string()),
        None => return Err(format!("bad local dir: {}", local_dir)),
    };
    for host in &unit.ssh_hosts {
        let (userhost, port_opt) = match host.rsplit_once(':') {
            Some((uh, p)) if p.chars().all(|c| c.is_ascii_digit()) => {
                (uh.to_string(), Some(p.to_string()))
            }
            _ => (host.clone(), None),
        };
        let ssh_opt = match &port_opt {
            Some(p) => format!("ssh -p {} -o BatchMode=yes -o ConnectTimeout=10", p),
            None => "ssh -o BatchMode=yes -o ConnectTimeout=10".to_string(),
        };
        tracing::info!(host = %userhost, name, "distributing model weights to cluster node");
        let status = tokio::process::Command::new("rsync")
            .args(["-aq", "-e", &ssh_opt])
            .arg(format!("{}/", local_dir))
            .arg(format!("{}:{}/{}", userhost, base, name))
            .status()
            .await
            .map_err(|e| format!("rsync spawn failed: {}", e))?;
        if !status.success() {
            return Err(format!(
                "rsync to {} failed (exit {:?}); check ssh_hosts and node disk",
                userhost,
                status.code()
            ));
        }
        tracing::info!(host = %userhost, name, "weights distributed");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Selection helpers
// ---------------------------------------------------------------------------

/// Pick the least-loaded unit that (a) is registered, (b) can host the model
/// (either it has a free slot, or it has a loaded model that can be swapped
/// out), and (c) does not already serve this model.
fn select_target_engine(state: &ControlPlaneState, req: &DeployRequest) -> Option<ChildUnit> {
    let children = state.list_children();
    let mut candidates: Vec<ChildUnit> = children
        .into_iter()
        .filter(|u| u.kind == "pd_cluster" || u.kind == "engine")
        .filter(|u| u.decode_url.is_some() || u.prefill_url.is_some())
        .filter(|u| u.models.iter().all(|m| m.name != req.name))
        // Free slot OR at least one model that can be swapped out. Capacity
        // slots are LoRA-only (base model never counts toward capacity).
        .filter(|u| {
            let lora_count = u
                .models
                .iter()
                .filter(|m| m.model_type != "base")
                .count();
            lora_count < u.capacity.max(1) || !u.models.is_empty()
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Sort by pressure (running reqs), then by loaded-model count.
    candidates.sort_by(|a, b| {
        let pa = state
            .get_metrics(&a.url)
            .map(|m| m.running_reqs as i64)
            .unwrap_or(0);
        let pb = state
            .get_metrics(&b.url)
            .map(|m| m.running_reqs as i64)
            .unwrap_or(0);
        pa.cmp(&pb).then(a.models.len().cmp(&b.models.len()))
    });

    candidates.into_iter().next()
}

/// Pick the replacee on a unit: the loaded LoRA (never the base model) with
/// the lowest inflight — i.e. least busy, the natural "swap slot" candidate.
/// The swap flow drains it fully before unloading, so reusing its buffer is
/// safe (no live request reads it).
fn select_replacee(state: &ControlPlaneState, unit: &ChildUnit) -> Option<String> {
    unit.models
        .iter()
        .filter(|m| m.model_type != "base")
        .map(|m| {
            let inflight = state.inflight_of(&m.name);
            (m.name.clone(), inflight)
        })
        .min_by_key(|(_, inflight)| *inflight)
        .map(|(name, _)| name)
}

/// Recursive descent: this router has no local engine with capacity, so
/// forward the declarative deploy request to the least-loaded sub-router and
/// return ITS response verbatim (the sub-tree executes the deployment).
async fn forward_to_subrouter(
    state: &ControlPlaneState,
    req: &DeployRequest,
) -> Option<(axum::http::StatusCode, serde_json::Value)> {
    let children = state.list_children();
    let mut routers: Vec<ChildUnit> = children
        .into_iter()
        .filter(|c| c.kind == "router")
        .collect();
    routers.sort_by_key(|c| {
        state
            .get_metrics(&c.url)
            .map(|m| m.running_reqs as i64)
            .unwrap_or(0)
    });
    let target = routers.into_iter().next()?;

    // Forward the same declarative request down the tree.
    let url = format!("{}/v1/control/models", target.url.trim_end_matches('/'));
    let body = json!({
        "name": req.name,
        "type": req.model_type,
        "path": req.path,
        "strategy": req.strategy,
    });
    let mut builder = client().post(&url).json(&body);
    if let Some(key) = &target.api_key {
        builder = builder.header("Authorization", format!("Bearer {}", key));
    }
    let resp = builder.send().await.ok()?;
    let status = resp.status();
    let value = resp.json::<serde_json::Value>().await.ok()?;
    Some((status, value))
}

async fn wait_inflight_zero(state: &ControlPlaneState, name: &str) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(DRAIN_TIMEOUT_SECS);
    while tokio::time::Instant::now() < deadline {
        if state.inflight_of(name) == 0 {
            return true;
        }
        sleep(Duration::from_secs(DRAIN_POLL_INTERVAL_SECS)).await;
    }
    false
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub async fn deploy_model(
    State(state): State<Arc<ControlPlaneState>>,
    Json(req): Json<DeployRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.name.is_empty() || req.path.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "name and path are required" })),
        );
    }

    // Already deployed on any registered unit → explicit error, not a 503.
    let already = state
        .list_children()
        .iter()
        .find(|u| u.models.iter().any(|m| m.name == req.name))
        .map(|u| u.id.clone());
    if let Some(unit_id) = already {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "already_deployed",
                "model": req.name,
                "engine": unit_id,
            })),
        );
    }

    // Already queued / loading on this router → idempotent 202.
    if let Some(dep) = state.get_deployment(&req.name) {
        if dep.state == "QUEUED" || dep.state == "LOADING" {
            return (
                StatusCode::ACCEPTED,
                Json(json!({ "status": dep.state, "name": req.name })),
            );
        }
    }

    // Only one deployment may run at a time on this router — concurrent
    // drains target the same replacee and deadlock the engine's unload.
    let busy = state
        .list_deployments()
        .iter()
        .any(|d| d.state == "QUEUED" || d.state == "LOADING");
    if busy {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "error": "deployment_in_progress", "model": req.name })),
        );
    }

    // Async deploy: enqueue now, return 202 immediately. The heavy work
    // (engine-side URL download of multi-GB checkpoints, drain, load) runs
    // in the background so a client timeout / disconnect cannot abort it.
    state.set_deployment(ModelDeployment {
        name: req.name.clone(),
        model_type: req.model_type.clone(),
        path: req.path.clone(),
        state: "QUEUED".to_string(),
        engine_id: None,
        replaced_model: None,
        error: None,
    });
    let s = state.clone();
    let r = req.clone();
    tokio::spawn(async move {
        let outcome = execute_deploy(&s, &r).await;
        let mut deploys = s.deployments.write().unwrap();
        if let Some(d) = deploys.get_mut(&r.name) {
            match outcome {
                Ok(()) => {
                    // Forwarded deployments are owned by the sub-tree; keep
                    // the FORWARDED marker (real state lives downstream).
                    if d.state != "FORWARDED" {
                        d.state = "ACTIVE".to_string();
                    }
                }
                Err(e) => {
                    d.state = "FAILED".to_string();
                    d.error = Some(e);
                }
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(json!({
            "status": "accepted",
            "name": req.name,
            "hint": "poll GET /v1/control/models for the deployment state",
        })),
    )
}

/// Execute a declarative deployment in the background.
async fn execute_deploy(
    state: &Arc<ControlPlaneState>,
    req: &DeployRequest,
) -> Result<(), String> {
    tracing::info!(name = %req.name, "execute_deploy: start");
    // 1. Choose the target unit (least pressure + capacity). If this router
    //    has no local engine that can host it, forward the request down to
    //    the least-loaded sub-router (its 202 means the sub-tree owns it).
    let target = match select_target_engine(state, req) {
        Some(t) => Some(t),
        None => {
            if let Some((status, _)) = forward_to_subrouter(state, req).await {
                if status.is_success() {
                    if let Some(d) = state.deployments.write().unwrap().get_mut(&req.name) {
                        d.state = "FORWARDED".to_string();
                    }
                    return Ok(());
                }
            }
            None
        }
    };
    let target = target.ok_or_else(|| "no_unit_with_capacity".to_string())?;
    tracing::info!(name = %req.name, engine = %target.id, "execute_deploy: target selected");
    // NOTE: remote URL weights are downloaded by each engine node locally
    // (resolve_model_path passes the URL through; the engine's
    // load_lora_adapter resolves it). No ssh / rsync distribution needed.

    // 2. Free a slot if the unit is full (swap-slot semantics: the least-busy
    //    loaded model is drained, then unloaded before the new one loads).
    // NOTE: only LoRAs consume capacity slots — the base model lives in the
    //    same engine but is never a swap candidate, so it must not count
    //    toward `capacity` (else a 2-slot engine with 1 LoRA + base would
    //    appear full and swap on every deploy).
    let lora_count = target
        .models
        .iter()
        .filter(|m| m.model_type != "base")
        .count();
    let has_free_slot = lora_count < target.capacity.max(1);
    let replaced = if has_free_slot {
        None
    } else {
        select_replacee(state, &target)
    };
    let mut replaced_path: Option<String> = None;
    if let Some(rep) = &replaced {
        state.mark_model_draining(rep);
        let drained = wait_inflight_zero(state, rep).await;
        if !drained {
            tracing::warn!(
                model = %rep,
                engine = %target.id,
                "drain timeout — swap aborted for replacee"
            );
            return Err(format!("drain timeout: {}", rep));
        }
        replaced_path = state.get_deployment(rep).map(|d| d.path);
        unload_model_on_unit(&target, replaced_path.as_deref().unwrap_or(rep)).await;
        state.remove_deployment(rep);
        state.remove_child_model(&target.id, rep);
    }
    tracing::info!(name = %req.name, replaced = ?replaced, "execute_deploy: drain done");

    // 3. LOADING: decode first (slower), then prefill. Both must succeed.
    //    Remote URL weights are downloaded by each engine node locally.
    //    Hard timeout: a hung engine load must not wedge the deployment in
    //    LOADING forever (engines can legitimately take minutes for large
    //    LoRAs, so the cap is generous — see LOAD_TIMEOUT_SECS).
    if let Some(d) = state.deployments.write().unwrap().get_mut(&req.name) {
        d.state = "LOADING".to_string();
        d.engine_id = Some(target.id.clone());
        d.replaced_model = replaced.clone();
    }
    let load_started = std::time::Instant::now();
    // Engine-side registration key = path (NOT the control-plane name), so
    // runtime requests carrying `lora_path` hit the loaded adapter directly
    // (sglang indexes loaded adapters by the name passed to load_lora_adapter;
    // PD decode rejects unknown lora_path with "never been loaded").
    let load_result = tokio::time::timeout(
        Duration::from_secs(LOAD_TIMEOUT_SECS),
        load_model_on_unit(&target, &req.path, &req.path),
    )
    .await;
    let load_ok = match load_result {
        Ok(ok) => ok,
        Err(_) => {
            tracing::error!(
                name = %req.name,
                engine = %target.id,
                elapsed_secs = load_started.elapsed().as_secs(),
                "execute_deploy: load phase timed out — unloading partial load"
            );
            // Best-effort cleanup of whatever did load.
            let _ = unload_model_on_unit(&target, &req.path).await;
            return Err(format!(
                "load timed out on {} after {}s",
                target.id,
                load_started.elapsed().as_secs()
            ));
        }
    };
    if !load_ok {
        // Rollback: try to restore the replacee we just unloaded.
        if let (Some(rep), Some(path)) = (&replaced, &replaced_path) {
            if load_model_on_unit(&target, path, path).await {
                state.set_deployment(ModelDeployment {
                    name: rep.clone(),
                    model_type: "lora".to_string(),
                    path: path.clone(),
                    state: "ACTIVE".to_string(),
                    engine_id: Some(target.id.clone()),
                    replaced_model: None,
                    error: None,
                });
                state.add_child_model(&target.id, rep, "lora");
            }
        }
        // The load partially succeeded on one engine at most — clean up so
        // the engine does not keep a half-deployed adapter.
        let _ = unload_model_on_unit(&target, &req.path).await;
        return Err(format!(
            "load failed on {}; check weights exist on all cluster nodes",
            target.id
        ));
    }

    // 3b. The deployment may have been DELETEd while we were loading (the
    // delete path unloads what it can). If so, do NOT resurrect it — undo
    // the load we just performed and report cancellation.
    let still_loading = {
        let deps = state.deployments.read().unwrap();
        deps.get(&req.name)
            .map(|d| d.state == "LOADING" && d.engine_id.as_deref() == Some(target.id.as_str()))
            .unwrap_or(false)
    };
    if !still_loading {
        tracing::warn!(
            name = %req.name,
            engine = %target.id,
            "execute_deploy: deployment was removed/mutated during load — rolling back"
        );
        let _ = unload_model_on_unit(&target, &req.path).await;
        return Err("deployment cancelled during load".to_string());
    }

    // 4. ACTIVE: the new model is now routable.
    state.add_child_model(&target.id, &req.name, &req.model_type);
    Ok(())
}

pub async fn delete_model(
    State(state): State<Arc<ControlPlaneState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let deployment = state.get_deployment(&name);
    let Some(dep) = deployment else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "not_deployed" })));
    };

    tracing::info!(model = %name, state = %dep.state, "delete_model: start");

    // Fast paths: a deployment that never reached ACTIVE cannot carry
    // traffic, so there is nothing to drain — waiting DRAIN_TIMEOUT_SECS
    // here is what used to hang DELETE for half an hour.
    let needs_drain = !matches!(dep.state.as_str(), "QUEUED" | "LOADING" | "FAILED");
    if needs_drain {
        // Drain first, then unload on the deployment's engine.
        state.mark_model_draining(&name);
        let drained = wait_inflight_zero(&state, &name).await;
        if !drained {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "drain_timeout" })),
            );
        }
    } else {
        state.mark_model_draining(&name);
    }

    // Unload even for FAILED/LOADING states: the engine may hold a partial
    // load (unload of a non-existent adapter is a harmless 400).
    if let Some(engine_id) = &dep.engine_id {
        let children = state.list_children();
        if let Some(unit) = children.iter().find(|u| &u.id == engine_id) {
            unload_model_on_unit(unit, &dep.path).await;
        }
    }
    state.remove_deployment(&name);
    tracing::info!(model = %name, "delete_model: removed");
    (
        StatusCode::OK,
        Json(json!({ "status": "removed", "model": name })),
    )
}
