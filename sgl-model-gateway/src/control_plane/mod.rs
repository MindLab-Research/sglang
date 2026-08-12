//! Control plane: model views, live metrics aggregation, unit registration.
//!
//! Every router exposes the same control-plane API (only its scope differs):
//!   GET  /v1/models    aggregated model view of this subtree
//!   GET  /v1/units     topology + live load of children (engines / sub-routers)
//!   POST /v1/register  child unit registration (recursive tree building)
//!   GET  /v1/healthz   liveness + summary snapshot for parent polling
//!
//! The design is deliberately generic — LoRA adapters and base models are both
//! just `Model` entries; PD clusters and standalone engines are both just
//! `Unit` entries. No engine-specific logic lives here.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::time;

use crate::core::WorkerRegistry;

pub mod deploy;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Live load metrics scraped from a child's `/metrics` (sglang prometheus text).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EngineMetrics {
    pub running_reqs: f64,
    pub queue_reqs: f64,
    pub gen_throughput: f64,
    pub cache_hit_rate: f64,
    pub tpot_avg_seconds: f64,
    pub hicache_used_tokens: f64,
    pub hicache_total_tokens: f64,
    pub healthy: bool,
    pub updated_at_secs: u64,
}

/// A single model (base or LoRA) as observed on one unit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelState {
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String, // "base" | "lora"
    pub state: String,      // ACTIVE | LOADING | DRAINING | SWAP_OUT | EVICTED
    pub inflight: usize,
}

/// Aggregated view of one model across the whole subtree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelView {
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub state: String,
    pub engine_count: usize,
    pub per_engine: Vec<PerEngineInflight>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerEngineInflight {
    pub engine_id: String,
    pub inflight: usize,
}

/// A routeable unit below this router (either an engine or a sub-router).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChildUnit {
    pub id: String,
    pub url: String,
    pub kind: String, // "router" | "engine" | "pd_cluster"
    /// PD cluster prefill admin endpoint (for load/unload).
    #[serde(default)]
    pub prefill_url: Option<String>,
    /// PD cluster decode admin endpoint (for load/unload).
    #[serde(default)]
    pub decode_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Models currently deployed on this unit (engine capability view).
    #[serde(default)]
    pub models: Vec<ModelState>,
    /// Max concurrently loaded models on this unit (e.g. max_loaded_loras).
    #[serde(default)]
    pub capacity: usize,
    /// SSH targets for distributing downloaded weights to the cluster nodes
    /// (router runs on a different host than prefill/decode). Format:
    /// `user@host[:ssh_port]`, e.g. `root@10.0.0.67:1022`.
    #[serde(default)]
    pub ssh_hosts: Vec<String>,
}

/// Topology + live load of one child unit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitView {
    pub id: String,
    pub url: String,
    pub kind: String,
    pub healthy: bool,
    pub metrics: EngineMetrics,
    pub models: Vec<ModelState>,
}

/// Declarative deployment intent / observed state for one model.
#[derive(Clone, Debug, Serialize)]
pub struct ModelDeployment {
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String, // "lora" | "base"
    pub path: String,
    pub state: String, // QUEUED | EVICTED | LOADING | ACTIVE | DRAINING | SWAP_OUT | FAILED
    pub engine_id: Option<String>,
    pub replaced_model: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Full subtree view returned by GET /v1/units?scope=global.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubtreeView {
    pub units: Vec<UnitView>,
    pub generated_at_secs: u64,
    pub ttl_secs: u64,
}

/// Aggregated model view returned by GET /v1/models?scope=global.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalModelsView {
    pub models: HashMap<String, ModelView>,
    pub generated_at_secs: u64,
    pub ttl_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthzSummary {
    pub status: String,
    pub unit_count: usize,
    pub model_count: usize,
    pub total_running: f64,
    pub generated_at_secs: u64,
}

/// Affinity / pressure tuning knobs for this router's data plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Weight of affinity (routing-key stickiness / prefix bucket) vs load in
    /// child selection: 0.0 = pure load, 1.0 = pure affinity.
    #[serde(default = "default_affinity_weight")]
    pub affinity_weight: f64,
    /// Number of leading prompt tokens hashed into the prefix bucket.
    #[serde(default = "default_prefix_bucket_tokens")]
    pub prefix_bucket_tokens: usize,
    /// Normalized-pressure threshold above which a child is considered
    /// saturated and requests spill to the next-least-loaded child.
    #[serde(default = "default_spill_threshold")]
    pub spill_threshold: f64,
}

fn default_affinity_weight() -> f64 {
    0.5
}
fn default_prefix_bucket_tokens() -> usize {
    256
}
fn default_spill_threshold() -> f64 {
    0.8
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            affinity_weight: default_affinity_weight(),
            prefix_bucket_tokens: default_prefix_bucket_tokens(),
            spill_threshold: default_spill_threshold(),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared control-plane state
// ---------------------------------------------------------------------------

pub struct ControlPlaneState {
    /// url -> latest scraped metrics from the child's /metrics endpoint.
    pub metrics_cache: RwLock<HashMap<String, EngineMetrics>>,
    /// model name -> in-flight request count as seen by this router's data plane.
    pub model_inflight: DashMap<String, AtomicUsize>,
    /// registered child units (recursive tree).
    pub children: RwLock<Vec<ChildUnit>>,
    /// declarative model deployments (desired state tracked here).
    pub deployments: RwLock<HashMap<String, ModelDeployment>>,
    /// affinity/pressure routing knobs.
    pub routing_config: RwLock<RoutingConfig>,
    /// data-plane observation window for inflight (secs).
    pub inflight_ttl_secs: u64,
    /// last time model_inflight was swept.
    last_sweep: RwLock<Instant>,
}

impl ControlPlaneState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            metrics_cache: RwLock::new(HashMap::new()),
            model_inflight: DashMap::new(),
            children: RwLock::new(Vec::new()),
            deployments: RwLock::new(HashMap::new()),
            routing_config: RwLock::new(RoutingConfig::default()),
            inflight_ttl_secs: 60,
            last_sweep: RwLock::new(Instant::now()),
        })
    }

    /// Data-plane hook: call when a request with `model` starts (inflight++).
    pub fn request_started(&self, model: &str) {
        let counter = self
            .model_inflight
            .entry(model.to_string())
            .or_insert_with(|| AtomicUsize::new(0));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Data-plane hook: call when a request with `model` finishes (inflight--).
    pub fn request_finished(&self, model: &str) {
        if let Some(counter) = self.model_inflight.get(model) {
            // fetch_update is stable; relax ordering is fine for counters.
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
        }
        self.maybe_sweep();
    }

    fn maybe_sweep(&self) {
        let now = Instant::now();
        let need = {
            let last = self.last_sweep.read().unwrap();
            now.duration_since(*last) > Duration::from_secs(self.inflight_ttl_secs)
        };
        if !need {
            return;
        }
        let mut last = self.last_sweep.write().unwrap();
        *last = now;
        self.model_inflight
            .retain(|_, counter| counter.load(Ordering::Relaxed) > 0);
    }

    pub fn inflight_of(&self, model: &str) -> usize {
        self.model_inflight
            .get(model)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    pub fn update_metrics(&self, url: &str, metrics: EngineMetrics) {
        self.metrics_cache.write().unwrap().insert(url.to_string(), metrics);
    }

    pub fn get_metrics(&self, url: &str) -> Option<EngineMetrics> {
        self.metrics_cache.read().unwrap().get(url).cloned()
    }

    pub fn register_child(&self, unit: ChildUnit) {
        let mut children = self.children.write().unwrap();
        children.retain(|c| c.id != unit.id);
        children.push(unit);
    }

    pub fn list_children(&self) -> Vec<ChildUnit> {
        self.children.read().unwrap().clone()
    }

    /// Update the model-capability view of a registered child (discovered by
    /// polling its /v1/models or /v1/control/models).
    pub fn update_child_models(&self, id: &str, models: Vec<ModelState>) {
        let mut children = self.children.write().unwrap();
        if let Some(c) = children.iter_mut().find(|c| c.id == id) {
            c.models = models;
        }
    }

    /// Add one model to a child's capability view (after a successful deploy).
    pub fn add_child_model(&self, id: &str, name: &str, model_type: &str) {
        let mut children = self.children.write().unwrap();
        if let Some(c) = children.iter_mut().find(|c| c.id == id) {
            if c.models.iter().all(|m| m.name != name) {
                c.models.push(ModelState {
                    name: name.to_string(),
                    model_type: model_type.to_string(),
                    state: "ACTIVE".to_string(),
                    inflight: 0,
                });
            }
        }
    }

    /// Remove one model from a child's capability view (after a swap-out).
    pub fn remove_child_model(&self, id: &str, name: &str) {
        let mut children = self.children.write().unwrap();
        if let Some(c) = children.iter_mut().find(|c| c.id == id) {
            c.models.retain(|m| m.name != name);
        }
    }

    /// A child that declares it can serve `model` (or no model filter).
    pub fn child_for_model(&self, model: Option<&str>) -> Option<ChildUnit> {
        let children = self.list_children();
        children
            .into_iter()
            .filter(|c| c.kind == "router" || c.kind == "engine" || c.kind == "pd_cluster")
            .filter(|c| {
                model
                    .map(|m| c.models.iter().any(|ms| ms.name == m))
                    .unwrap_or(true)
            })
            .min_by_key(|c| {
                self.get_metrics(&c.url)
                    .map(|m| m.running_reqs as i64)
                    .unwrap_or(0)
            })
    }

    /// Declarative deployment bookkeeping.
    pub fn set_deployment(&self, d: ModelDeployment) {
        self.deployments.write().unwrap().insert(d.name.clone(), d);
    }

    pub fn remove_deployment(&self, name: &str) {
        self.deployments.write().unwrap().remove(name);
    }

    pub fn get_deployment(&self, name: &str) -> Option<ModelDeployment> {
        self.deployments.read().unwrap().get(name).cloned()
    }

    pub fn list_deployments(&self) -> Vec<ModelDeployment> {
        self.deployments
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    pub fn mark_model_draining(&self, name: &str) {
        if let Some(d) = self.deployments.write().unwrap().get_mut(name) {
            d.state = "DRAINING".to_string();
        }
    }

    pub fn routing_config(&self) -> RoutingConfig {
        self.routing_config.read().unwrap().clone()
    }

    pub fn set_routing_config(&self, cfg: RoutingConfig) {
        *self.routing_config.write().unwrap() = cfg;
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Prometheus text scraping
// ---------------------------------------------------------------------------

/// Scrape one sglang /metrics endpoint and return the parsed EngineMetrics.
/// The text format is the standard Prometheus exposition; we only look up the
/// keys we care about and leave everything else untouched.
async fn scrape_metrics(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
) -> Option<EngineMetrics> {
    let url = format!("{}/metrics", base_url.trim_end_matches('/'));
    let mut builder = client.get(&url);
    if let Some(key) = api_key {
        builder = builder.header("Authorization", format!("Bearer {}", key));
    }

    let resp = builder.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let text = resp.text().await.ok()?;
    Some(parse_prometheus(&text))
}

fn parse_float_line<'a>(lines: impl Iterator<Item = &'a str>, prefix: &str) -> f64 {
    for line in lines {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(prefix) {
            if let Some((_, value)) = rest.split_once(' ') {
                if let Ok(v) = value.trim().parse::<f64>() {
                    return v;
                }
            }
        }
    }
    0.0
}

fn parse_prometheus(text: &str) -> EngineMetrics {
    let lines: Vec<&str> = text.lines().collect();

    let mut m = EngineMetrics {
        running_reqs: parse_float_line(
            lines.iter().copied(),
            "sglang:num_running_reqs",
        ),
        queue_reqs: parse_float_line(lines.iter().copied(), "sglang:num_queue_reqs"),
        gen_throughput: parse_float_line(lines.iter().copied(), "sglang:gen_throughput"),
        cache_hit_rate: parse_float_line(lines.iter().copied(), "sglang:cache_hit_rate"),
        hicache_used_tokens: parse_float_line(
            lines.iter().copied(),
            "sglang:hicache_host_used_tokens",
        ),
        hicache_total_tokens: parse_float_line(
            lines.iter().copied(),
            "sglang:hicache_host_total_tokens",
        ),
        healthy: true,
        updated_at_secs: ControlPlaneState::now_secs(),
        ..Default::default()
    };

    // inter_token_latency histogram -> mean latency (fallback for tpot).
    let sum = parse_float_line(
        lines.iter().copied(),
        "sglang:inter_token_latency_seconds_sum",
    );
    let count = parse_float_line(
        lines.iter().copied(),
        "sglang:inter_token_latency_seconds_count",
    );
    if count > 0.0 {
        m.tpot_avg_seconds = sum / count;
    }
    m
}

// ---------------------------------------------------------------------------
// Background collector
// ---------------------------------------------------------------------------

/// Spawn the periodic metrics collector. Scrapes every registered child
/// (engine / sub-router) every `interval_secs` seconds and stores the result.
pub async fn spawn_metrics_collector(
    state: Arc<ControlPlaneState>,
    workers: Arc<WorkerRegistry>,
    interval_secs: u64,
) {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("metrics collector client");
        let mut ticker = time::interval(Duration::from_secs(interval_secs.max(1)));
        loop {
            ticker.tick().await;
            let units: Vec<ChildUnit> = state.list_children();
            // 1) scrape registered children first + discover their model capability
            for unit in &units {
                // For a pd_cluster the metrics/capability source is the engine
                // endpoint (prefill), not the cluster router URL.
                let source_url = if unit.kind == "pd_cluster" {
                    unit.prefill_url
                        .clone()
                        .unwrap_or_else(|| unit.url.clone())
                } else {
                    unit.url.clone()
                };
                if let Some(m) = scrape_metrics(&client, &source_url, unit.api_key.as_deref()).await
                {
                    state.update_metrics(&unit.url, m);
                }
                // capability discovery: routers expose /v1/control/models;
                // engines expose /v1/models. For pd_cluster prefer the decode
                // endpoint — sglang's decode /v1/models returns the full LoRA
                // list, while prefill only reports the base model.
                let cap_url = if unit.kind == "router" {
                    format!("{}/v1/control/models", unit.url.trim_end_matches('/'))
                } else {
                    let cap_source = if unit.kind == "pd_cluster" {
                        unit.decode_url
                            .clone()
                            .unwrap_or_else(|| source_url.clone())
                    } else {
                        source_url.clone()
                    };
                    format!("{}/v1/models", cap_source.trim_end_matches('/'))
                };
                if let Some(models) =
                    fetch_capability(&client, &cap_url, unit.api_key.as_deref()).await
                {
                    state.update_child_models(&unit.id, models);
                }
            }
            // 2) also scrape workers in the registry (leaf engines) so a router
            //    started with --worker-urls / --prefill gets metrics too.
            for w in workers.get_all() {
                let url = w.url().to_string();
                if let Some(m) = scrape_metrics(&client, &url, w.api_key().as_deref()).await {
                    state.update_metrics(&url, m);
                }
            }
        }
    });
}

/// Fetch the model capability list of a child (either OpenAI /v1/models or
/// control-plane /v1/control/models). Returns an empty vec on any failure so
/// transient errors do not clear the last known capability.
async fn fetch_capability(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
) -> Option<Vec<ModelState>> {
    let mut builder = client.get(url);
    if let Some(key) = api_key {
        builder = builder.header("Authorization", format!("Bearer {}", key));
    }
    let resp = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(url, error = %e, "capability fetch send failed");
            return None;
        }
    };
    if !resp.status().is_success() {
        tracing::warn!(url, status = %resp.status(), "capability fetch non-200");
        return None;
    }
    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(url, error = %e, "capability fetch json failed");
            return None;
        }
    };

    // control-plane shape: { models: { name: ModelView, ... } }
    if let Some(models_obj) = body.get("models").and_then(|m| m.as_object()) {
        let mut out = Vec::new();
        for (name, _v) in models_obj {
            out.push(ModelState {
                name: name.clone(),
                model_type: "lora".to_string(),
                state: "ACTIVE".to_string(),
                inflight: 0,
            });
        }
        return Some(out);
    }

    // OpenAI shape: { data: [ { id, parent, ... } ] }
    // A base model has parent=null; LoRA adapters reference the base as parent.
    if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
        let mut out = Vec::new();
        for item in data {
            let name = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let is_base = item.get("parent").and_then(|p| p.as_str()).is_none();
            out.push(ModelState {
                name: name.to_string(),
                model_type: if is_base { "base" } else { "lora" }.to_string(),
                state: "ACTIVE".to_string(),
                inflight: 0,
            });
        }
        return Some(out);
    }

    None
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn get_models(
    State(state): State<Arc<ControlPlaneState>>,
) -> Json<GlobalModelsView> {
    // Aggregate model views from every known unit. A router child exposes its
    // own control-plane view (recursive); an engine child exposes /v1/models
    // (prefer the decode endpoint for pd_cluster — it returns the full list).
    let children = state.list_children();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let mut models: HashMap<String, ModelView> = HashMap::new();

    for unit in &children {
        if unit.kind == "router" {
            if let Some(sub) = fetch_router_models(&client, &unit.url).await {
                for (name, mv) in sub {
                    merge_model_view(&mut models, name, mv);
                }
            }
        } else {
            let source = if unit.kind == "pd_cluster" {
                unit.decode_url
                    .clone()
                    .unwrap_or_else(|| unit.url.clone())
            } else {
                unit.url.clone()
            };
            if let Some(ms) = fetch_unit_models(&client, &source, unit.api_key.as_deref()).await {
                for m in ms {
                    let entry = models.entry(m.name.clone()).or_insert_with(|| ModelView {
                        name: m.name.clone(),
                        model_type: m.model_type.clone(),
                        state: "ACTIVE".to_string(),
                        engine_count: 0,
                        per_engine: Vec::new(),
                    });
                    entry.engine_count += 1;
                    entry.per_engine.push(PerEngineInflight {
                        engine_id: unit.id.clone(),
                        inflight: m.inflight,
                    });
                }
            }
        }
    }

    // Merge local declarative deployments (this router's own control plane).
    // FAILED deployments are hidden so a retry is not seen as "already
    // deployed" by parent routers.
    for dep in state.list_deployments() {
        if dep.state == "FAILED" {
            continue;
        }
        let entry = models.entry(dep.name.clone()).or_insert_with(|| ModelView {
            name: dep.name.clone(),
            model_type: dep.model_type.clone(),
            state: dep.state.clone(),
            engine_count: 0,
            per_engine: Vec::new(),
        });
        // Only update the state when the subtree has no fresher data.
        if entry.engine_count == 0 {
            entry.state = dep.state.clone();
        }
    }

    Json(GlobalModelsView {
        models,
        generated_at_secs: ControlPlaneState::now_secs(),
        ttl_secs: 10,
    })
}

/// Fetch a sub-router's aggregated model view (`/v1/control/models`).
async fn fetch_router_models(
    client: &reqwest::Client,
    base_url: &str,
) -> Option<HashMap<String, ModelView>> {
    let url = format!("{}/v1/control/models", base_url.trim_end_matches('/'));
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let view: GlobalModelsView = resp.json().await.ok()?;
    Some(view.models)
}

fn merge_model_view(models: &mut HashMap<String, ModelView>, name: String, mv: ModelView) {
    match models.entry(name) {
        std::collections::hash_map::Entry::Occupied(mut e) => {
            let cur = e.get_mut();
            cur.engine_count += mv.engine_count;
            cur.per_engine.extend(mv.per_engine);
            if cur.state == "ACTIVE" && mv.state != "ACTIVE" {
                cur.state = mv.state.clone();
            }
        }
        std::collections::hash_map::Entry::Vacant(v) => {
            v.insert(mv);
        }
    }
}

async fn fetch_unit_models(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
) -> Option<Vec<ModelState>> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let mut builder = client.get(&url);
    if let Some(key) = api_key {
        builder = builder.header("Authorization", format!("Bearer {}", key));
    }
    let resp = builder.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    let data = body.get("data").and_then(|d| d.as_array())?;
    let mut out = Vec::new();
    for item in data {
        let name = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let is_base = item.get("parent").and_then(|p| p.as_str()).is_none();
        out.push(ModelState {
            name: name.to_string(),
            model_type: if is_base { "base" } else { "lora" }.to_string(),
            state: "ACTIVE".to_string(),
            inflight: 0,
        });
    }
    Some(out)
}

pub async fn get_units(
    State(state): State<Arc<ControlPlaneState>>,
) -> Json<SubtreeView> {
    let children = state.list_children();
    let client = reqwest::Client::new();
    let mut units = Vec::new();

    for c in &children {
        if c.kind == "router" {
            // Recursively expand a sub-router's units.
            let url = format!("{}/v1/control/units", c.url.trim_end_matches('/'));
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    if let Ok(view) = resp.json::<SubtreeView>().await {
                        units.extend(view.units);
                    }
                }
            }
        } else {
            let metrics = state.get_metrics(&c.url).unwrap_or_default();
            let healthy = metrics.healthy;
            // Capability-discovered models on this unit, merged with live
            // in-flight counts from this router's data plane.
            let mut models = c.models.clone();
            for entry in state.model_inflight.iter() {
                let name = entry.key().clone();
                match models.iter_mut().find(|m| m.name == name) {
                    Some(m) => m.inflight = state.inflight_of(&name),
                    None => models.push(ModelState {
                        name: name.clone(),
                        model_type: "lora".to_string(),
                        state: "ACTIVE".to_string(),
                        inflight: state.inflight_of(&name),
                    }),
                }
            }
            units.push(UnitView {
                id: c.id.clone(),
                url: c.url.clone(),
                kind: c.kind.clone(),
                healthy,
                metrics,
                models,
            });
        }
    }
    Json(SubtreeView {
        units,
        generated_at_secs: ControlPlaneState::now_secs(),
        ttl_secs: 10,
    })
}

pub async fn register(
    State(state): State<Arc<ControlPlaneState>>,
    Json(unit): Json<ChildUnit>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.register_child(unit);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "registered" })),
    )
}

pub async fn healthz(State(state): State<Arc<ControlPlaneState>>) -> Json<HealthzSummary> {
    let units = state.list_children();
    let model_count = state.model_inflight.len();
    let total_running: f64 = units
        .iter()
        .filter_map(|c| state.get_metrics(&c.url))
        .map(|m| m.running_reqs)
        .sum();
    Json(HealthzSummary {
        status: "ok".to_string(),
        unit_count: units.len(),
        model_count,
        total_running,
        generated_at_secs: ControlPlaneState::now_secs(),
    })
}

pub async fn get_routing(State(state): State<Arc<ControlPlaneState>>) -> Json<RoutingConfig> {
    Json(state.routing_config())
}

pub async fn put_routing(
    State(state): State<Arc<ControlPlaneState>>,
    Json(cfg): Json<RoutingConfig>,
) -> (StatusCode, Json<RoutingConfig>) {
    state.set_routing_config(cfg.clone());
    (StatusCode::OK, Json(cfg))
}

/// Update the manual policy cache-entry gauge so the existing metrics dashboard
/// keeps working while control plane adds its own gauges.
pub fn record_registered_units(count: usize) {
    // Reuse the router-level metric if present; otherwise no-op.
    let _ = count;
}
