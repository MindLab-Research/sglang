//! Training job submission / execution / polling / download endpoints.
//!
//! POST   /v1/control/jobs                          submit an array of tasks
//! GET    /v1/control/jobs/{job_id}                 poll job status
//! GET    /v1/control/jobs/{job_id}/result          download full results
//! GET    /v1/control/jobs/{job_id}/tasks/{tid}/result  download one task
//! DELETE /v1/control/jobs/{job_id}                 cleanup
//!
//! Each task is an OpenAI-completions-style request (prompt / max_tokens /
//! temperature / top_p / n / logprobs) plus an optional `lora_path` field.
//! Tasks are executed against this router's own /generate endpoint (native
//! sglang API) so the results carry token ids + per-token logprobs, which the
//! OpenAI /v1/completions API cannot provide. Execution streams over SSE and
//! aggregates chunks incrementally.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{Notify, RwLock, Semaphore};

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

fn default_n() -> u64 {
    1
}

/// Reference to an existing task whose tokens seed a continuation.
///
/// The continuation input is `source input tokens ++ source generated tokens`,
/// resolved server-side, so the client never re-sends the (possibly huge) token
/// arrays. `sample_index` picks which sample of a multi-sample source to
/// continue (default 0).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContinueFrom {
    pub job_id: String,
    /// Source task id (globally unique, e.g. `job_..._t0000`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Source task index inside `job_id` (alternative to `task_id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_index: Option<usize>,
}

/// One training task, in the training client's OpenAI-ish format.
///
/// Exactly one input source must be provided: `prompt` (text), `input_ids`
/// (tokens) or `continue_from` (server-side continuation).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TaskRequest {
    #[serde(default)]
    pub prompt: String,
    /// Token-level input. Takes the place of `prompt`; the engine skips
    /// tokenization entirely, so the sequence is bit-exact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<Vec<i64>>,
    /// Continue an existing task without re-sending any tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continue_from: Option<ContinueFrom>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default = "default_n")]
    pub n: u64,
    /// Optional LoRA adapter path; when absent the base model is used.
    #[serde(default)]
    pub lora_path: Option<String>,
    /// Accepted for compatibility, ignored (single-served-model deployment).
    #[serde(default)]
    pub model: Option<String>,
    /// Accepted for compatibility (we always stream internally).
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub logprobs: Option<Value>,
    #[serde(default)]
    pub stream_options: Option<Value>,
}

impl TaskRequest {
    fn sanitize_n(&self) -> u64 {
        self.n.clamp(1, 64)
    }
}

// ---------------------------------------------------------------------------
// Input-source rules + native /generate body construction. Pure helpers (unit
// tested below) — the job manager only supplies resolved token ids.
// ---------------------------------------------------------------------------

/// Exactly one input source must be given, and it must be non-empty.
fn validate_task_input(req: &TaskRequest) -> Result<(), String> {
    let has_text = !req.prompt.trim().is_empty();
    let has_ids = req.input_ids.is_some();
    let has_cont = req.continue_from.is_some();
    let given = [has_text, has_ids, has_cont].iter().filter(|b| **b).count();
    if given == 0 {
        return Err(
            "task has no input: provide exactly one of 'prompt', 'input_ids' or 'continue_from'"
                .into(),
        );
    }
    if given > 1 {
        return Err(
            "task must provide exactly one of 'prompt', 'input_ids' or 'continue_from'".into(),
        );
    }
    if let Some(ids) = &req.input_ids {
        if ids.is_empty() {
            return Err("task with empty input_ids".into());
        }
    }
    Ok(())
}

/// Build the native /generate body for one task. A resolved `input_ids` (either
/// client-supplied or composed by a continuation) replaces the text prompt so
/// the token sequence the model sees is bit-exact.
fn build_generate_body(req: &TaskRequest, input_ids: Option<&[i64]>) -> Value {
    let mut body = json!({
        "sampling_params": {
            "max_new_tokens": req.max_tokens.unwrap_or(4096),
        },
        "return_logprob": true,
        "stream": true,
    });
    match input_ids {
        Some(ids) => body["input_ids"] = json!(ids),
        None => body["text"] = json!(req.prompt),
    }
    {
        let sp = body["sampling_params"].as_object_mut().unwrap();
        if let Some(t) = req.temperature {
            sp.insert("temperature".into(), json!(t));
        }
        if let Some(p) = req.top_p {
            sp.insert("top_p".into(), json!(p));
        }
    }
    if let Some(lp) = &req.lora_path {
        body["lora_path"] = json!(lp);
    }
    body
}

/// Continuation input: everything the source task was given, plus everything it
/// generated — the model then resumes exactly where it stopped.
fn compose_input_ids(base: &[i64], generated: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(base.len() + generated.len());
    out.extend_from_slice(base);
    out.extend_from_slice(generated);
    out
}

/// Generated tokens of the sample a continuation resumes from.
fn select_sample_ids(result: &TaskResult, sample_index: usize) -> Result<Vec<i64>, String> {
    match result.samples.get(sample_index) {
        Some(s) => Ok(s.output_ids.clone()),
        None => Err(format!(
            "sample_index {} out of range: source has {} sample(s)",
            sample_index,
            result.samples.len()
        )),
    }
}

/// Sampling/adapter overrides for a job-level resume body.
#[derive(Clone, Debug, Default)]
struct ResumeOverrides {
    max_tokens: Option<u64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    lora_path: Option<String>,
    n: Option<u64>,
}

/// Parsed submit body: (job-level lora default, explicit tasks, resume spec).
type ParsedSubmit = (
    Option<String>,
    Vec<TaskRequest>,
    Option<(ContinueFrom, ResumeOverrides)>,
);

/// Parse a submit body into (job-level lora default, explicit tasks, job-level
/// resume spec). Three accepted shapes:
///   * `[ {...}, ... ]`                          — task array
///   * `{ "requests": [ ... ], "lora_path": x }` — object form
///   * `{ "continue_from": { "job_id": ... } }`  — resume a whole job
fn parse_submit_body(body: &Value) -> Result<ParsedSubmit, String> {
    let parse_requests = |arr: &[Value]| -> Result<Vec<TaskRequest>, String> {
        arr.iter()
            .enumerate()
            .map(|(i, v)| {
                serde_json::from_value::<TaskRequest>(v.clone())
                    .map_err(|e| format!("invalid task at index {i}: {e}"))
            })
            .collect()
    };
    match body {
        Value::Array(arr) => Ok((None, parse_requests(arr)?, None)),
        Value::Object(obj) => {
            let lora = obj
                .get("lora_path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            if let Some(cf_val) = obj.get("continue_from") {
                if obj.contains_key("requests") {
                    return Err(
                        "body cannot contain both 'requests' and 'continue_from'".to_string()
                    );
                }
                let cf: ContinueFrom = serde_json::from_value(cf_val.clone())
                    .map_err(|e| format!("invalid continue_from: {e}"))?;
                let ov = ResumeOverrides {
                    max_tokens: obj.get("max_tokens").and_then(|v| v.as_u64()),
                    temperature: obj.get("temperature").and_then(|v| v.as_f64()),
                    top_p: obj.get("top_p").and_then(|v| v.as_f64()),
                    lora_path: lora.clone(),
                    n: obj.get("n").and_then(|v| v.as_u64()),
                };
                return Ok((lora, Vec::new(), Some((cf, ov))));
            }
            let reqs = obj
                .get("requests")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    "object body must contain a 'requests' array or a 'continue_from' object"
                        .to_string()
                })?;
            Ok((lora, parse_requests(reqs)?, None))
        }
        _ => Err(
            "body must be a JSON array of tasks, {requests: [...]}, or {continue_from: {...}}"
                .to_string(),
        ),
    }
}

/// Result of a single sample (one generation of possibly n).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SampleResult {
    pub output_text: String,
    pub output_ids: Vec<i64>,
    /// Per-token logprob values, aligned 1:1 with output_ids.
    pub output_token_logprobs: Vec<f64>,
    /// Raw logprob entries straight from the backend: one
    /// `[logprob, token_id, top_logprobs|null]` triple per output token,
    /// aligned 1:1 with output_ids (the authoritative full-detail form).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_logprob_entries: Vec<Value>,
    /// Top-k logprob detail per token, aligned 1:1 with output_ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_top_logprobs: Vec<Value>,
    pub finish_reason: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

/// Persisted per-task result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub index: usize,
    pub samples: Vec<SampleResult>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub index: usize,
    pub request: TaskRequest,
    pub status: TaskStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<TaskResult>,
}

#[derive(Clone, Debug)]
pub struct Job {
    pub job_id: String,
    pub created_at_unix: u64,
    /// tasks in submission order; mutated under the store's job lock.
    /// (Serialization is manual — see persist_job / recover_from_disk.)
    pub tasks: Arc<RwLock<Vec<Task>>>,
}

impl Job {
    pub async fn aggregate_status(&self) -> &'static str {
        let tasks = self.tasks.read().await;
        let total = tasks.len();
        let mut done = 0usize;
        let mut failed = 0usize;
        let mut queued = 0usize;
        let mut running = 0usize;
        let mut cancelled = 0usize;
        for t in tasks.iter() {
            match t.status {
                TaskStatus::Completed => done += 1,
                TaskStatus::Failed => failed += 1,
                TaskStatus::Queued => queued += 1,
                TaskStatus::Running => running += 1,
                TaskStatus::Cancelled => cancelled += 1,
            }
        }
        if done + failed + cancelled == total {
            if failed == 0 && cancelled == 0 {
                "completed"
            } else if done == 0 && cancelled == 0 {
                "failed"
            } else if done == 0 && failed == 0 {
                "cancelled"
            } else {
                "partial"
            }
        } else if running > 0 {
            "running"
        } else if queued > 0 {
            "queued"
        } else {
            "running"
        }
    }
}

// ---------------------------------------------------------------------------
// Job manager
// ---------------------------------------------------------------------------

pub struct JobManager {
    client: reqwest::Client,
    self_base_url: String,
    api_key: Option<String>,
    /// Engine base URLs, used to tokenize a source prompt exactly when a
    /// `continue_from` continuation needs its token ids.
    engine_urls: Vec<String>,
    data_dir: PathBuf,
    semaphore: Arc<Semaphore>,
    request_timeout: Duration,
    jobs: DashMap<String, Arc<Job>>,
    seq: AtomicU64,
    /// Per-task cancellation flags (task_id -> requested). Live only; not
    /// persisted (a restart marks interrupted tasks Failed anyway).
    cancel_flags: DashMap<String, Arc<AtomicBool>>,
    /// Per-task generated-token progress (task_id -> tokens so far). Live only.
    progress_tokens: DashMap<String, Arc<AtomicU64>>,
    /// Per-task live partial result: the latest aggregated sample snapshot for
    /// a running task, so download endpoints can return "output so far" even
    /// while the job is still generating. Live only (final results are
    /// persisted to task_{index}.json on completion).
    partial_results: DashMap<String, Arc<tokio::sync::RwLock<TaskResult>>>,
    /// Local retention for **finished** jobs. `None` disables garbage
    /// collection. Data older than this is deleted from disk (results and
    /// requests alike), so the training client must have downloaded it by then.
    retention: Option<Duration>,
}

/// Default local retention for finished jobs (hours).
const DEFAULT_RETENTION_HOURS: f64 = 48.0;
/// Default GC scan interval (seconds).
const DEFAULT_GC_INTERVAL_SECS: u64 = 1800;

/// Retention from `SMG_JOBS_RETENTION_HOURS` (hours, fractional allowed).
/// Unset -> 48h; `0` or negative -> GC disabled (`None`).
fn retention_from_env() -> Option<Duration> {
    match std::env::var("SMG_JOBS_RETENTION_HOURS") {
        Ok(raw) => raw
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|h| *h > 0.0)
            .map(|h| Duration::from_secs_f64(h * 3600.0)),
        Err(_) => Some(Duration::from_secs_f64(DEFAULT_RETENTION_HOURS * 3600.0)),
    }
}

/// GC scan interval from `SMG_JOBS_GC_INTERVAL_SECS` (default 1800s).
fn gc_interval_from_env() -> Duration {
    Duration::from_secs(
        std::env::var("SMG_JOBS_GC_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|s| *s > 0)
            .unwrap_or(DEFAULT_GC_INTERVAL_SECS),
    )
}

impl JobManager {
    pub fn new(
        self_base_url: String,
        api_key: Option<String>,
        data_dir: PathBuf,
        max_concurrency: usize,
        request_timeout_secs: u64,
        engine_urls: Vec<String>,
    ) -> Arc<Self> {
        Self::build(
            self_base_url,
            api_key,
            data_dir,
            max_concurrency,
            request_timeout_secs,
            retention_from_env(),
        )
    }

    fn build(
        self_base_url: String,
        api_key: Option<String>,
        data_dir: PathBuf,
        max_concurrency: usize,
        request_timeout_secs: u64,
        retention: Option<Duration>,
    ) -> Arc<Self> {
        // Pool idle timeout MUST stay below the engine's uvicorn keep-alive
        // (SGLANG_TIMEOUT_KEEP_ALIVE, default 5s): a pooled connection idle
        // longer than the server-side keep-alive is dead on the server but
        // still "usable" in our pool — reusing it kills the stream mid-body
        // ("error decoding response body"). 4s < 5s makes that impossible.
        let client = reqwest::Client::builder()
            .pool_idle_timeout(Duration::from_secs(4))
            .connect_timeout(Duration::from_secs(15))
            .build()
            .expect("failed to build jobs http client");
        Arc::new(Self {
            client,
            self_base_url,
            api_key,
            engine_urls,
            data_dir,
            semaphore: Arc::new(Semaphore::new(max_concurrency.max(1))),
            request_timeout: Duration::from_secs(request_timeout_secs.max(60)),
            jobs: DashMap::new(),
            seq: AtomicU64::new(0),
            cancel_flags: DashMap::new(),
            progress_tokens: DashMap::new(),
            partial_results: DashMap::new(),
            retention,
        })
    }

    fn gen_id(&self, prefix: &str) -> String {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let rnd: u32 = rand::rng().random_range(0..10000);
        format!("{prefix}_{millis}_{seq:03}_{rnd:04}")
    }

    fn job_dir(&self, job_id: &str) -> PathBuf {
        self.data_dir.join(sanitize_id(job_id))
    }

    // -- continuation (`continue_from`) --------------------------------------

    fn task_input_path(&self, job_id: &str, index: usize) -> PathBuf {
        self.job_dir(job_id).join(format!("input_{index:04}.json"))
    }

    /// Persist the exact token input a continuation was run with, so a later
    /// continuation of *that* task is bit-exact too (the ids cannot be
    /// recomputed later: a continued task has no text prompt of its own).
    fn persist_task_input(&self, job_id: &str, index: usize, input_ids: &[i64]) {
        let dir = self.job_dir(job_id);
        let _ = std::fs::create_dir_all(&dir);
        atomic_write(
            &self.task_input_path(job_id, index),
            &json!({ "task_id_index": index, "input_ids": input_ids }),
        );
    }

    fn load_task_input(&self, job_id: &str, index: usize) -> Option<Vec<i64>> {
        let raw = std::fs::read_to_string(self.task_input_path(job_id, index)).ok()?;
        let doc: Value = serde_json::from_str(&raw).ok()?;
        let arr = doc.get("input_ids")?.as_array()?;
        let ids: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
        if ids.len() != arr.len() || ids.is_empty() {
            return None;
        }
        Some(ids)
    }

    /// Exact token ids for a prompt string, from an engine's own tokenizer.
    ///
    /// `/generate` text prompts and `/v1/tokenize` run through the same
    /// tokenizer object inside the engine, so the ids are identical; asking an
    /// engine avoids both storing prompt ids with every task (which would
    /// duplicate the whole prompt on disk) and the engine's optional
    /// `prompt_token_ids` echo, which is repeated on every streaming chunk and
    /// would therefore balloon long prompts.
    async fn tokenize_prompt(&self, text: &str) -> Result<Vec<i64>, String> {
        if self.engine_urls.is_empty() {
            return Err(
                "continue_from needs an engine tokenizer, but no engine URL is configured \
                 (set SMG_JOBS_ENGINE_URLS or launch the router with worker URLs)"
                    .into(),
            );
        }
        let mut last_err = String::new();
        for url in &self.engine_urls {
            let endpoint = format!("{}/v1/tokenize", url.trim_end_matches('/'));
            let mut builder = self.client.post(&endpoint).json(&json!({ "prompt": text }));
            if let Some(key) = &self.api_key {
                builder = builder.bearer_auth(key);
            }
            let resp = match builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("{endpoint} request failed: {e}");
                    continue;
                }
            };
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                last_err = format!("{endpoint} returned {status}: {}", truncate(&body, 200));
                continue;
            }
            let doc: Value = match resp.json().await {
                Ok(v) => v,
                Err(e) => {
                    last_err = format!("{endpoint} returned invalid JSON: {e}");
                    continue;
                }
            };
            match doc.get("tokens").and_then(|v| v.as_array()) {
                Some(arr) if !arr.is_empty() && arr.iter().all(|v| v.is_i64()) => {
                    return Ok(arr.iter().filter_map(|v| v.as_i64()).collect());
                }
                Some(arr) => {
                    last_err =
                        format!("{endpoint} returned an unusable 'tokens' array (len={})", arr.len());
                }
                None => last_err = format!("{endpoint} response has no 'tokens' array"),
            }
        }
        Err(format!("could not tokenize the source prompt: {last_err}"))
    }

    /// Mark a task failed before any sample was produced (pre-flight failures:
    /// adapter missing, input resolution error).
    async fn fail_task_preflight(&self, job: &Arc<Job>, task_id: &str, err: String) {
        {
            let mut tasks = job.tasks.write().await;
            if let Some(t) = tasks.iter_mut().find(|t| t.task_id == task_id) {
                t.status = TaskStatus::Failed;
                t.error = Some(err.clone());
            }
        }
        tracing::warn!("jobs: task {} pre-flight failed: {}", task_id, err);
        self.persist_job(job);
    }

    /// Verify that every engine has the adapter loaded.
    ///
    /// PD caveat this guards against: an adapter that is missing on one engine
    /// makes the request hang in `KVPoll.Bootstrapping` until the 600s bootstrap
    /// timeout (decode cannot allocate KV / send KV indices without it), and the
    /// engines deliberately refuse to reload adapters inside a request. Failing
    /// the task here turns a 10-minute silent stall into an actionable error.
    async fn ensure_lora_loaded(&self, lora_path: &str) -> Result<(), String> {
        if self.engine_urls.is_empty() {
            // Nothing to check against: keep working (the engine fails fast on
            // its own now) rather than blocking every adapter task.
            return Ok(());
        }
        let mut missing: Vec<String> = Vec::new();
        for url in &self.engine_urls {
            match self.engine_has_lora(url, lora_path).await {
                Ok(true) => {}
                Ok(false) => missing.push(url.clone()),
                Err(e) => return Err(format!("could not verify adapter on {url}: {e}")),
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "LoRA adapter is not loaded on {} of {} engine(s): {}. A PD request for a \
             missing adapter hangs until the bootstrap timeout (600s). Load it on every \
             engine first:\n  POST <engine>/load_lora_adapter \
             {{\"lora_name\": \"{lora_path}\", \"lora_path\": \"{lora_path}\"}}\n\
             and resubmit afterwards.",
            missing.len(),
            self.engine_urls.len(),
            missing.join(", ")
        ))
    }

    /// Does `base`'s `/v1/models` list `lora_path`? (Engines register an adapter
    /// under the path they were given.)
    async fn engine_has_lora(&self, base: &str, lora_path: &str) -> Result<bool, String> {
        let url = format!("{}/v1/models", base.trim_end_matches('/'));
        let mut builder = self.client.get(&url).timeout(Duration::from_secs(10));
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder.send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("status {}", resp.status()));
        }
        let doc: Value = resp.json().await.map_err(|e| e.to_string())?;
        let has = doc
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter().any(|m| {
                    m.get("id").and_then(|v| v.as_str()) == Some(lora_path)
                        || m.get("root").and_then(|v| v.as_str()) == Some(lora_path)
                })
            })
            .unwrap_or(false);
        Ok(has)
    }

    /// Source tasks a resume reference selects, in submission order.
    async fn select_resume_sources(
        &self,
        cf: &ContinueFrom,
    ) -> Result<(Arc<Job>, Vec<Task>), String> {
        let job = self
            .get_job(&cf.job_id)
            .ok_or_else(|| format!("continue_from: unknown job_id '{}'", cf.job_id))?;
        let tasks = job.tasks.read().await;
        let mut selected: Vec<Task> = if let Some(tid) = &cf.task_id {
            tasks.iter().filter(|t| &t.task_id == tid).cloned().collect()
        } else if let Some(idx) = cf.index {
            tasks.iter().filter(|t| t.index == idx).cloned().collect()
        } else {
            tasks.iter().cloned().collect()
        };
        drop(tasks);
        if selected.is_empty() {
            return Err(format!(
                "continue_from: no matching task in job '{}'",
                cf.job_id
            ));
        }
        // Keep submission order so the resumed job's task indices line up.
        selected.sort_by_key(|t| t.index);
        Ok((job, selected))
    }

    /// Single source task (task-level `continue_from`).
    async fn find_source_task(&self, cf: &ContinueFrom) -> Result<(Arc<Job>, Task), String> {
        let (job, mut selected) = self.select_resume_sources(cf).await?;
        if selected.len() > 1 {
            return Err(format!(
                "continue_from: job '{}' has {} tasks — specify 'task_id' or 'index'",
                cf.job_id,
                selected.len()
            ));
        }
        Ok((job, selected.remove(0)))
    }

    /// Generated tokens of a source task: live snapshot while it runs, else the
    /// persisted result.
    async fn source_output_ids(
        &self,
        source: &Task,
        sample_index: usize,
    ) -> Result<Vec<i64>, String> {
        if let Some(live) = self
            .partial_results
            .get(&source.task_id)
            .map(|e| e.value().clone())
        {
            let snapshot = live.read().await.clone();
            if snapshot.samples.len() > sample_index {
                return Ok(snapshot.samples[sample_index].output_ids.clone());
            }
        }
        if let Some(res) = &source.result {
            return select_sample_ids(res, sample_index);
        }
        Err(format!(
            "continue_from: task '{}' has no generated tokens yet (status={:?})",
            source.task_id, source.status
        ))
    }

    /// Token input a source task was run with: explicit ids, else the persisted
    /// composition of its own continuation, else `None` (= text prompt).
    fn source_input_ids(&self, job_id: &str, source: &Task) -> Option<Vec<i64>> {
        if let Some(ids) = &source.request.input_ids {
            return Some(ids.clone());
        }
        self.load_task_input(job_id, source.index)
    }

    /// Resolve a task's exact token input. `None` means "plain text prompt" —
    /// the engine tokenizes it as usual.
    async fn resolve_input_ids(&self, req: &TaskRequest) -> Result<Option<Vec<i64>>, String> {
        if let Some(ids) = &req.input_ids {
            return Ok(Some(ids.clone()));
        }
        let Some(cf) = &req.continue_from else {
            return Ok(None);
        };
        let (job, source) = self.find_source_task(cf).await?;
        let generated = self
            .source_output_ids(&source, cf.sample_index.unwrap_or(0))
            .await?;
        let base = match self.source_input_ids(&job.job_id, &source) {
            Some(ids) => ids,
            None if !source.request.prompt.trim().is_empty() => {
                self.tokenize_prompt(&source.request.prompt).await?
            }
            None => {
                return Err(format!(
                    "continue_from: task '{}' has no recoverable input tokens \
                     (its composed input was not persisted)",
                    source.task_id
                ))
            }
        };
        Ok(Some(compose_input_ids(&base, &generated)))
    }

    /// A continuation inherits the source adapter + sampling params unless the
    /// client overrides them, so a bare `continue_from` reruns the same setup.
    async fn apply_continuation_defaults(&self, req: &mut TaskRequest) -> Result<(), String> {
        let Some(cf) = req.continue_from.clone() else {
            return Ok(());
        };
        let (_job, source) = self.find_source_task(&cf).await?;
        if req.lora_path.is_none() {
            req.lora_path = source.request.lora_path.clone();
        }
        if req.temperature.is_none() {
            req.temperature = source.request.temperature;
        }
        if req.top_p.is_none() {
            req.top_p = source.request.top_p;
        }
        Ok(())
    }

    // -- persistence ---------------------------------------------------------

    fn persist_job(&self, job: &Job) {
        let dir = self.job_dir(&job.job_id);
        if std::fs::create_dir_all(&dir).is_err() {
            tracing::warn!("jobs: cannot create dir {}", dir.display());
            return;
        }
        // Snapshot tasks synchronously is not possible with async lock; use
        // try_read and fall back to skipping (task files are the source of
        // truth for results; job.json is metadata + requests).
        let tasks_snapshot = job.tasks.try_read().ok().map(|g| g.clone());
        if let Some(tasks) = tasks_snapshot {
            let doc = json!({
                "job_id": job.job_id,
                "created_at_unix": job.created_at_unix,
                "tasks": tasks,
            });
            atomic_write(&dir.join("job.json"), &doc);
        }
    }

    fn persist_task_result(&self, job_id: &str, result: &TaskResult) {
        let dir = self.job_dir(job_id);
        let _ = std::fs::create_dir_all(&dir);
        atomic_write(
            &dir.join(format!("task_{:04}.json", result.index)),
            &serde_json::to_value(result).unwrap_or(Value::Null),
        );
    }

    /// Rebuild in-memory state from disk after a restart. Running/queued tasks
    /// are marked failed (interrupted); completed tasks are restored.
    pub fn recover_from_disk(&self) {
        let Ok(entries) = std::fs::read_dir(&self.data_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let job_file = entry.path().join("job.json");
            let Ok(raw) = std::fs::read_to_string(&job_file) else {
                continue;
            };
            let Ok(doc) = serde_json::from_str::<Value>(&raw) else {
                continue;
            };
            let Some(job_id) = doc["job_id"].as_str().map(|s| s.to_string()) else {
                continue;
            };
            let created = doc["created_at_unix"].as_u64().unwrap_or(0);
            let mut tasks: Vec<Task> = Vec::new();
            if let Some(arr) = doc["tasks"].as_array() {
                for t in arr {
                    if let Ok(mut task) = serde_json::from_value::<Task>(t.clone()) {
                        // completed/cancelled tasks whose result file exists
                        // stay in their terminal state with result restored.
                        if task.status == TaskStatus::Completed
                            || task.status == TaskStatus::Cancelled
                        {
                            let rf = self
                                .job_dir(&job_id)
                                .join(format!("task_{:04}.json", task.index));
                            if rf.exists() {
                                if let Ok(r) = std::fs::read_to_string(&rf) {
                                    if let Ok(tr) = serde_json::from_str::<TaskResult>(&r) {
                                        task.result = Some(tr);
                                    }
                                }
                            }
                        } else {
                            task.status = TaskStatus::Failed;
                            task.error = Some("interrupted by router restart".into());
                        }
                        tasks.push(task);
                    }
                }
            }
            let job = Arc::new(Job {
                job_id: job_id.clone(),
                created_at_unix: created,
                tasks: Arc::new(RwLock::new(tasks)),
            });
            tracing::info!("jobs: recovered job {} from disk", job.job_id);
            self.jobs.insert(job_id, job);
        }
    }

    // -- submission ----------------------------------------------------------

    /// Accepts:
    ///   * a raw JSON array of task requests,
    ///   * an object `{ "lora_path": optional, "requests": [...] }` (job-level
    ///     lora as the default, overridable per task),
    ///   * a job-level resume `{ "continue_from": {...}, ... }` which rebuilds
    ///     the source job's task list, each task continuing from its own output.
    pub async fn submit(self: &Arc<Self>, body: Value) -> Result<Arc<Job>, String> {
        let (job_lora, mut requests, resume) = parse_submit_body(&body)?;
        if let Some((cf, ov)) = resume {
            let (source_job, sources) = self.select_resume_sources(&cf).await?;
            let mut expanded = Vec::with_capacity(sources.len());
            for src in sources {
                expanded.push(TaskRequest {
                    continue_from: Some(ContinueFrom {
                        job_id: source_job.job_id.clone(),
                        task_id: Some(src.task_id.clone()),
                        index: None,
                        sample_index: cf.sample_index,
                    }),
                    max_tokens: ov.max_tokens,
                    temperature: ov.temperature,
                    top_p: ov.top_p,
                    lora_path: ov.lora_path.clone(),
                    n: ov.n.unwrap_or(1),
                    ..Default::default()
                });
            }
            requests = expanded;
        }
        if requests.is_empty() {
            return Err("empty task array".into());
        }
        if requests.len() > 4096 {
            return Err("too many tasks in one job (max 4096)".into());
        }
        for r in requests.iter_mut() {
            if r.lora_path.is_none() {
                r.lora_path = job_lora.clone();
            }
            validate_task_input(r)?;
        }
        // Continuations default to the source's adapter + sampling params.
        for r in requests.iter_mut() {
            self.apply_continuation_defaults(r).await?;
        }

        let job_id = self.gen_id("job");
        let mut tasks = Vec::with_capacity(requests.len());
        for (i, mut req) in requests.into_iter().enumerate() {
            if req.lora_path.is_none() {
                req.lora_path = job_lora.clone();
            }
            tasks.push(Task {
                task_id: format!("{job_id}_t{:04}", i),
                index: i,
                request: req,
                status: TaskStatus::Queued,
                error: None,
                result: None,
            });
        }
        let job = Arc::new(Job {
            job_id: job_id.clone(),
            created_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            tasks: Arc::new(RwLock::new(tasks)),
        });
        self.jobs.insert(job_id.clone(), job.clone());
        self.persist_job(&job);
        tracing::info!(
            "jobs: submitted {} ({} tasks)",
            job.job_id,
            job.tasks.read().await.len()
        );

        // Spawn execution for every task (each acquires a concurrency permit).
        let mut spawns = Vec::new();
        {
            let guard = job.tasks.read().await;
            for t in guard.iter() {
                spawns.push((t.task_id.clone(), t.index, t.request.clone()));
            }
        }
        for (task_id, index, req) in spawns {
            let mgr = Arc::clone(self);
            let job_ref = Arc::clone(&job);
            tokio::spawn(async move {
                mgr.execute_task(job_ref, task_id, index, req).await;
            });
        }
        Ok(job)
    }

    // -- execution -----------------------------------------------------------

    async fn set_task_status(&self, job: &Arc<Job>, task_id: &str, status: TaskStatus) {
        let mut tasks = job.tasks.write().await;
        if let Some(t) = tasks.iter_mut().find(|t| t.task_id == task_id) {
            t.status = status;
        }
    }

    async fn execute_task(&self, job: Arc<Job>, task_id: String, index: usize, req: TaskRequest) {
        let _permit = match self.semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => return,
        };
        // If this task was cancelled while it was still queued, the cancel
        // handler already marked it Cancelled — do not start (or overwrite).
        {
            let tasks = job.tasks.read().await;
            if let Some(t) = tasks.iter().find(|t| t.task_id == task_id) {
                if t.status == TaskStatus::Cancelled {
                    return;
                }
            }
        }
        // Pre-flight the adapter before anything is registered. In PD an
        // adapter that is missing on any engine makes the request hang in
        // KVPoll.Bootstrapping until the 600s bootstrap timeout (the decode
        // engine cannot allocate KV / send its KV indices until it has the
        // adapter, and the engines no longer reload implicitly inside a
        // request). Fail the task immediately with the exact remediation
        // instead of burning the timeout.
        if let Some(lora_path) = req.lora_path.clone() {
            if let Err(err) = self.ensure_lora_loaded(&lora_path).await {
                self.fail_task_preflight(&job, &task_id, err).await;
                return;
            }
        }

        // Resolve the exact token input before anything is registered: a
        // `continue_from` task may need an engine tokenizer call, and a failure
        // here must fail the task without producing a (misleading) sample.
        let input_ids = match self.resolve_input_ids(&req).await {
            Ok(ids) => ids,
            Err(err) => {
                self.fail_task_preflight(&job, &task_id, err).await;
                return;
            }
        };
        if req.continue_from.is_some() {
            if let Some(ids) = input_ids.as_ref() {
                // Persist the composed ids: continuing *this* task later must
                // not have to re-derive them from a source that may have moved on.
                self.persist_task_input(&job.job_id, index, ids);
            }
        }
        let body = build_generate_body(&req, input_ids.as_deref());

        // Register live cancel flag + token progress before the run so the
        // cancel endpoint and status polling can see them immediately.
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(AtomicU64::new(0));
        self.cancel_flags.insert(task_id.clone(), cancel.clone());
        self.progress_tokens.insert(task_id.clone(), progress.clone());
        // Live partial result: download endpoints may read this while the task
        // is still running to return "output so far" (completed samples + the
        // in-flight sample snapshot). Removed when the task finishes.
        let live = Arc::new(tokio::sync::RwLock::new(TaskResult {
            task_id: task_id.clone(),
            index,
            samples: Vec::new(),
        }));
        self.partial_results.insert(task_id.clone(), live.clone());

        self.set_task_status(&job, &task_id, TaskStatus::Running)
            .await;

        let n = req.sanitize_n();
        let mut samples = Vec::with_capacity(n as usize);
        let mut first_err: Option<String> = None;

        for _ in 0..n {
            if cancel.load(Ordering::Relaxed) {
                break; // cancelled between samples: keep what we have
            }
            // Reserve the live slot for this sample so a mid-flight download
            // sees the partial snapshot (updated by aggregate_sse).
            {
                let mut g = live.write().await;
                g.samples.push(SampleResult::default());
            }
            match self
                .run_one_sample(&body, &cancel, &progress, Some(&live))
                .await
            {
                Ok(s) => {
                    {
                        let mut g = live.write().await;
                        if let Some(cur) = g.samples.last_mut() {
                            *cur = s.clone();
                        }
                    }
                    samples.push(s);
                }
                Err(e) => {
                    if cancel.load(Ordering::Relaxed) {
                        break; // cancelled mid-sample: keep partial samples
                    }
                    first_err.get_or_insert(e);
                    break; // stop this task's remaining samples on error
                }
            }
        }

        let cancelled = cancel.load(Ordering::Relaxed);
        if let Some(err) = first_err {
            if samples.is_empty() && !cancelled {
                let mut tasks = job.tasks.write().await;
                if let Some(t) = tasks.iter_mut().find(|t| t.task_id == task_id) {
                    t.status = TaskStatus::Failed;
                    t.error = Some(err.clone());
                }
                drop(tasks);
                tracing::warn!("jobs: task {} failed: {}", task_id, err);
                self.cancel_flags.remove(&task_id);
                self.progress_tokens.remove(&task_id);
                self.partial_results.remove(&task_id);
                return;
            }
            // partial samples (or cancelled mid-flight): keep them
            tracing::warn!(
                "jobs: task {} partial failure (cancelled={}): {}",
                task_id,
                cancelled,
                err
            );
        }

        // A cancelled task marks its samples so downstream consumers can tell a
        // partial (cancelled) result from a naturally finished one.
        if cancelled {
            for s in samples.iter_mut() {
                s.finish_reason = Some("cancelled".to_string());
            }
        }

        let result = TaskResult {
            task_id: task_id.clone(),
            index,
            samples,
        };
        // A cancelled job keeps the same downloadable result shape as a
        // completed one — persist whatever samples were generated before the
        // cancel so clients can download partial tokens + logprobs.
        let finished = !result.samples.is_empty() || cancelled;
        self.persist_task_result(&job.job_id, &result);
        {
            let mut tasks = job.tasks.write().await;
            if let Some(t) = tasks.iter_mut().find(|t| t.task_id == task_id) {
                if cancelled {
                    t.status = TaskStatus::Cancelled;
                    t.result = Some(result);
                } else if finished {
                    t.status = TaskStatus::Completed;
                    t.result = Some(result);
                } else {
                    t.status = TaskStatus::Failed;
                    t.error = Some("no samples produced".into());
                }
            }
        }
        // Refresh job.json so cancel state survives a router restart.
        self.persist_job(&job);
        tracing::info!(
            "jobs: task {} done (cancelled={}, finished={})",
            task_id,
            cancelled,
            finished
        );
        self.cancel_flags.remove(&task_id);
        self.progress_tokens.remove(&task_id);
        self.partial_results.remove(&task_id);
    }

    /// Convert one OpenAI-style request to native /generate and aggregate the
    /// SSE stream into a single sample result. `body` is the prebuilt native
    /// /generate body (see `build_generate_body`); `cancel` (when set) stops the
    /// stream at the next chunk boundary and returns whatever was aggregated so
    /// far; `progress` is updated with the running generated-token count;
    /// `live` (task-level partial result) receives periodic snapshots of the
    /// in-flight sample so download endpoints can stream partial output.
    async fn run_one_sample(
        &self,
        body: &Value,
        cancel: &Arc<AtomicBool>,
        progress: &Arc<AtomicU64>,
        live: Option<&Arc<tokio::sync::RwLock<TaskResult>>>,
    ) -> Result<SampleResult, String> {
        let url = format!("{}/generate", self.self_base_url.trim_end_matches('/'));
        let mut builder = self
            .client
            .post(&url)
            // No request-level total deadline: a long generation that keeps
            // producing SSE chunks must not be cut off at a fixed wall-clock
            // limit (e.g. 3600s) just because it took longer than that. The
            // only time-based guard is the per-chunk idle timeout in
            // `aggregate_sse` (no-new-data window), so a live stream runs
            // indefinitely while the engine keeps emitting tokens.
            .json(body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder
            .send()
            .await
            .map_err(|e| format!("generate request failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("generate returned {status}: {}", truncate(&text, 500)));
        }

        let agg = aggregate_sse(resp, cancel, progress, live, self.request_timeout).await?;
        Ok(agg)
    }

    // -- queries -------------------------------------------------------------

    pub fn get_job(&self, job_id: &str) -> Option<Arc<Job>> {
        self.jobs.get(job_id).map(|e| e.value().clone())
    }

    pub fn remove_job(&self, job_id: &str) -> bool {
        let removed = self.jobs.remove(job_id).is_some();
        if removed {
            // Clean up live per-task flags/progress for this job.
            let job = self.jobs.get(job_id).map(|e| e.value().clone());
            if let Some(job) = job {
                let tasks = job.tasks.try_read();
                if let Ok(tasks) = tasks {
                    for t in tasks.iter() {
                        self.cancel_flags.remove(&t.task_id);
                        self.progress_tokens.remove(&t.task_id);
                        self.partial_results.remove(&t.task_id);
                    }
                }
            }
            let dir = self.job_dir(job_id);
            let _ = std::fs::remove_dir_all(dir);
        }
        removed
    }

    // -- garbage collection --------------------------------------------------

    /// Delete job directories whose last activity is older than the retention
    /// window.
    ///
    /// Only **finished** jobs are eligible: a job that is still queued or
    /// running is never removed based on age alone (its files are rewritten as
    /// it progresses, so in practice a live job is also never "old" — this
    /// check is the belt to that suspenders). Jobs that were recovered from
    /// disk without a live entry are treated as finished.
    ///
    /// Returns the number of jobs removed.
    pub async fn gc_once(&self) -> usize {
        let Some(retention) = self.retention else {
            return 0; // GC disabled
        };
        let Ok(entries) = std::fs::read_dir(&self.data_dir) else {
            return 0;
        };
        let now = SystemTime::now();
        let mut removed = 0usize;
        let mut freed = 0u64;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(job_id) = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_string)
            else {
                continue;
            };
            // Last activity = newest mtime among the job's files: job.json is
            // rewritten on every state change and task files land as tasks
            // finish, so this tracks real progress.
            let Some(last) = newest_mtime(&path) else {
                continue;
            };
            let Ok(age) = now.duration_since(last) else {
                continue; // future mtime (clock skew): keep it
            };
            if age < retention {
                continue;
            }
            // Clone the Arc out of the map so no DashMap guard is held across
            // the await below.
            let tracked = self.jobs.get(&job_id).map(|e| e.value().clone());
            if let Some(job) = &tracked {
                let status = job.aggregate_status().await;
                if status == "running" || status == "queued" {
                    continue;
                }
            }
            let bytes = dir_size(&path);
            if tracked.is_some() {
                let _ = self.remove_job(&job_id);
            } else {
                let _ = std::fs::remove_dir_all(&path);
            }
            tracing::info!(
                "jobs: gc removed {} (idle {:.1}h > retention {:.1}h, freed {:.1} MB)",
                job_id,
                age.as_secs_f64() / 3600.0,
                retention.as_secs_f64() / 3600.0,
                bytes as f64 / 1e6
            );
            removed += 1;
            freed += bytes;
        }
        if removed > 0 {
            tracing::info!(
                "jobs: gc pass removed {} job(s), freed {:.1} MB",
                removed,
                freed as f64 / 1e6
            );
        }
        removed
    }

    /// Start the background GC loop (no-op when retention is disabled).
    ///
    /// The first tick fires immediately so a router restart reclaims space that
    /// accumulated while it was down.
    pub fn spawn_gc_task(self: &Arc<Self>) {
        let Some(retention) = self.retention else {
            tracing::info!("jobs: gc disabled (SMG_JOBS_RETENTION_HOURS=0)");
            return;
        };
        let interval = gc_interval_from_env();
        let me = self.clone();
        tokio::spawn(async move {
            tracing::info!(
                "jobs: gc enabled (retention {:.1}h, idempotent scan every {}s)",
                retention.as_secs_f64() / 3600.0,
                interval.as_secs()
            );
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                let _ = me.gc_once().await;
            }
        });
    }

    pub fn list_jobs(&self) -> Vec<Arc<Job>> {
        self.jobs.iter().map(|e| e.value().clone()).collect()
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() > max {
        &s[..max]
    } else {
        s
    }
}

/// Snapshot of the current aggregation state, used to serve "output so far"
/// for a running task. Derives the aligned per-token logprob views exactly
/// like the final result does (so a mid-flight download and the completed
/// result share the same shape).
fn snapshot_live_sample(text: &str, ids: &[i64], entries: &[Value]) -> SampleResult {
    let lps: Vec<f64> = entries
        .iter()
        .map(|e| e.get(0).and_then(|v| v.as_f64()).unwrap_or_default())
        .collect();
    let tops: Vec<Value> = entries
        .iter()
        .map(|e| e.get(2).cloned().unwrap_or(Value::Null))
        .collect();
    SampleResult {
        output_text: text.to_string(),
        output_ids: ids.to_vec(),
        output_token_logprobs: lps,
        output_logprob_entries: entries.to_vec(),
        output_top_logprobs: tops,
        finish_reason: None,
        prompt_tokens: None,
        completion_tokens: Some(ids.len() as u64),
    }
}

/// Newest mtime among the files directly inside `dir`, falling back to the
/// directory's own mtime. `None` when nothing is stat-able.
///
/// Used as "last activity": `job.json` is rewritten on every state change and
/// `task_*.json` lands as each task finishes, so this tracks real progress.
fn newest_mtime(dir: &Path) -> Option<SystemTime> {
    let mut newest: Option<SystemTime> = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if let Ok(t) = e.metadata().and_then(|m| m.modified()) {
                newest = Some(match newest {
                    Some(prev) if prev > t => prev,
                    _ => t,
                });
            }
        }
    }
    // Fall back to the directory's own mtime when it holds no files yet.
    newest.or_else(|| std::fs::metadata(dir).and_then(|m| m.modified()).ok())
}

/// Recursive byte size of a directory (GC accounting / logging).
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            match e.metadata() {
                Ok(md) if md.is_dir() => total += dir_size(&e.path()),
                Ok(md) => total += md.len(),
                Err(_) => {}
            }
        }
    }
    total
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

fn atomic_write(path: &Path, doc: &Value) {
    let tmp = path.with_extension("json.tmp");
    if let Ok(text) = serde_json::to_string(doc) {
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

// ---------------------------------------------------------------------------
// SSE aggregation (dual-mode safe: handles both incremental and cumulative
// chunks, so it works regardless of the backend's chunking convention).
// ---------------------------------------------------------------------------

async fn aggregate_sse(
    resp: reqwest::Response,
    cancel: &std::sync::atomic::AtomicBool,
    progress: &std::sync::atomic::AtomicU64,
    live: Option<&Arc<tokio::sync::RwLock<TaskResult>>>,
    idle_timeout: Duration,
) -> Result<SampleResult, String> {
    let mut stream = resp.bytes_stream();
    use futures_util::StreamExt;

    let mut buf: Vec<u8> = Vec::with_capacity(16 * 1024);
    let mut text = String::new();
    let mut ids: Vec<i64> = Vec::new();
    // Raw [logprob, token_id, top_k] triples, cumulative per chunk.
    let mut entries: Vec<Value> = Vec::new();
    let mut final_meta: Option<Value> = None;
    let mut saw_done = false;
    let mut cancelled = false;
    let mut last_live_snap: Option<std::time::Instant> = None;

    // Poll cancel every ~150ms so a long gap between SSE chunks (or a
    // stalled upstream) still reacts promptly to a job-cancel request.
    let mut cancel_poll = tokio::time::interval(Duration::from_millis(150));
    cancel_poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        if cancel.load(Ordering::Relaxed) {
            cancelled = true;
            break; // keep whatever was aggregated before the cancel
        }
        // No-new-data idle timeout guards a genuinely stalled upstream; cancel
        // poll (150ms) still wins via the select below so cancels react
        // promptly.  As long as the engine keeps emitting chunks, this timer
        // resets every chunk — a live stream never hits a wall-clock deadline.
        let next_chunk = tokio::time::timeout(idle_timeout, stream.next());
        tokio::pin!(next_chunk);
        let chunk = tokio::select! {
            c = &mut next_chunk => match c {
                Ok(c) => c,
                Err(_) => return Err(format!("SSE idle timeout ({}s without data)", idle_timeout.as_secs())),
            },
            _ = cancel_poll.tick() => {
                if cancel.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
                continue;
            }
        };
        match chunk {
            Some(Ok(bytes)) => {
                buf.extend_from_slice(&bytes);
                // process complete lines
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line).trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    if let Some(data) = line.strip_prefix("data:") {
                        let data = data.trim();
                        if data == "[DONE]" {
                            saw_done = true;
                            continue;
                        }
                        let Ok(v) = serde_json::from_str::<Value>(data) else {
                            continue;
                        };
                        process_chunk(&v, &mut text, &mut ids, &mut entries, &mut final_meta);
                        progress.store(ids.len() as u64, Ordering::Relaxed);
                        // Throttled live snapshot: expose "output so far" for a
                        // running task (~every 500ms) without hot-loop writes.
                        if let Some(live) = live {
                            let now = std::time::Instant::now();
                            if last_live_snap
                                .map(|t: std::time::Instant| now.duration_since(t) >= Duration::from_millis(500))
                                .unwrap_or(true)
                            {
                                let snap = snapshot_live_sample(&text, &ids, &entries);
                                let mut g = live.write().await;
                                if let Some(cur) = g.samples.last_mut() {
                                    *cur = snap;
                                }
                                last_live_snap = Some(now);
                            }
                        }
                    }
                }
            }
            Some(Err(e)) => {
                // Stream broken mid-flight. If we already have tokens, treat as
                // complete-with-what-we-have only if final_meta existed.
                if final_meta.is_some() && !ids.is_empty() {
                    tracing::warn!("jobs: SSE stream ended with error after data: {e}");
                    break;
                }
                return Err(format!("SSE stream error: {e}"));
            }
            None => break, // stream ended
        }
        if saw_done {
            break;
        }
    }

    if ids.is_empty() && text.is_empty() && final_meta.is_none() {
        return Err("empty SSE response".into());
    }
    // Cross-fill: logprob triples carry token ids too — use them if the
    // top-level output_ids were absent.
    if ids.is_empty() {
        ids = entries
            .iter()
            .filter_map(|e| e.get(1).and_then(|v| v.as_i64()))
            .collect();
    }
    if ids.is_empty() && final_meta.is_none() {
        return Err("no output tokens in SSE response".into());
    }

    // Derive aligned views from the raw triples.
    let lps: Vec<f64> = entries
        .iter()
        .map(|e| e.get(0).and_then(|v| v.as_f64()).unwrap_or_default())
        .collect();
    let tops: Vec<Value> = entries
        .iter()
        .map(|e| e.get(2).cloned().unwrap_or(Value::Null))
        .collect();

    let mut out = SampleResult {
        output_text: text,
        output_ids: ids,
        output_token_logprobs: lps,
        output_logprob_entries: entries,
        output_top_logprobs: tops,
        ..Default::default()
    };
    if let Some(meta) = &final_meta {
        out.finish_reason = meta["finish_reason"]
            .as_object()
            .and_then(|o| o.get("type"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                meta["finish_reason"]
                    .as_str()
                    .map(|s| s.to_string())
            });
        out.prompt_tokens = meta["prompt_tokens"].as_u64();
        out.completion_tokens = meta["completion_tokens"].as_u64();
    } else if cancelled {
        // A cancel cut the stream before the final meta arrived: surface it so
        // clients can tell a partial (cancelled) sample from a truncated one.
        out.finish_reason = Some("cancelled".to_string());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn process_chunk(
    v: &Value,
    text: &mut String,
    ids: &mut Vec<i64>,
    entries: &mut Vec<Value>,
    final_meta: &mut Option<Value>,
) {
    if let Some(t) = v["text"].as_str() {
        merge_str(text, t);
    }
    // Output ids: chunk top-level (cumulative in sglang streaming) or meta.
    let chunk_ids: Vec<i64> = extract_ids(v.get("output_ids"))
        .or_else(|| extract_ids(v["meta_info"].get("output_ids")))
        .unwrap_or_default();
    merge_ids(ids, &chunk_ids);

    if let Some(meta) = v.get("meta_info") {
        if meta.get("finish_reason").is_some() {
            *final_meta = Some(meta.clone());
        }
        // sglang native format: output_token_logprobs is a list of
        // [logprob, token_id, top_logprobs|null] triples, cumulative per
        // streaming chunk.
        if let Some(arr) = meta["output_token_logprobs"].as_array() {
            merge_values(entries, arr);
        }
    }
}

fn extract_ids(v: Option<&Value>) -> Option<Vec<i64>> {
    match v {
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for x in arr {
                match x {
                    Value::Number(n) => out.push(n.as_i64().unwrap_or(0)),
                    Value::Array(nested) => {
                        // batched form: take last (or first) inner array
                        let inner: Vec<i64> = nested
                            .iter()
                            .filter_map(|y| y.as_i64())
                            .collect();
                        out.extend(inner);
                    }
                    _ => {}
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// If `incoming` starts with the current accumulated content and is longer,
/// the backend sends cumulative data -> replace. Otherwise append.
fn merge_str(acc: &mut String, incoming: &str) {
    if incoming.is_empty() {
        return;
    }
    if incoming.len() > acc.len() && incoming.starts_with(acc.as_str()) {
        *acc = incoming.to_string();
    } else if !incoming.starts_with(acc.as_str()) || incoming.len() != acc.len() {
        acc.push_str(incoming);
    }
}

fn merge_ids(acc: &mut Vec<i64>, incoming: &[i64]) {
    if incoming.is_empty() {
        return;
    }
    if incoming.len() > acc.len() && incoming.starts_with(&acc[..]) {
        *acc = incoming.to_vec();
    } else {
        acc.extend_from_slice(incoming);
    }
}

/// Merge raw JSON arrays supporting both cumulative and incremental chunk
/// conventions: if `incoming` extends the accumulated prefix, replace;
/// otherwise append.
fn merge_values(acc: &mut Vec<Value>, incoming: &[Value]) {
    if incoming.is_empty() {
        return;
    }
    if incoming.len() > acc.len() {
        let prefix_eq = incoming.iter().zip(acc.iter()).all(|(a, b)| a == b);
        if prefix_eq {
            *acc = incoming.to_vec();
            return;
        }
    }
    acc.extend_from_slice(incoming);
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

fn err(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({"error": msg}))).into_response()
}

pub async fn submit_jobs(
    State(mgr): State<Arc<JobManager>>,
    body: axum::body::Bytes,
) -> Response {
    let parsed: Result<Value, _> = serde_json::from_slice(&body);
    let Ok(value) = parsed else {
        return err(StatusCode::BAD_REQUEST, "invalid JSON body");
    };
    match mgr.submit(value).await {
        Ok(job) => {
            let tasks: Vec<Value> = job
                .tasks
                .read()
                .await
                .iter()
                .map(|t| {
                    json!({
                        "task_id": t.task_id,
                        "index": t.index,
                        "status": t.status,
                        "lora_path": t.request.lora_path,
                        "prompt_chars": t.request.prompt.len(),
                    })
                })
                .collect();
            (
                StatusCode::ACCEPTED,
                Json(json!({
                    "job_id": job.job_id,
                    "status_url": format!("/v1/control/jobs/{}", job.job_id),
                    "result_url": format!("/v1/control/jobs/{}/result", job.job_id),
                    "task_count": tasks.len(),
                    "tasks": tasks,
                })),
            )
                .into_response()
        }
        Err(e) => err(StatusCode::BAD_REQUEST, &e),
    }
}

pub async fn get_job_status(
    State(mgr): State<Arc<JobManager>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    let Some(job) = mgr.get_job(&job_id) else {
        return err(StatusCode::NOT_FOUND, "job not found");
    };
    let status = job.aggregate_status().await;
    let tasks = job.tasks.read().await;
    let mut done = 0;
    let mut failed = 0;
    let mut cancelled = 0;
    let mut queued = 0;
    let mut running = 0;
    let mut total_tokens = 0u64;
    let mut task_views = Vec::with_capacity(tasks.len());
    for t in tasks.iter() {
        match t.status {
            TaskStatus::Completed => done += 1,
            TaskStatus::Failed => failed += 1,
            TaskStatus::Cancelled => cancelled += 1,
            TaskStatus::Queued => queued += 1,
            TaskStatus::Running => running += 1,
        }
        let mut token_count = 0u64;
        if let Some(r) = &t.result {
            for s in &r.samples {
                token_count += s.output_ids.len() as u64;
            }
        } else {
            // running task: report live generated-token progress
            if let Some(p) = mgr.progress_tokens.get(&t.task_id) {
                token_count = p.load(Ordering::Relaxed);
            }
        }
        total_tokens += token_count;
        task_views.push(json!({
            "task_id": t.task_id,
            "index": t.index,
            "status": t.status,
            "token_count": token_count,
            "error": t.error,
        }));
    }
    Json(json!({
        "job_id": job.job_id,
        "status": status,
        "progress": {
            "done": done,
            "failed": failed,
            "cancelled": cancelled,
            "running": running,
            "queued": queued,
            "total": tasks.len(),
        },
        "total_output_tokens": total_tokens,
        "tasks": task_views,
    }))
    .into_response()
}

pub async fn get_job_result(
    State(mgr): State<Arc<JobManager>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    let Some(job) = mgr.get_job(&job_id) else {
        return err(StatusCode::NOT_FOUND, "job not found");
    };
    let status = job.aggregate_status().await;
    // Allow download at any time: completed tasks contribute their persisted
    // result, running tasks contribute their live partial snapshot (output
    // text + token ids + logprobs generated so far). The `status` field tells
    // the client whether this is final (completed/partial/cancelled/failed) or
    // still in flight.
    let tasks = job.tasks.read().await;
    let mut results: Vec<TaskResult> = Vec::new();
    for t in tasks.iter() {
        if let Some(r) = &t.result {
            results.push(r.clone());
        } else if let Some(live) = mgr.partial_results.get(&t.task_id) {
            let snap = live.read().await;
            let any = snap
                .samples
                .iter()
                .any(|s| !s.output_ids.is_empty() || !s.output_text.is_empty());
            if any {
                results.push(snap.clone());
            }
        }
    }
    Json(json!({
        "job_id": job.job_id,
        "status": status,
        "task_count": tasks.len(),
        "results": results,
    }))
    .into_response()
}

pub async fn get_task_result(
    State(mgr): State<Arc<JobManager>>,
    AxumPath((job_id, task_id)): AxumPath<(String, String)>,
) -> Response {
    let Some(job) = mgr.get_job(&job_id) else {
        return err(StatusCode::NOT_FOUND, "job not found");
    };
    let tasks = job.tasks.read().await;
    let Some(t) = tasks.iter().find(|t| t.task_id == task_id) else {
        return err(StatusCode::NOT_FOUND, "task not found");
    };
    match &t.result {
        Some(r) => Json(r.clone()).into_response(),
        None => {
            // Running task: serve the live partial snapshot (output so far,
            // with token ids + logprobs) instead of a 409.
            if let Some(live) = mgr.partial_results.get(&t.task_id) {
                let snap = live.read().await;
                let any = snap.samples.iter().any(|s| {
                    !s.output_ids.is_empty() || !s.output_text.is_empty()
                });
                if any {
                    return Json(snap.clone()).into_response();
                }
            }
            // A cancelled task that never produced samples still downloads in
            // the same shape as a completed one (empty samples array).
            if t.status == TaskStatus::Cancelled {
                let empty = TaskResult {
                    task_id: task_id.clone(),
                    index: t.index,
                    samples: Vec::new(),
                };
                return Json(empty).into_response();
            }
            err(
                StatusCode::CONFLICT,
                t.error.as_deref().unwrap_or("task not finished"),
            )
        }
    }
}

pub async fn delete_job(
    State(mgr): State<Arc<JobManager>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    if mgr.remove_job(&job_id) {
        (StatusCode::OK, Json(json!({"deleted": job_id}))).into_response()
    } else {
        err(StatusCode::NOT_FOUND, "job not found")
    }
}

/// Cancel a running/queued job. Queued tasks flip to `cancelled` immediately;
/// running tasks set their live cancel flag and stop at the next SSE chunk,
/// persisting whatever tokens (with logprobs) were already generated — the
/// downloadable result shape is identical to a completed job.
pub async fn cancel_job(
    State(mgr): State<Arc<JobManager>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    let Some(job) = mgr.get_job(&job_id) else {
        return err(StatusCode::NOT_FOUND, "job not found");
    };
    // Flip live cancel flags for running tasks so their SSE loop stops at the
    // next chunk boundary and persists partial tokens+logprobs.
    let running_task_ids: Vec<String> = {
        let tasks = job.tasks.read().await;
        tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Running)
            .map(|t| t.task_id.clone())
            .collect()
    };
    for tid in &running_task_ids {
        if let Some(flag) = mgr.cancel_flags.get(tid) {
            flag.value().store(true, Ordering::Relaxed);
        }
    }
    // Queued tasks never started: mark cancelled directly (no result file).
    {
        let mut tasks = job.tasks.write().await;
        for t in tasks.iter_mut() {
            if t.status == TaskStatus::Queued {
                t.status = TaskStatus::Cancelled;
            }
        }
    }
    mgr.persist_job(&job);
    Json(json!({
        "job_id": job.job_id,
        "status": job.aggregate_status().await,
        "running_cancelled": running_task_ids.len(),
    }))
    .into_response()
}

pub async fn list_jobs(State(mgr): State<Arc<JobManager>>) -> Response {
    let mut out = Vec::new();
    for job in mgr.list_jobs() {
        out.push(json!({
            "job_id": job.job_id,
            "status": job.aggregate_status().await,
            "created_at_unix": job.created_at_unix,
        }));
    }
    Json(json!({"jobs": out})).into_response()
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_incremental_and_cumulative() {
        let mut ids = vec![1, 2];
        merge_ids(&mut ids, &[3]);
        assert_eq!(ids, vec![1, 2, 3]);

        let mut ids2 = vec![1, 2];
        merge_ids(&mut ids2, &[1, 2, 3, 4]);
        assert_eq!(ids2, vec![1, 2, 3, 4]);

        let mut text = String::from("hel");
        merge_str(&mut text, "lo");
        assert_eq!(text, "hello");

        let mut text2 = String::from("hel");
        merge_str(&mut text2, "hello world");
        assert_eq!(text2, "hello world");
    }

    #[test]
    fn merge_logprob_triples_cumulative() {
        // sglang streaming sends cumulative [logprob, token_id, top|null]
        // triples per chunk — the aggregator must replace, not append.
        let t1 = serde_json::json!([[-0.7, 124134, null]]);
        let t2 = serde_json::json!([[-0.7, 124134, null], [-2.3, 22, null]]);

        let mut acc: Vec<Value> = Vec::new();
        merge_values(&mut acc, t1.as_array().unwrap());
        assert_eq!(acc.len(), 1);
        merge_values(&mut acc, t2.as_array().unwrap());
        assert_eq!(acc.len(), 2, "cumulative chunk must replace, not append");
        assert_eq!(acc[1][1], serde_json::json!(22));
    }

    #[test]
    fn merge_values_incremental() {
        let a = serde_json::json!([[1.0, 5, null]]);
        let b = serde_json::json!([[2.0, 6, null]]);
        let mut acc: Vec<Value> = Vec::new();
        merge_values(&mut acc, a.as_array().unwrap());
        merge_values(&mut acc, b.as_array().unwrap());
        assert_eq!(acc.len(), 2, "non-prefix data must be appended");
    }

    #[test]
    fn cancelled_status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&TaskStatus::Cancelled).unwrap(),
            "\"cancelled\""
        );
        let back: TaskStatus =
            serde_json::from_str("\"cancelled\"").unwrap();
        assert_eq!(back, TaskStatus::Cancelled);
    }

    fn req_text(prompt: &str) -> TaskRequest {
        TaskRequest {
            prompt: prompt.into(),
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: Some(0.9),
            lora_path: Some("/loras/L0".into()),
            ..Default::default()
        }
    }

    #[test]
    fn build_body_prefers_tokens_over_text() {
        // Text task: body carries the prompt text and no token array.
        let body = build_generate_body(&req_text("hello"), None);
        assert_eq!(body["text"], json!("hello"));
        assert!(body.get("input_ids").is_none());
        assert_eq!(body["sampling_params"]["max_new_tokens"], json!(128));
        assert_eq!(body["sampling_params"]["temperature"], json!(0.7));
        assert_eq!(body["sampling_params"]["top_p"], json!(0.9));
        assert_eq!(body["lora_path"], json!("/loras/L0"));
        assert_eq!(body["return_logprob"], json!(true));
        assert_eq!(body["stream"], json!(true));

        // Continuation: the resolved tokens replace the text entirely, so the
        // engine must not re-tokenize anything.
        let body = build_generate_body(&req_text("ignored"), Some(&[1, 2, 3]));
        assert_eq!(body["input_ids"], json!([1, 2, 3]));
        assert!(body.get("text").is_none());
    }

    #[test]
    fn compose_input_ids_is_source_plus_generated() {
        assert_eq!(compose_input_ids(&[1, 2], &[3, 4]), vec![1, 2, 3, 4]);
        // A cancel at 0 tokens still yields a runnable input (the source prompt).
        assert_eq!(compose_input_ids(&[1, 2], &[]), vec![1, 2]);
    }

    #[test]
    fn select_sample_ids_checks_bounds() {
        let result = TaskResult {
            task_id: "t".into(),
            index: 0,
            samples: vec![
                SampleResult {
                    output_ids: vec![7, 8],
                    ..Default::default()
                },
                SampleResult {
                    output_ids: vec![9],
                    ..Default::default()
                },
            ],
        };
        assert_eq!(select_sample_ids(&result, 0).unwrap(), vec![7, 8]);
        assert_eq!(select_sample_ids(&result, 1).unwrap(), vec![9]);
        let err = select_sample_ids(&result, 2).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn validate_task_input_requires_exactly_one_source() {
        assert!(validate_task_input(&req_text("x")).is_ok());

        let mut ids_only = TaskRequest {
            input_ids: Some(vec![1, 2]),
            ..Default::default()
        };
        assert!(validate_task_input(&ids_only).is_ok());

        let cont_only = TaskRequest {
            continue_from: Some(ContinueFrom {
                job_id: "j".into(),
                task_id: None,
                index: None,
                sample_index: None,
            }),
            ..Default::default()
        };
        assert!(validate_task_input(&cont_only).is_ok());

        // No input at all.
        let empty = TaskRequest::default();
        assert!(validate_task_input(&empty).is_err());

        // Empty token array is not an input.
        ids_only.input_ids = Some(vec![]);
        assert!(validate_task_input(&ids_only).is_err());

        // Two sources at once is ambiguous.
        let both = TaskRequest {
            prompt: "x".into(),
            input_ids: Some(vec![1]),
            ..Default::default()
        };
        assert!(validate_task_input(&both).is_err());
        let mut text_and_cont = req_text("x");
        text_and_cont.continue_from = cont_only.continue_from.clone();
        assert!(validate_task_input(&text_and_cont).is_err());
    }

    #[test]
    fn continue_from_parses_job_and_task_forms() {
        let cf: ContinueFrom =
            serde_json::from_value(json!({"job_id": "job_1"})).unwrap();
        assert_eq!(cf.job_id, "job_1");
        assert!(cf.task_id.is_none());

        let cf: ContinueFrom = serde_json::from_value(
            json!({"job_id": "job_1", "task_id": "job_1_t0002", "sample_index": 1}),
        )
        .unwrap();
        assert_eq!(cf.task_id.as_deref(), Some("job_1_t0002"));
        assert_eq!(cf.sample_index, Some(1));

        // job_id is mandatory.
        assert!(serde_json::from_value::<ContinueFrom>(json!({"task_id": "x"})).is_err());
    }

    #[test]
    fn parse_submit_body_supports_all_shapes() {
        // Array form, unchanged.
        let body = json!([{"prompt": "a", "max_tokens": 16}]);
        let (lora, reqs, resume) = parse_submit_body(&body).unwrap();
        assert!(lora.is_none());
        assert_eq!(reqs.len(), 1);
        assert!(resume.is_none());

        // Object form with job-level lora.
        let body = json!({"lora_path": "/l", "requests": [{"prompt": "a"}]});
        let (lora, reqs, _) = parse_submit_body(&body).unwrap();
        assert_eq!(lora.as_deref(), Some("/l"));
        assert_eq!(reqs.len(), 1);

        // Job-level resume sugar.
        let body = json!({
            "continue_from": {"job_id": "job_1"},
            "max_tokens": 256,
            "temperature": 0.0,
            "lora_path": "/l2",
            "n": 2
        });
        let (lora, reqs, resume) = parse_submit_body(&body).unwrap();
        assert_eq!(lora.as_deref(), Some("/l2"));
        assert!(reqs.is_empty());
        let (cf, ov) = resume.expect("resume spec");
        assert_eq!(cf.job_id, "job_1");
        assert_eq!(ov.max_tokens, Some(256));
        assert_eq!(ov.temperature, Some(0.0));
        assert_eq!(ov.n, Some(2));

        // requests + continue_from is contradictory.
        let body = json!({"continue_from": {"job_id": "j"}, "requests": []});
        assert!(parse_submit_body(&body).is_err());
        // Unknown object shape.
        assert!(parse_submit_body(&json!({"foo": 1})).is_err());
        // Invalid continue_from payload.
        assert!(parse_submit_body(&json!({"continue_from": {"task_id": "x"}})).is_err());
    }

    #[test]
    fn continuation_request_deserializes_without_prompt() {
        // The whole point: a continuation body carries only a reference.
        let req: TaskRequest = serde_json::from_value(json!({
            "continue_from": {"job_id": "job_1", "task_id": "job_1_t0000"},
            "max_tokens": 512
        }))
        .unwrap();
        assert!(req.prompt.is_empty());
        assert!(req.input_ids.is_none());
        assert_eq!(
            req.continue_from.as_ref().unwrap().task_id.as_deref(),
            Some("job_1_t0000")
        );
        assert_eq!(req.max_tokens, Some(512));
        validate_task_input(&req).unwrap();
    }

    #[test]
    fn aggregate_status_counts_cancelled() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mk = |s: TaskStatus| Task {
                task_id: "t".into(),
                index: 0,
                request: TaskRequest {
                    prompt: "x".into(),
                    ..Default::default()
                },
                status: s,
                error: None,
                result: None,
            };
            let job = Job {
                job_id: "j".into(),
                created_at_unix: 0,
                tasks: Arc::new(RwLock::new(vec![
                    mk(TaskStatus::Cancelled),
                    mk(TaskStatus::Cancelled),
                ])),
            };
            assert_eq!(job.aggregate_status().await, "cancelled");
            let job2 = Job {
                job_id: "j2".into(),
                created_at_unix: 0,
                tasks: Arc::new(RwLock::new(vec![
                    mk(TaskStatus::Completed),
                    mk(TaskStatus::Cancelled),
                ])),
            };
            assert_eq!(job2.aggregate_status().await, "partial");
        });
    }

    // -- continuation resolution (incl. pre-feature/historical jobs) ---------

    fn mk_manager(dir: &Path, engine_urls: Vec<String>) -> Arc<JobManager> {
        JobManager::new(
            "http://127.0.0.1:1".to_string(),
            None,
            dir.to_path_buf(),
            4,
            60,
            engine_urls,
        )
    }

    fn sample(ids: Vec<i64>) -> SampleResult {
        SampleResult {
            output_ids: ids,
            ..Default::default()
        }
    }

    fn insert_source_job(
        mgr: &Arc<JobManager>,
        job_id: &str,
        tasks: Vec<Task>,
    ) {
        let job = Arc::new(Job {
            job_id: job_id.to_string(),
            created_at_unix: 0,
            tasks: Arc::new(RwLock::new(tasks)),
        });
        mgr.jobs.insert(job_id.to_string(), job);
    }

    #[tokio::test]
    async fn resolve_continuation_composes_input_and_generated_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = mk_manager(dir.path(), Vec::new());
        insert_source_job(
            &mgr,
            "job_a",
            vec![Task {
                task_id: "job_a_t0000".into(),
                index: 0,
                request: TaskRequest {
                    input_ids: Some(vec![1, 2, 3]),
                    ..Default::default()
                },
                status: TaskStatus::Cancelled,
                error: None,
                result: Some(TaskResult {
                    task_id: "job_a_t0000".into(),
                    index: 0,
                    samples: vec![sample(vec![4, 5])],
                }),
            }],
        );

        let req = TaskRequest {
            continue_from: Some(ContinueFrom {
                job_id: "job_a".into(),
                task_id: None,
                index: None,
                sample_index: None,
            }),
            max_tokens: Some(16),
            ..Default::default()
        };
        let ids = mgr.resolve_input_ids(&req).await.unwrap().unwrap();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);

        // The composed input is persisted, so continuing *this* task later is
        // exact too (no dependency on a source that may have moved on).
        mgr.persist_task_input("job_b", 0, &ids);
        assert_eq!(mgr.load_task_input("job_b", 0).unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn resolve_continuation_uses_selected_sample() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = mk_manager(dir.path(), Vec::new());
        insert_source_job(
            &mgr,
            "job_s",
            vec![Task {
                task_id: "job_s_t0000".into(),
                index: 0,
                request: TaskRequest {
                    input_ids: Some(vec![10]),
                    ..Default::default()
                },
                status: TaskStatus::Completed,
                error: None,
                result: Some(TaskResult {
                    task_id: "job_s_t0000".into(),
                    index: 0,
                    samples: vec![sample(vec![11]), sample(vec![12, 13])],
                }),
            }],
        );
        let mk_req = |sample_index: Option<usize>| TaskRequest {
            continue_from: Some(ContinueFrom {
                job_id: "job_s".into(),
                task_id: Some("job_s_t0000".into()),
                index: None,
                sample_index,
            }),
            ..Default::default()
        };
        assert_eq!(
            mgr.resolve_input_ids(&mk_req(None)).await.unwrap().unwrap(),
            vec![10, 11]
        );
        assert_eq!(
            mgr.resolve_input_ids(&mk_req(Some(1)))
                .await
                .unwrap()
                .unwrap(),
            vec![10, 12, 13]
        );
        let err = mgr.resolve_input_ids(&mk_req(Some(9))).await.unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[tokio::test]
    async fn resolve_continuation_reports_unknown_or_ambiguous_sources() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = mk_manager(dir.path(), Vec::new());
        let mk_src = |task_id: &str, index: usize| Task {
            task_id: task_id.to_string(),
            index,
            request: TaskRequest {
                prompt: "p".into(),
                ..Default::default()
            },
            status: TaskStatus::Completed,
            error: None,
            result: Some(TaskResult {
                task_id: task_id.to_string(),
                index,
                samples: vec![sample(vec![1])],
            }),
        };
        insert_source_job(&mgr, "job_m", vec![mk_src("job_m_t0000", 0), mk_src("job_m_t0001", 1)]);

        // Unknown job id.
        let err = mgr
            .resolve_input_ids(&TaskRequest {
                continue_from: Some(ContinueFrom {
                    job_id: "nope".into(),
                    task_id: None,
                    index: None,
                    sample_index: None,
                }),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(err.contains("unknown job_id"), "{err}");

        // Multi-task job without a selector: must ask for task_id/index.
        // (Its prompt would need an engine, so the selector check comes first.)
        let err = mgr
            .resolve_input_ids(&TaskRequest {
                continue_from: Some(ContinueFrom {
                    job_id: "job_m".into(),
                    task_id: None,
                    index: None,
                    sample_index: None,
                }),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(err.contains("specify 'task_id' or 'index'"), "{err}");

        // Text source with no engine configured → explicit, actionable error.
        let err = mgr
            .resolve_input_ids(&TaskRequest {
                continue_from: Some(ContinueFrom {
                    job_id: "job_m".into(),
                    task_id: Some("job_m_t0000".into()),
                    index: None,
                    sample_index: None,
                }),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(err.contains("no engine URL"), "{err}");
    }

    #[tokio::test]
    async fn legacy_job_files_are_recovered_and_continuable() {
        let dir = tempfile::tempdir().unwrap();
        // Pre-feature on-disk shape: no input_ids / continue_from fields.
        let job_dir = dir.path().join("job_old");
        std::fs::create_dir_all(&job_dir).unwrap();
        std::fs::write(
            job_dir.join("job.json"),
            r#"{"job_id":"job_old","created_at_unix":1,"tasks":[{"task_id":"job_old_t0000","index":0,"request":{"prompt":"legacy prompt","max_tokens":64},"status":"cancelled","error":null,"result":null}]}"#,
        )
        .unwrap();
        std::fs::write(
            job_dir.join("task_0000.json"),
            r#"{"task_id":"job_old_t0000","index":0,"samples":[{"output_text":"hi","output_ids":[7,8],"output_token_logprobs":[-0.1,-0.2],"finish_reason":"cancelled","prompt_tokens":2,"completion_tokens":2}]}"#,
        )
        .unwrap();

        let mgr = mk_manager(dir.path(), Vec::new());
        mgr.recover_from_disk();
        let job = mgr.get_job("job_old").expect("legacy job recovered");
        let tasks = job.tasks.read().await;
        let task = &tasks[0];
        assert!(task.request.input_ids.is_none());
        assert!(task.request.continue_from.is_none());
        assert_eq!(
            select_sample_ids(task.result.as_ref().expect("legacy result"), 0).unwrap(),
            vec![7, 8]
        );
        drop(tasks);

        // A legacy (text) source resolves by tokenizing its prompt on an engine;
        // without one configured the failure says exactly what is missing.
        let err = mgr
            .resolve_input_ids(&TaskRequest {
                continue_from: Some(ContinueFrom {
                    job_id: "job_old".into(),
                    task_id: None,
                    index: None,
                    sample_index: None,
                }),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(err.contains("no engine URL"), "{err}");
    }

    #[tokio::test]
    async fn job_level_resume_expands_one_task_per_source() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = mk_manager(dir.path(), Vec::new());
        let mk_src = |task_id: &str, index: usize| Task {
            task_id: task_id.to_string(),
            index,
            request: TaskRequest {
                input_ids: Some(vec![index as i64 + 1]),
                temperature: Some(0.3),
                top_p: Some(0.8),
                lora_path: Some("/loras/L9".into()),
                ..Default::default()
            },
            status: TaskStatus::Cancelled,
            error: None,
            result: Some(TaskResult {
                task_id: task_id.to_string(),
                index,
                samples: vec![sample(vec![100 + index as i64])],
            }),
        };
        insert_source_job(&mgr, "job_src", vec![mk_src("job_src_t0000", 0), mk_src("job_src_t0001", 1)]);

        let job = mgr
            .submit(json!({
                "continue_from": {"job_id": "job_src"},
                "max_tokens": 32
            }))
            .await
            .expect("resume accepted");
        let tasks = job.tasks.read().await;
        assert_eq!(tasks.len(), 2, "one continuation task per source task");
        for (i, t) in tasks.iter().enumerate() {
            let cf = t.request.continue_from.as_ref().expect("continue_from set");
            assert_eq!(cf.job_id, "job_src");
            assert_eq!(cf.task_id, Some(format!("job_src_t{i:04}")));
            assert_eq!(t.request.max_tokens, Some(32));
            // Sampling params + adapter are inherited from the source task.
            assert_eq!(t.request.temperature, Some(0.3));
            assert_eq!(t.request.top_p, Some(0.8));
            assert_eq!(t.request.lora_path.as_deref(), Some("/loras/L9"));
            assert_eq!(t.index, i);
        }
    // -- garbage collection --------------------------------------------------

    fn test_request() -> TaskRequest {
        TaskRequest {
            prompt: "x".into(),
            max_tokens: Some(8),
            temperature: None,
            top_p: None,
            n: 1,
            lora_path: None,
            model: None,
            stream: None,
            logprobs: None,
            stream_options: None,
        }
    }

    /// Write a minimal but valid on-disk job (job.json) for `job_id`.
    fn write_job_dir(root: &Path, job_id: &str, task_status: &str) -> std::path::PathBuf {
        let dir = root.join(job_id);
        std::fs::create_dir_all(&dir).unwrap();
        let doc = json!({
            "job_id": job_id,
            "created_at_unix": 1,
            "tasks": [{
                "task_id": format!("{job_id}_t0000"),
                "index": 0,
                "request": {"prompt": "x", "max_tokens": 8},
                "status": task_status,
            }],
        });
        std::fs::write(dir.join("job.json"), serde_json::to_string(&doc).unwrap()).unwrap();
        dir
    }

    fn age_file(path: &Path, age: Duration) {
        let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        f.set_modified(SystemTime::now() - age).unwrap();
    }

    fn gc_manager(root: &Path, retention: Option<Duration>) -> Arc<JobManager> {
        JobManager::build(
            "http://127.0.0.1:1".to_string(),
            None,
            root.to_path_buf(),
            4,
            60,
            retention,
        )
    }

    #[tokio::test]
    async fn gc_removes_expired_finished_jobs_only() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let old_done = write_job_dir(root, "job_old_done", "completed");
        let old_running = write_job_dir(root, "job_old_running", "running");
        let fresh = write_job_dir(root, "job_fresh", "completed");

        let three_days = Duration::from_secs(72 * 3600);
        age_file(&old_done.join("job.json"), three_days);
        age_file(&old_running.join("job.json"), three_days);

        let mgr = gc_manager(root, Some(Duration::from_secs(48 * 3600)));
        // The running job is still tracked in memory (a live job must survive
        // regardless of how old its files look).
        mgr.jobs.insert(
            "job_old_running".to_string(),
            Arc::new(Job {
                job_id: "job_old_running".to_string(),
                created_at_unix: 1,
                tasks: Arc::new(RwLock::new(vec![Task {
                    task_id: "job_old_running_t0000".to_string(),
                    index: 0,
                    request: test_request(),
                    status: TaskStatus::Running,
                    error: None,
                    result: None,
                }])),
            }),
        );

        assert_eq!(mgr.gc_once().await, 1, "only the expired finished job goes");
        assert!(!old_done.exists(), "expired finished job removed");
        assert!(old_running.exists(), "running job never removed");
        assert!(fresh.exists(), "fresh job kept");
        assert!(mgr.get_job("job_old_running").is_some());
        assert!(mgr.get_job("job_old_done").is_none() || !old_done.exists());
    }

    #[tokio::test]
    async fn gc_disabled_keeps_everything() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let old = write_job_dir(root, "job_old", "completed");
        age_file(&old.join("job.json"), Duration::from_secs(30 * 24 * 3600));

        let mgr = gc_manager(root, None); // SMG_JOBS_RETENTION_HOURS=0
        assert_eq!(mgr.gc_once().await, 0);
        assert!(old.exists(), "GC disabled: nothing is deleted");
    }

    #[tokio::test]
    async fn gc_removes_untracked_expired_dirs_and_orphan_entries() {
        // A dir left by an older run (not in memory) must still be reclaimed,
        // and an in-memory finished job must disappear from the registry too.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let orphan = write_job_dir(root, "job_orphan", "completed");
        let tracked = write_job_dir(root, "job_tracked", "completed");
        let old = Duration::from_secs(96 * 3600);
        age_file(&orphan.join("job.json"), old);
        age_file(&tracked.join("job.json"), old);

        let mgr = gc_manager(root, Some(Duration::from_secs(48 * 3600)));
        mgr.jobs.insert(
            "job_tracked".to_string(),
            Arc::new(Job {
                job_id: "job_tracked".to_string(),
                created_at_unix: 1,
                tasks: Arc::new(RwLock::new(vec![Task {
                    task_id: "job_tracked_t0000".to_string(),
                    index: 0,
                    request: test_request(),
                    status: TaskStatus::Completed,
                    error: None,
                    result: None,
                }])),
            }),
        );

        assert_eq!(mgr.gc_once().await, 2);
        assert!(!orphan.exists());
        assert!(!tracked.exists());
        assert!(mgr.get_job("job_tracked").is_none(), "registry entry dropped");
    }

    #[test]
    fn gc_helpers_report_last_activity_and_size() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_job_dir(tmp.path(), "job_x", "completed");
        let size = dir_size(&dir);
        assert!(size > 0, "job dir has bytes");
        let m = newest_mtime(&dir).expect("mtime");
        assert!(SystemTime::now().duration_since(m).unwrap() < Duration::from_secs(60));

    }
}
