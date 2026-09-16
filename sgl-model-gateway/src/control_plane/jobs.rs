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

/// One training task, in the training client's OpenAI-ish format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskRequest {
    pub prompt: String,
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

    /// Accepts either a raw JSON array of task requests, or an object
    /// `{ "lora_path": optional, "requests": [...] }` (job-level lora as the
    /// default, overridable per task).
    pub async fn submit(self: &Arc<Self>, body: Value) -> Result<Arc<Job>, String> {
        let parse_requests = |arr: &[Value]| -> Result<Vec<TaskRequest>, String> {
            arr.iter()
                .enumerate()
                .map(|(i, v)| {
                    serde_json::from_value::<TaskRequest>(v.clone())
                        .map_err(|e| format!("invalid task at index {i}: {e}"))
                })
                .collect()
        };
        let (job_lora, requests): (Option<String>, Vec<TaskRequest>) = match &body {
            Value::Array(arr) => (None, parse_requests(arr)?),
            Value::Object(obj) => {
                let lora = obj
                    .get("lora_path")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let reqs = obj
                    .get("requests")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| "object body must contain a 'requests' array".to_string())?;
                (lora, parse_requests(reqs)?)
            }
            _ => return Err("body must be a JSON array of tasks or {requests: [...]}".into()),
        };
        if requests.is_empty() {
            return Err("empty task array".into());
        }
        if requests.len() > 4096 {
            return Err("too many tasks in one job (max 4096)".into());
        }
        for r in &requests {
            if r.prompt.trim().is_empty() {
                return Err("task with empty prompt".into());
            }
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
                .run_one_sample(&req, &cancel, &progress, Some(&live))
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
    /// SSE stream into a single sample result. `cancel` (when set) stops the
    /// stream at the next chunk boundary and returns whatever was aggregated so
    /// far; `progress` is updated with the running generated-token count;
    /// `live` (task-level partial result) receives periodic snapshots of the
    /// in-flight sample so download endpoints can stream partial output.
    async fn run_one_sample(
        &self,
        req: &TaskRequest,
        cancel: &Arc<AtomicBool>,
        progress: &Arc<AtomicU64>,
        live: Option<&Arc<tokio::sync::RwLock<TaskResult>>>,
    ) -> Result<SampleResult, String> {
        let mut gen_body = json!({
            "text": req.prompt,
            "sampling_params": {
                "max_new_tokens": req.max_tokens.unwrap_or(4096),
            },
            "return_logprob": true,
            "stream": true,
        });
        {
            let sp = gen_body["sampling_params"].as_object_mut().unwrap();
            if let Some(t) = req.temperature {
                sp.insert("temperature".into(), json!(t));
            }
            if let Some(p) = req.top_p {
                sp.insert("top_p".into(), json!(p));
            }
        }
        if let Some(lp) = &req.lora_path {
            gen_body["lora_path"] = json!(lp);
        }

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
            .json(&gen_body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder
            .send()
            .await
            .map_err(|e| format!("generate request failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("generate returned {status}: {}", truncate(&body, 500)));
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

    #[test]
    fn aggregate_status_counts_cancelled() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mk = |s: TaskStatus| Task {
                task_id: "t".into(),
                index: 0,
                request: TaskRequest {
                    prompt: "x".into(),
                    max_tokens: None,
                    temperature: None,
                    top_p: None,
                    n: 1,
                    lora_path: None,
                    model: None,
                    stream: None,
                    logprobs: None,
                    stream_options: None,
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
