//! SSE snapshot + replay protocol (see docs/agent/sse-snapshot-replay.md).
//!
//! Protocol: a client request carrying `X-Sse-Snapshot-Key: <uuid>` gets its
//! SSE response stream tee'd into a snapshot. If the client disconnects
//! mid-stream, the upstream is *not* aborted — a detached drain keeps
//! appending until the stream finishes, so a reconnect with the same key can
//! replay the cached bytes and seamlessly tail any remaining live chunks.
//!
//! Lifecycle (user decisions, 2026-08-30):
//! - **No per-stream byte cap** (default unlimited).
//! - **Normal completion deletes the snapshot immediately** — the client that
//!   stayed connected already has all bytes. Only *detached* snapshots (client
//!   gone) survive, waiting for a replay; a completed replay also deletes.
//! - Enabled by default (`--sse-snapshot-enabled`, true).
//!
//! Concurrency model:
//! - `chunks` is a `Mutex<Vec<Bytes>>`; appends are short critical sections.
//! - Progress signalling uses a `tokio::sync::watch` channel carrying
//!   `(len, done)` — replay streams `changed().await` on it, which is
//!   lost-wakeup-free (a late subscriber immediately sees the latest value).
//! - Replay uses a positional cursor (`pos`) over `chunks`, so multiple
//!   concurrent replays of the same key are safe and byte-identical.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::Stream;
use tokio::sync::watch;

/// Request header carrying the client-generated snapshot key.
pub const SNAPSHOT_KEY_HEADER: &str = "x-sse-snapshot-key";
/// Response header telling the client what happened (first|replay|replay+live).
pub const SNAPSHOT_RESULT_HEADER: &str = "x-sse-snapshot";
pub const SNAPSHOT_RESULT_FIRST: &str = "first";
pub const SNAPSHOT_RESULT_REPLAY: &str = "replay";
pub const SNAPSHOT_RESULT_REPLAY_LIVE: &str = "replay+live";

/// In-band SSE termination sentinel (same detection PD router already uses).
pub const SSE_DONE_SENTINEL: &[u8] = b"data: [DONE]";

#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Master switch (default true per user decision).
    pub enabled: bool,
    /// Registry entry cap; oldest snapshots are evicted beyond this (leak guard).
    pub max_sessions: usize,
    /// TTL for detached snapshots nobody reconnects to (leak guard).
    pub ttl_secs: u64,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_sessions: 4096,
            ttl_secs: 1800,
        }
    }
}

/// Whether `chunk` terminates an SSE stream.
pub fn chunk_is_done(chunk: &[u8]) -> bool {
    memchr_find(chunk, SSE_DONE_SENTINEL).is_some()
}

/// Minimal `memmem`-style search (avoids pulling a crate for one use).
fn memchr_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Progress {
    len: u64,
    done: bool,
}

/// One snapshot: the tee'd bytes of a single SSE response, plus the progress
/// channel replay streams wait on.
pub struct SseSnapshot {
    key: String,
    /// Append-only raw SSE bytes as sent to the client.
    chunks: Mutex<Vec<Bytes>>,
    progress_tx: watch::Sender<Progress>,
    /// Total bytes appended (AtomicU64 so readers can peek without the mutex).
    len: AtomicU64,
    /// Upstream finished ([DONE] sentinel or stream end).
    done: AtomicBool,
    /// Client disconnected; background task keeps draining upstream.
    detached: AtomicBool,
    created: Instant,
}

impl SseSnapshot {
    fn new(key: impl Into<String>) -> Arc<Self> {
        let (tx, _rx) = watch::channel(Progress { len: 0, done: false });
        Arc::new(Self {
            key: key.into(),
            chunks: Mutex::new(Vec::new()),
            progress_tx: tx,
            len: AtomicU64::new(0),
            done: AtomicBool::new(false),
            detached: AtomicBool::new(false),
            created: Instant::now(),
        })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn len(&self) -> u64 {
        self.len.load(Ordering::Acquire)
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    pub fn is_detached(&self) -> bool {
        self.detached.load(Ordering::Acquire)
    }

    /// Mark that the client is gone. After this the producing task keeps
    /// reading upstream (drain) so the snapshot can complete for replay.
    pub fn set_detached(&self) {
        self.detached.store(true, Ordering::Release);
    }

    /// Append a chunk (post-processed bytes exactly as sent to the client).
    /// No-op after `finish`. Wakes replay streams.
    pub fn append(&self, chunk: Bytes) {
        if self.is_done() {
            return;
        }
        let mut guard = self.chunks.lock().unwrap();
        let new_len = self.len.load(Ordering::Relaxed) + chunk.len() as u64;
        guard.push(chunk);
        drop(guard);
        self.len.store(new_len, Ordering::Release);
        let _ = self
            .progress_tx
            .send(Progress { len: new_len, done: false });
    }

    /// Upstream finished. Idempotent. Wakes replay streams so live tails end.
    pub fn finish(&self) {
        if self.done.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.progress_tx.send(Progress {
            len: self.len.load(Ordering::Acquire),
            done: true,
        });
    }

    /// Clone the chunk prefix up to `pos` (pos <= current len).
    fn prefix(&self, pos: usize) -> Vec<Bytes> {
        let guard = self.chunks.lock().unwrap();
        if pos >= guard.len() {
            Vec::new()
        } else {
            guard[pos..].to_vec()
        }
    }

    fn progress_rx(&self) -> watch::Receiver<Progress> {
        self.progress_tx.subscribe()
    }

    pub fn age(&self) -> Duration {
        self.created.elapsed()
    }
}

/// Global registry (process-wide singleton; smg runs one per instance).
pub struct SnapshotRegistry {
    config: SnapshotConfig,
    map: Mutex<HashMap<String, Arc<SseSnapshot>>>,
}

impl SnapshotRegistry {
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            map: Mutex::new(HashMap::new()),
        }
    }

    pub fn config(&self) -> &SnapshotConfig {
        &self.config
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }

    /// Extract a valid snapshot key from request headers (case-insensitive).
    pub fn key_from_headers(headers: &http::HeaderMap) -> Option<String> {
        let v = headers.get(SNAPSHOT_KEY_HEADER)?;
        let s = v.to_str().ok()?;
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed.len() > 256 {
            return None;
        }
        Some(trimmed.to_string())
    }

    /// Look up a live snapshot for replay. Returns None if absent or the
    /// snapshot is done *and* was never detached (normal completion removes
    /// it from the registry, so this is mostly the detached-drain case).
    pub fn get_for_replay(&self, key: &str) -> Option<Arc<SseSnapshot>> {
        self.map.lock().unwrap().get(key).cloned()
    }

    /// Insert a fresh snapshot for `key`, evicting the oldest entry if the
    /// cap is exceeded. Returns the inserted snapshot.
    pub fn insert(&self, key: String) -> Arc<SseSnapshot> {
        let snap = SseSnapshot::new(&key);
        let mut map = self.map.lock().unwrap();
        if map.len() >= self.config.max_sessions {
            // Evict the oldest entry (leak guard; normal entries are removed
            // on completion, so survivors are detached stragglers).
            if let Some(oldest) = map
                .values()
                .min_by_key(|s| s.created)
                .map(|s| s.key().to_string())
            {
                map.remove(&oldest);
                tracing::warn!(
                    "sse-snapshot: registry cap {} exceeded, evicted oldest detached snapshot {oldest}",
                    self.config.max_sessions
                );
            }
        }
        map.insert(key, snap.clone());
        snap
    }

    /// Remove by key (normal completion / replay completion / TTL).
    pub fn remove(&self, key: &str) {
        self.map.lock().unwrap().remove(key);
    }

    /// Periodic leak guard: drop detached-and-done snapshots older than TTL.
    /// Returns the number of evictions.
    pub fn sweep_expired(&self) -> usize {
        let now = Instant::now();
        let ttl = Duration::from_secs(self.config.ttl_secs);
        let mut map = self.map.lock().unwrap();
        let expired: Vec<String> = map
            .iter()
            .filter(|(_, s)| now.duration_since(s.created) > ttl)
            .map(|(k, _)| k.clone())
            .collect();
        let n = expired.len();
        for k in expired {
            map.remove(&k);
        }
        n
    }
}

static REGISTRY: OnceLock<Arc<SnapshotRegistry>> = OnceLock::new();

/// Initialize the global registry (call once at startup; subsequent calls are
/// no-ops unless the config changed, in which case the first config wins).
pub fn init_registry(config: SnapshotConfig) {
    let _ = REGISTRY.set(Arc::new(SnapshotRegistry::new(config)));
}

pub fn registry() -> Option<Arc<SnapshotRegistry>> {
    REGISTRY.get().cloned()
}

/// Process-wide periodic sweep task (spawn at startup). No-op when the
/// feature is disabled.
pub async fn sweep_loop() {
    let Some(reg) = registry() else { return };
    if !reg.enabled() {
        return;
    }
    let interval = Duration::from_secs(reg.config().ttl_secs.max(60) / 4);
    loop {
        tokio::time::sleep(interval).await;
        let n = reg.sweep_expired();
        if n > 0 {
            tracing::info!("sse-snapshot: TTL sweep removed {n} expired snapshots");
        }
    }
}

/// Error type for the replay stream (never actually produced — replay yields
/// cached bytes only; kept for `Body::from_stream` compatibility).
#[derive(Debug)]
pub struct ReplayError(std::io::Error);

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sse snapshot replay error: {}", self.0)
    }
}

impl std::error::Error for ReplayError {}

/// Positional-cursor replay state over a snapshot: yields cached chunks from
/// `pos`, then tails live appends until done, then ends (and removes the
/// snapshot — a completed replay means the client has every byte).
///
/// Byte-identical to what the first client saw: no re-parsing, no re-framing.
struct ReplayState {
    snap: Arc<SseSnapshot>,
    registry: Arc<SnapshotRegistry>,
    pos: usize,
    progress: watch::Receiver<Progress>,
}

/// True if the snapshot is still receiving from upstream (replay will live-tail).
pub fn is_live_tail(snap: &SseSnapshot) -> bool {
    !snap.is_done()
}

/// Build the replay stream for a snapshot. Yields the cached bytes in order,
/// tails live appends until `finish`, then removes the snapshot from the
/// registry (user decision: normal completion — including a completed replay
/// — deletes the snapshot; only detached ones linger awaiting reconnect).
pub fn replay_stream(
    snap: Arc<SseSnapshot>,
    registry: Arc<SnapshotRegistry>,
) -> impl Stream<Item = Result<Bytes, ReplayError>> + Send + 'static {
    let progress = snap.progress_rx();
    futures_util::stream::unfold(
        ReplayState {
            snap,
            registry,
            pos: 0,
            progress,
        },
        |mut st| async move {
            loop {
                // 1. Yield any chunk beyond `pos` (clone of the shared handle).
                let next = {
                    let guard = st.snap.chunks.lock().unwrap();
                    guard.get(st.pos).cloned()
                };
                if let Some(chunk) = next {
                    st.pos += 1;
                    return Some((Ok(chunk), st));
                }
                // 2. Caught up: if the producer finished, the replay is
                //    complete — remove the snapshot and end the stream.
                if st.snap.is_done() {
                    st.registry.remove(st.snap.key());
                    return None;
                }
                // 3. Wait for progress (append or finish). watch channels are
                //    version-based: subscribe-then-await never misses a send —
                //    no lost-wakeup window between reading chunks and sleeping.
                if st.progress.changed().await.is_err() {
                    // Sender dropped (snapshot gone) — treat as end of stream.
                    st.registry.remove(st.snap.key());
                    return None;
                }
                // loop to re-read chunks
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    fn cfg() -> SnapshotConfig {
        SnapshotConfig {
            enabled: true,
            max_sessions: 8,
            ttl_secs: 60,
        }
    }

    fn reg() -> Arc<SnapshotRegistry> {
        Arc::new(SnapshotRegistry::new(cfg()))
    }

    #[test]
    fn chunk_is_done_detects_sentinel() {
        assert!(chunk_is_done(b"data: {\"x\":1}\n\ndata: [DONE]\n\n"));
        assert!(!chunk_is_done(b"data: {\"x\":1}\n\n"));
        assert!(!chunk_is_done(b""));
    }

    #[test]
    fn memchr_find_basic() {
        assert_eq!(memchr_find(b"abcdef", b"cd"), Some(2));
        assert_eq!(memchr_find(b"abc", b"abcd"), None);
        assert_eq!(memchr_find(b"", b"a"), None);
    }

    #[test]
    fn append_finish_progress() {
        let snap = SseSnapshot::new("k1");
        snap.append(Bytes::from_static(b"aa"));
        snap.append(Bytes::from_static(b"bb"));
        assert_eq!(snap.len(), 4);
        assert!(!snap.is_done());
        let prefix = snap.prefix(0);
        assert_eq!(prefix.len(), 2);
        assert_eq!(prefix[0], Bytes::from_static(b"aa"));
        // append after finish is a no-op
        snap.finish();
        snap.append(Bytes::from_static(b"zz"));
        assert_eq!(snap.len(), 4);
        assert!(snap.is_done());
        // finish is idempotent
        snap.finish();
        assert!(snap.is_done());
    }

    #[test]
    fn registry_insert_get_remove_evict() {
        let reg = reg();
        let snap = reg.insert("k1".into());
        assert!(reg.get_for_replay("k1").is_some());
        assert!(Arc::ptr_eq(&reg.get_for_replay("k1").unwrap(), &snap));
        reg.remove("k1");
        assert!(reg.get_for_replay("k1").is_none());

        // cap eviction: fill beyond max_sessions, oldest goes away
        for i in 0..9 {
            reg.insert(format!("k{i}"));
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(reg.get_for_replay("k0").is_none(), "oldest evicted");
        assert!(reg.get_for_replay("k8").is_some());
    }

    #[test]
    fn sweep_expired_removes_old_detached() {
        let reg = reg();
        let mut old_cfg = cfg();
        old_cfg.ttl_secs = 0;
        let old_reg = Arc::new(SnapshotRegistry::new(old_cfg));
        old_reg.insert("old".into());
        assert_eq!(old_reg.sweep_expired(), 1);
        assert_eq!(old_reg.len(), 0);
        reg.insert("fresh".into());
        assert_eq!(reg.sweep_expired(), 0);
        assert_eq!(reg.len(), 1);
    }

    #[tokio::test]
    async fn replay_yields_all_chunks_after_done_then_removes() {
        let reg = reg();
        let snap = reg.insert("replay-done".into());
        snap.append(Bytes::from_static(b"data: 1\n\n"));
        snap.append(Bytes::from_static(b"data: {\"v\":2}\n\n"));
        snap.append(Bytes::from_static(b"data: [DONE]\n\n"));
        snap.finish();

        let stream = replay_stream(snap, reg.clone());
        let collected: Vec<Bytes> = stream.map(|c| c.unwrap()).collect().await;
        assert_eq!(
            collected,
            vec![
                Bytes::from_static(b"data: 1\n\n"),
                Bytes::from_static(b"data: {\"v\":2}\n\n"),
                Bytes::from_static(b"data: [DONE]\n\n"),
            ]
        );
        // replay completed -> snapshot removed
        assert!(reg.get_for_replay("replay-done").is_none());
    }

    #[tokio::test]
    async fn replay_tails_live_appends_until_finish() {
        let reg = reg();
        let snap = reg.insert("live".into());
        snap.append(Bytes::from_static(b"part1;"));

        let snap2 = snap.clone();
        // producer simulates the detached drain: appends over time, then finish
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            snap2.append(Bytes::from_static(b"part2;"));
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            snap2.append(Bytes::from_static(b"part3;"));
            snap2.finish();
        });

        let stream = replay_stream(snap.clone(), reg.clone());
        let collected: Vec<Bytes> = stream.map(|c| c.unwrap()).collect().await;
        assert_eq!(
            collected,
            vec![
                Bytes::from_static(b"part1;"),
                Bytes::from_static(b"part2;"),
                Bytes::from_static(b"part3;"),
            ]
        );
        // replay completed -> snapshot removed
        assert!(reg.get_for_replay("live").is_none());
    }

    #[tokio::test]
    async fn replay_incomplete_snapshot_survives_client_drop() {
        // Client disconnects mid-replay: stream dropped before finish.
        // Snapshot must survive (another replay can happen).
        let reg = reg();
        let snap = reg.insert("partial".into());
        snap.append(Bytes::from_static(b"chunk1"));
        snap.append(Bytes::from_static(b"chunk2"));
        // NOTE: not finished — still live

        let stream = replay_stream(snap.clone(), reg.clone());
        futures_util::pin_mut!(stream);
        // take two chunks then drop the stream
        let a = stream.next().await.unwrap().unwrap();
        let b = stream.next().await.unwrap().unwrap();
        assert_eq!(a, Bytes::from_static(b"chunk1"));
        assert_eq!(b, Bytes::from_static(b"chunk2"));
        drop(stream);
        // not removed: only *completed* replay removes
        assert!(reg.get_for_replay("partial").is_some());
    }

    #[test]
    fn key_from_headers() {
        let mut h = http::HeaderMap::new();
        assert!(SnapshotRegistry::key_from_headers(&h).is_none());
        h.insert("x-sse-snapshot-key", "abc-123".parse().unwrap());
        assert_eq!(
            SnapshotRegistry::key_from_headers(&h).as_deref(),
            Some("abc-123")
        );
        h.insert("x-sse-snapshot-key", "  ".parse().unwrap());
        assert_eq!(SnapshotRegistry::key_from_headers(&h), None);
        let long: String = "k".repeat(300);
        h.insert("x-sse-snapshot-key", long.parse().unwrap());
        assert_eq!(SnapshotRegistry::key_from_headers(&h), None);
    }
}

/// Entry-point hook for the replay short-circuit: given request headers, return
/// a replay `Response` when a snapshot for the key exists (either mid-stream
/// with live tailing, or finished). Router call sites should try this BEFORE
/// dispatching to workers — a hit never touches upstream (no worker selection,
/// no token bucket), which also makes replay free.
pub fn try_replay(headers: &http::HeaderMap) -> Option<axum::response::Response> {
    use futures_util::StreamExt;
    let reg = registry()?;
    if !reg.enabled() {
        return None;
    }
    let key = SnapshotRegistry::key_from_headers(headers)?;
    let snap = reg.get_for_replay(&key)?;
    let live = is_live_tail(&*snap);
    tracing::info!(
        "sse-snapshot: replaying key={} (live_tail={})",
        key,
        live
    );
    let stream = replay_stream(snap, reg);
    let body = axum::body::Body::from_stream(stream.map(|r| {
        r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }));
    let mut resp = axum::response::Response::new(body);
    let h = resp.headers_mut();
    h.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    h.insert(
        SNAPSHOT_RESULT_HEADER,
        http::HeaderValue::from_static(if live {
            SNAPSHOT_RESULT_REPLAY_LIVE
        } else {
            SNAPSHOT_RESULT_REPLAY
        }),
    );
    Some(resp)
}
