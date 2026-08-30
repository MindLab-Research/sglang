//! Embedded admin Web UI — single-file, zero build chain (no npm, no CDN).
//!
//! `GET /admin/ui/lora` serves the LoRA Console: mounted-adapter dashboard
//! (GET /v1/control/units), load/unload (POST/DELETE /v1/control/models),
//! request how-to (lora_path field) and the SSE snapshot+replay protocol
//! walkthrough. The page itself carries no secrets — data is fetched from
//! the smg control plane with the API key the user types into the login
//! screen (stored in localStorage, sent as Bearer).

use axum::http::{header, HeaderValue};
use axum::response::Response;

pub async fn lora_ui() -> Response {
    let html = include_str!("admin/lora_ui.html");
    let mut resp = Response::new(axum::body::Body::from(html));
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}
