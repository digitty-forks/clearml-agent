//! Event emission: hand the metered `Event` to the in-process reporting channel
//! (`channel::try_send`), which forwards it to the reporter thread or — when no
//! reporter was started (e.g. an operator running `curl` under LD_PRELOAD with
//! no descriptor) — falls back to a `[snug-event] {json}` stderr line so events
//! still surface somewhere.

use clearml_snug_messages::Event;

/// Whether `event` should lazily start the in-process reporter.
///
/// Startup is confined to processes that actually meter LLM traffic, so
/// multi-process desktop hosts (Electron/Chromium and helper/sandbox children)
/// that load the shim but never touch a whitelisted host never spawn a reporter
/// (which would abort them). Two events qualify:
///
/// - `RequestCompleted`: carries the usage worth reporting.
/// - a *whitelisted* `RequestStarted`: the same "this process meters LLM
///   traffic" signal, but it is emitted at request-write time — before the
///   completion. Starting the reporter here installs the event channel before
///   this very `RequestStarted` is sent. Otherwise the connection's first
///   `RequestStarted` is lost to the stderr fallback (the channel isn't
///   installed until the first `RequestCompleted`), and the matching
///   `RequestCompleted` — which carries no host/whitelisted of its own and is
///   attributed only by joining the stashed `RequestStarted` by `conn_id` —
///   finds no stash and is silently dropped, losing the first metered call.
///
/// A non-whitelisted `RequestStarted` must NOT qualify: it is emitted for every
/// HTTP/1 request under `default_action == "meter"`, so an idle desktop helper
/// making any request would otherwise spawn a reporter and abort.
fn starts_reporter(event: &Event) -> bool {
    matches!(
        event,
        Event::RequestCompleted { .. } | Event::RequestStarted { whitelisted: true, .. }
    )
}

/// Ship `event` to the reporter. Takes the event BY VALUE and moves it into the
/// bounded channel — no JSON serialization on the user's hot path (the reporter
/// serializes it off-thread for log-forwarding). Non-blocking; see `channel`.
pub fn emit(event: Event) {
    // Lazily start the in-process reporter on the first metering signal. This
    // must run BEFORE `try_send` so the channel exists when the triggering
    // event is sent (see `starts_reporter`). No-op after the first call and in
    // processes with no descriptor.
    if starts_reporter(&event) {
        crate::reporter_handle::ensure_started();
    }
    crate::channel::try_send(event);
}

#[cfg(test)]
mod tests {
    use super::starts_reporter;
    use clearml_snug_messages::{Direction, Event};

    fn request_started(whitelisted: bool) -> Event {
        Event::RequestStarted {
            conn_id: 1,
            ts_ms: 0,
            host: "api.anthropic.com".into(),
            path: "/v1/messages".into(),
            method: "POST".into(),
            whitelisted,
            inject_headers: true,
        }
    }

    #[test]
    fn whitelisted_request_start_triggers_reporter_startup() {
        // Regression: the connection's first RequestStarted must start the
        // reporter so the channel is installed before that event is sent and its
        // matching RequestCompleted can be attributed. Without this the first
        // metered call of the process is silently dropped.
        assert!(starts_reporter(&request_started(true)));
    }

    #[test]
    fn non_whitelisted_request_start_does_not_start_reporter() {
        // Desktop-host safety: an idle Electron/Chromium helper emits only
        // non-whitelisted RequestStarted and must never spawn a reporter.
        assert!(!starts_reporter(&request_started(false)));
    }

    #[test]
    fn request_completed_triggers_reporter_startup() {
        // The original trigger stays intact (final request drains on SSL_free /
        // process exit even if its start was somehow missed).
        let completed = Event::RequestCompleted {
            conn_id: 1,
            ts_ms: 0,
            status: Some(200),
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tokens_in: 0,
            tokens_out: 0,
            tokens_measured: false,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: Vec::new(),
            tool_call_error_names: Vec::new(),
            chat_id: None,
            model: None,
        };
        assert!(starts_reporter(&completed));
    }

    #[test]
    fn bytes_observed_does_not_start_reporter() {
        // Byte reports (emitted for every write, including on non-metering
        // connections) must not spawn a reporter on their own.
        assert!(!starts_reporter(&Event::BytesObserved {
            conn_id: 1,
            ts_ms: 0,
            direction: Direction::Tx,
            bytes: 10,
            tokens_est: 2,
        }));
    }
}
