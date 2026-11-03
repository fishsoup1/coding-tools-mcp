use std::sync::{Arc, RwLock};

use axum::http::HeaderMap;
use serde_json::{json, Value};

use crate::workspace::AuthConfig;

const MAX_RETIRED_PUBLIC_URLS: usize = 4;

/// The public base URL that is actually serving right now.
///
/// Quick tunnels hand out a new `trycloudflare.com` host every time they start.
/// The listener and its OAuth runtime must read the live value instead of the
/// URL that happened to be persisted when the process started.
#[derive(Clone, Debug, Default)]
pub struct PublicBaseUrl {
    inner: Arc<RwLock<PublicBaseUrlState>>,
}

#[derive(Debug, Default)]
struct PublicBaseUrlState {
    current: String,
    history: Vec<String>,
}

impl PublicBaseUrl {
    pub fn new(initial: impl AsRef<str>) -> Self {
        let state = Self::default();
        state.set(initial.as_ref());
        state
    }

    pub fn current(&self) -> String {
        self.read().current.clone()
    }

    /// Replace the authoritative URL and report whether it changed.
    ///
    /// Retired URLs stay accepted as token audiences so an OAuth grant issued
    /// before a tunnel domain change does not force the user to re-authorize.
    pub fn set(&self, url: &str) -> bool {
        let next = normalize_public_url(url);
        if next.is_empty() {
            return false;
        }

        let mut state = self
            .inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.current == next {
            return false;
        }
        if !state.current.is_empty() {
            let previous = std::mem::take(&mut state.current);
            state.history.retain(|url| url != &previous);
            state.history.push(previous);
            if state.history.len() > MAX_RETIRED_PUBLIC_URLS {
                state.history.remove(0);
            }
        }
        state.current = next;
        true
    }

    /// Current URL first, then recently retired URLs (newest first).
    pub fn accepted(&self) -> Vec<String> {
        let state = self.read();
        let mut accepted = Vec::with_capacity(state.history.len() + 1);
        if !state.current.is_empty() {
            accepted.push(state.current.clone());
        }
        for url in state.history.iter().rev() {
            if !accepted.contains(url) {
                accepted.push(url.clone());
            }
        }
        accepted
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, PublicBaseUrlState> {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl From<String> for PublicBaseUrl {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for PublicBaseUrl {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

pub fn normalize_public_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

impl AuthConfig {
    pub fn oauth_enabled(&self) -> bool {
        self.auth_type == "oauth"
    }

    pub fn bearer_enabled(&self) -> bool {
        self.auth_type == "bearer"
    }

    pub fn auth_enabled(&self) -> bool {
        self.auth_type != "noauth"
    }
}

/// Resolve the external OAuth/MCP base URL for a request.
///
/// A configured or runtime-updated URL always wins. Without one, only loopback
/// `Host` values are trusted; arbitrary external `Host`/`X-Forwarded-Host`
/// headers must not be able to redefine the OAuth issuer and token audience.
pub fn external_base_url(headers: &HeaderMap, bind_port: u16, configured_url: &str) -> String {
    let configured = normalize_public_url(configured_url);
    if !configured.is_empty() {
        return configured;
    }

    let proto = {
        let value = first_header_value(headers, "x-forwarded-proto");
        if value.is_empty() {
            forwarded_header_param(headers, "proto")
        } else {
            value
        }
    };
    let host = {
        let value = safe_external_host(&first_header_value(headers, "x-forwarded-host"));
        if !value.is_empty() {
            value
        } else {
            let value = safe_external_host(&forwarded_header_param(headers, "host"));
            if !value.is_empty() {
                value
            } else {
                safe_external_host(&first_header_value(headers, "host"))
            }
        }
    };

    let host = if host.is_empty() || !is_loopback_host(host_without_port(&host)) {
        format!("127.0.0.1:{bind_port}")
    } else {
        host
    };
    let proto = resolve_external_proto(
        if proto.is_empty() {
            None
        } else {
            Some(proto.as_str())
        },
        &host,
    );
    format!("{proto}://{host}")
}

fn first_header_value(headers: &HeaderMap, name: &str) -> String {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(',').next().unwrap_or("").trim().to_string())
        .unwrap_or_default()
}

fn forwarded_header_param(headers: &HeaderMap, name: &str) -> String {
    let first = first_header_value(headers, "forwarded");
    for part in first.split(';') {
        let part = part.trim();
        if let Some((key, value)) = part.split_once('=') {
            if key.trim().eq_ignore_ascii_case(name) {
                return value.trim().trim_matches('"').to_string();
            }
        }
    }
    String::new()
}

fn safe_external_host(host: &str) -> String {
    let host = host.trim();
    if host.is_empty()
        || host.chars().any(|ch| {
            matches!(ch, '\r' | '\n' | '/' | '\\') || ch.is_ascii_whitespace()
        })
    {
        String::new()
    } else {
        host.to_string()
    }
}

fn host_without_port(host: &str) -> &str {
    if host.starts_with('[') {
        return host
            .split_once(']')
            .map(|(value, _)| value.trim_start_matches('['))
            .unwrap_or_else(|| host.trim_matches('[').trim_matches(']'));
    }
    host.rsplit_once(':')
        .map(|(value, _)| value)
        .unwrap_or(host)
}

fn resolve_external_proto(proto: Option<&str>, host: &str) -> &'static str {
    if let Some(proto) = proto {
        let proto = proto.trim().to_ascii_lowercase();
        if proto == "http" {
            return "http";
        }
        if proto == "https" {
            return "https";
        }
    }

    if is_loopback_host(host_without_port(host)) {
        "http"
    } else {
        "https"
    }
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

fn token_endpoint_auth_methods(client_secret: Option<&str>) -> Vec<&'static str> {
    match client_secret {
        Some(secret) if !secret.is_empty() => {
            vec!["client_secret_post", "client_secret_basic"]
        }
        _ => vec!["none"],
    }
}

pub fn authorization_server_metadata(base_url: &str, client_secret: Option<&str>) -> Value {
    let base = base_url.trim_end_matches('/');
    let methods = token_endpoint_auth_methods(client_secret);
    json!({
        "issuer": base,
        "authorization_endpoint": format!("{base}/oauth/authorize"),
        "token_endpoint": format!("{base}/oauth/token"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": methods,
    })
}

pub fn protected_resource_metadata(base_url: &str) -> Value {
    let base = base_url.trim_end_matches('/');
    json!({
        "resource": base,
        "authorization_servers": [base],
        "bearer_methods_supported": ["header"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_enabled_only_for_oauth_type() {
        let mut auth = AuthConfig::default();
        assert!(auth.oauth_enabled());
        auth.auth_type = "bearer".into();
        assert!(!auth.oauth_enabled());
        auth.auth_type = "noauth".into();
        assert!(!auth.oauth_enabled());
    }

    #[test]
    fn authorization_metadata_includes_token_auth_methods() {
        let meta = authorization_server_metadata("https://example.com", None);
        assert_eq!(
            meta["token_endpoint_auth_methods_supported"],
            json!(["none"])
        );
        let meta = authorization_server_metadata("https://example.com", Some("secret"));
        assert_eq!(
            meta["token_endpoint_auth_methods_supported"],
            json!(["client_secret_post", "client_secret_basic"])
        );
    }

    #[test]
    fn protected_resource_metadata_lists_authorization_servers() {
        let meta = protected_resource_metadata("https://example.com");
        assert_eq!(meta["authorization_servers"], json!(["https://example.com"]));
    }

    #[test]
    fn external_base_url_prefers_configured_url() {
        let headers = HeaderMap::new();
        assert_eq!(
            external_base_url(&headers, 28767, "https://lb.frp-tx1.evwali.com"),
            "https://lb.frp-tx1.evwali.com"
        );
    }

    #[test]
    fn external_base_url_ignores_untrusted_forwarded_host() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-proto", "https".parse().unwrap());
        headers.insert("x-forwarded-host", "evil.example.com".parse().unwrap());
        assert_eq!(
            external_base_url(&headers, 28767, ""),
            "http://127.0.0.1:28767"
        );
    }

    #[test]
    fn external_base_url_ignores_untrusted_host_header() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "evil.example.com".parse().unwrap());
        assert_eq!(
            external_base_url(&headers, 28767, ""),
            "http://127.0.0.1:28767"
        );
    }

    #[test]
    fn external_base_url_keeps_loopback_host_header() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "localhost:28767".parse().unwrap());
        assert_eq!(
            external_base_url(&headers, 28767, ""),
            "http://localhost:28767"
        );
    }

    #[test]
    fn configured_url_wins_over_request_host_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "evil.example.com".parse().unwrap());
        assert_eq!(
            external_base_url(&headers, 28767, "https://old.trycloudflare.com/"),
            "https://old.trycloudflare.com"
        );
    }

    #[test]
    fn public_base_url_tracks_domain_changes_and_retires_old_audiences() {
        let state = PublicBaseUrl::new("https://old.trycloudflare.com/");
        assert_eq!(state.current(), "https://old.trycloudflare.com");
        assert!(state.set("https://new.trycloudflare.com"));
        assert_eq!(state.current(), "https://new.trycloudflare.com");
        assert_eq!(
            state.accepted(),
            vec![
                "https://new.trycloudflare.com".to_string(),
                "https://old.trycloudflare.com".to_string()
            ]
        );
        assert!(!state.set("https://new.trycloudflare.com"));
    }
}
