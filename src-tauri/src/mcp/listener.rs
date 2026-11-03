use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Form, Query, State};
use axum::http::{header::CACHE_CONTROL, HeaderMap, StatusCode};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tower_http::cors::CorsLayer;

use crate::audit::request_context_from_headers;
use crate::auth::{
    authorization_server_metadata, authorize_get, authorize_post, external_base_url,
    protected_resource_metadata, token_exchange, verify_bearer_header, verify_oauth_bearer_header,
    AuthorizeForm, AuthorizeParams, OAuthRuntime, PublicBaseUrl, TokenForm,
};
use crate::mcp::server::{handle_request_with_context, new_state, SharedState};
use crate::mcp::upstream::UpstreamMcpManager;
use crate::secret::SecretStore;
use crate::tools::Workspace;
use crate::tunnel::append_profile_log;
use crate::tools::policy::PolicySettings;
use crate::workspace::{AuthConfig, RuntimeConfig};

pub type ShutdownSender = oneshot::Sender<()>;

#[derive(Clone)]
struct ListenerState {
    mcp: SharedState,
    auth: AuthConfig,
    workspace_id: String,
    workspace_path: String,
    bind_port: u16,
    public_url: PublicBaseUrl,
    bearer_token: Option<String>,
    oauth: Option<Arc<OAuthRuntime>>,
    oauth_client_secret: Option<String>,
}

pub struct ListenerHandle {
    pub shutdown: ShutdownSender,
    pub public_url: PublicBaseUrl,
    pub join: tauri::async_runtime::JoinHandle<()>,
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_listener(
    port: u16,
    workspace_path: PathBuf,
    workspace_id: String,
    auth: AuthConfig,
    public_base_url: String,
    oauth_client_secret: Option<String>,
    oauth_password: Option<String>,
    oauth_token_secret: Option<String>,
    runtime: RuntimeConfig,
    upstream: Arc<UpstreamMcpManager>,
) -> Result<ListenerHandle, String> {
    let workspace_display = workspace_path.display().to_string();
    let workspace = Workspace::new(workspace_path).map_err(|e| e.message())?;
    if matches!(
        crate::tools::registry::normalize_tool_profile(&runtime.tool_profile),
        "read-only" | "compat-readonly-all"
    ) && !upstream.public_tools().is_empty()
    {
        return Err(
            "只读工具档位不能公开本地 MCP 工具；请切换到核心或完整工具档位后再启动"
                .to_string(),
        );
    }
    // Bind before starting local children. A port collision must not leave an
    // upstream process running after this MCP start attempt failed.
    let listener = bind_listener(port)?;
    let policy = PolicySettings::from_runtime(&runtime);
    // 监听器创建 SharedState 时注入稳定 workspace_id；服务端后续只持有共享 Context，无法再从
    // 单条 JSON-RPC 请求可靠恢复工作区归属。
    let mcp = new_state(
        workspace,
        workspace_id.clone(),
        auth.clone(),
        policy,
        runtime.tool_profile.clone(),
        runtime.permission_mode.clone(),
        upstream,
    );
    let bearer_token = if auth.bearer_enabled() {
        let key = "bearer_token";
        if auth.use_shared_secrets {
            SecretStore::get_shared(key).map_err(|e| e.to_string())?
        } else {
            SecretStore::get(&workspace_id, key).map_err(|e| e.to_string())?
        }
    } else {
        None
    };
    let public_url = PublicBaseUrl::new(public_base_url);
    let oauth = if auth.oauth_enabled() {
        let password = oauth_password.unwrap_or_default();
        let token_secret = oauth_token_secret.unwrap_or_default();
        // Share the URL handle with the listener so a Quick Tunnel domain change
        // can update OAuth metadata without restarting the listener.
        Some(Arc::new(OAuthRuntime::new(
            public_url.clone(),
            auth.oauth_client_id.clone(),
            oauth_client_secret.clone(),
            password,
            token_secret,
        )))
    } else {
        None
    };
    let state = ListenerState {
        mcp,
        auth,
        workspace_id,
        workspace_path: workspace_display,
        bind_port: port,
        public_url: public_url.clone(),
        bearer_token,
        oauth,
        oauth_client_secret,
    };
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let profile_id = state.workspace_id.clone();
    let handle = tauri::async_runtime::spawn(async move {
        let result = serve(listener, port, state, shutdown_rx).await;
        if let Err(err) = &result {
            append_profile_log(
                &profile_id,
                "stderr.log",
                &format!("[mcp] listener stopped: {err}"),
            );
            eprintln!("mcp listener stopped: {err}");
        } else {
            append_profile_log(&profile_id, "stderr.log", "[mcp] listener stopped");
        }
    });
    Ok(ListenerHandle {
        shutdown: shutdown_tx,
        public_url,
        join: handle,
    })
}

async fn serve(
    listener: tokio::net::TcpListener,
    port: u16,
    state: ListenerState,
    shutdown: oneshot::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let profile_id = state.workspace_id.clone();
    let upstream = state.mcp.upstream.clone();
    // access 中间件包在完整 MCP Router 外层，统一记录协议、OAuth、discovery 与 404；"mcp"
    // 分区避免和同工作区 Actions 日志混写。
    let access_log_state =
        crate::access_log::AccessLogState::new(profile_id.clone(), "mcp", state.mcp.audit_store());
    let app = Router::new()
        .route("/mcp", get(mcp_discovery).post(mcp_post))
        .route(
            "/.well-known/oauth-authorization-server",
            get(oauth_authorization_server_metadata),
        )
        .route(
            "/.well-known/oauth-protected-resource",
            get(oauth_protected_resource_metadata),
        )
        .route("/oauth/authorize", get(oauth_authorize_get).post(oauth_authorize_post))
        .route("/oauth/token", post(oauth_token_post))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(middleware::from_fn_with_state(
            access_log_state,
            crate::access_log::middleware,
        ));

    append_profile_log(
        &profile_id,
        "stdout.log",
        &format!("[mcp] listening on http://127.0.0.1:{port}/mcp"),
    );
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown.await;
        })
        .await;
    upstream.shutdown().await;
    result?;
    Ok(())
}

fn bind_listener(port: u16) -> Result<tokio::net::TcpListener, String> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = std::net::TcpListener::bind(addr)
        .map_err(|err| format!("MCP 本地端口 {port} 绑定失败: {err}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|err| format!("MCP 本地端口 {port} 设置非阻塞失败: {err}"))?;
    tokio::net::TcpListener::from_std(listener)
        .map_err(|err| format!("MCP 本地监听器初始化失败: {err}"))
}

async fn mcp_discovery() -> Response {
    ([(CACHE_CONTROL, "no-store")], Json(mcp_discovery_payload())).into_response()
}

fn mcp_discovery_payload() -> Value {
    json!({
        "name": "coding-tools-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "protocolVersion": "2025-06-18"
    })
}

fn resolve_oauth_base(state: &ListenerState, headers: &HeaderMap) -> String {
    resolve_oauth_base_from(&state.public_url, state.bind_port, headers)
}

fn resolve_oauth_base_from(
    public_url: &PublicBaseUrl,
    bind_port: u16,
    headers: &HeaderMap,
) -> String {
    external_base_url(headers, bind_port, &public_url.current())
}

async fn mcp_post(
    State(state): State<ListenerState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Some(response) = require_mcp_auth(&state, &headers) {
        return response;
    }
    let method = body
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let request_id = body.get("id").cloned().unwrap_or(Value::Null);
    let tool_name = body
        .get("params")
        .and_then(|params| params.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let is_upstream_tool = state.mcp.upstream.owns_tool(&tool_name);
    let started_at = Instant::now();
    // 在 body 移入 blocking 执行前提取 headers 与 JSON-RPC id，使服务端审计获得完整请求上下文；
    // id 只做协议关联，记录唯一性仍由审计 UUID 保证。
    let request_context = request_context_from_headers(
        &headers,
        "mcp",
        Some(method.clone()),
        body.get("id").and_then(crate::audit::request_id_from_value),
        Some("/mcp".into()),
    );
    append_profile_log(
        &state.workspace_id,
        "mcp-requests.log",
        &format!(
            "[rpc] request id={} method={} tool={}",
            request_id, method, tool_name
        ),
    );

    let mcp = state.mcp.clone();
    let profile_id = state.workspace_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        handle_request_with_context(&mcp, &body, &request_context)
    })
    .await;
    match result {
        Ok(response) => {
            append_profile_log(
                &profile_id,
                "mcp-requests.log",
                &format!("[rpc] completed id={} method={} tool={}", request_id, method, tool_name),
            );
            if tool_name == "exec_command" || tool_name == "exec_health_check" {
                let structured = response
                    .get("result")
                    .and_then(|result| result.get("structuredContent"));
                let status = structured
                    .and_then(|value| value.get("status"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let termination_reason = structured
                    .and_then(|value| value.get("termination_reason"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let exit_code = structured
                    .and_then(|value| value.get("exit_code"))
                    .map(Value::to_string)
                    .unwrap_or_default();
                let is_error = response
                    .get("result")
                    .and_then(|result| result.get("isError"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                append_profile_log(
                    &profile_id,
                    "mcp-requests.log",
                    &format!(
                        "[exec] id={} tool={} is_error={} status={} termination_reason={} exit_code={}",
                        request_id, tool_name, is_error, status, termination_reason, exit_code
                    ),
                );
            }
            if is_upstream_tool {
                let is_error = response
                    .get("result")
                    .and_then(|result| result.get("isError"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                append_profile_log(
                    &profile_id,
                    "mcp-requests.log",
                    &format!(
                        "[upstream] completed id={} tool={} is_error={} duration_ms={}",
                        request_id,
                        tool_name,
                        is_error,
                        started_at.elapsed().as_millis()
                    ),
                );
            }
            Json(response).into_response()
        }
        Err(error) => {
            append_profile_log(
                &profile_id,
                "mcp-requests.log",
                &format!(
                    "[rpc] worker_failed id={} method={} tool={} error={error}",
                    request_id, method, tool_name
                ),
            );
            if is_upstream_tool {
                append_profile_log(
                    &profile_id,
                    "mcp-requests.log",
                    &format!(
                        "[upstream] failed id={} tool={} duration_ms={}",
                        request_id,
                        tool_name,
                        started_at.elapsed().as_millis()
                    ),
                );
            }
            Json(json!({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Exec RPC worker failed",
                    "data": {
                        "stage": "rpc_worker",
                        "reason": "worker_failed",
                        "retryable": true,
                        "suggestion": "重试请求或重启 MCP 运行时"
                    }
                }
            }))
            .into_response()
        }
    }
}

fn require_mcp_auth(state: &ListenerState, headers: &HeaderMap) -> Option<Response> {
    if state.auth.bearer_enabled() {
        let expected = state.bearer_token.as_deref().unwrap_or("");
        return verify_bearer_header(headers, expected);
    }
    if state.auth.oauth_enabled() {
        if let Some(oauth) = state.oauth.as_ref() {
            let server_url = resolve_oauth_base(state, headers);
            return verify_oauth_bearer_header(headers, oauth, &server_url);
        }
    }
    None
}

async fn oauth_authorization_server_metadata(
    State(state): State<ListenerState>,
    headers: HeaderMap,
) -> Response {
    if !state.auth.oauth_enabled() {
        return oauth_not_configured();
    }
    let base = resolve_oauth_base(&state, &headers);
    Json(authorization_server_metadata(
        &base,
        state.oauth_client_secret.as_deref(),
    ))
    .into_response()
}

async fn oauth_protected_resource_metadata(
    State(state): State<ListenerState>,
    headers: HeaderMap,
) -> Response {
    if !state.auth.oauth_enabled() {
        return oauth_not_configured();
    }
    Json(protected_resource_metadata(&resolve_oauth_base(&state, &headers))).into_response()
}

async fn oauth_authorize_get(
    State(state): State<ListenerState>,
    Query(params): Query<AuthorizeParams>,
) -> Response {
    let Some(oauth) = state.oauth.as_ref() else {
        return oauth_not_configured();
    };
    authorize_get(
        oauth,
        params,
        Some(state.workspace_path.as_str()),
    )
}

async fn oauth_authorize_post(
    State(state): State<ListenerState>,
    headers: HeaderMap,
    Form(form): Form<AuthorizeForm>,
) -> Response {
    let Some(oauth) = state.oauth.as_ref() else {
        return oauth_not_configured();
    };
    authorize_post(oauth, form, &resolve_oauth_base(&state, &headers))
}

async fn oauth_token_post(
    State(state): State<ListenerState>,
    headers: HeaderMap,
    Form(form): Form<TokenForm>,
) -> Response {
    let Some(oauth) = state.oauth.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "unsupported_grant_type" })),
        )
            .into_response();
    };
    token_exchange(
        oauth,
        &headers,
        form,
        &resolve_oauth_base(&state, &headers),
    )
}

fn oauth_not_configured() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "OAuth not configured" })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::http::header::CACHE_CONTROL;
    use axum::http::HeaderMap;
    use axum::response::IntoResponse;

    use super::{
        bind_listener, mcp_discovery, mcp_discovery_payload, resolve_oauth_base_from, spawn_listener,
    };
    use crate::auth::PublicBaseUrl;
    use crate::mcp::upstream::UpstreamMcpManager;
    use crate::workspace::{AuthConfig, RuntimeConfig};
    use std::sync::Arc;

    #[test]
    fn bind_listener_reports_port_conflict_synchronously() {
        let occupied = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("占用测试端口");
        let port = occupied.local_addr().expect("读取测试端口").port();

        assert!(bind_listener(port).is_err());
    }

    #[tokio::test]
    async fn discovery_reports_the_current_package_version() {
        let discovery = mcp_discovery_payload();

        assert_eq!(discovery["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn discovery_prevents_stale_tool_catalog_caching() {
        let response = mcp_discovery().await.into_response();

        assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
    }

    #[test]
    fn oauth_base_prefers_the_synced_tunnel_url_over_host_headers() {
        let public_url = PublicBaseUrl::new("https://old.trycloudflare.com");
        let mut headers = HeaderMap::new();
        headers.insert("host", "evil.example.com".parse().unwrap());

        assert_eq!(
            resolve_oauth_base_from(&public_url, 28766, &headers),
            "https://old.trycloudflare.com"
        );

        public_url.set("https://new.trycloudflare.com");
        assert_eq!(
            resolve_oauth_base_from(&public_url, 28766, &headers),
            "https://new.trycloudflare.com"
        );
    }

    #[tokio::test]
    async fn oauth_metadata_endpoint_follows_a_tunnel_domain_change() {
        let dir = tempfile::tempdir().expect("tempdir");
        let port = {
            let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("pick port");
            listener.local_addr().expect("local addr").port()
        };
        let handle = spawn_listener(
            port,
            dir.path().to_path_buf(),
            "workspace-test".into(),
            AuthConfig::default(),
            "https://old.trycloudflare.com".into(),
            None,
            Some("password".into()),
            Some("token-secret".into()),
            RuntimeConfig::default(),
            Arc::new(UpstreamMcpManager::empty()),
        )
        .expect("spawn listener");

        let metadata_url = format!(
            "http://127.0.0.1:{port}/.well-known/oauth-authorization-server"
        );
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("http client");
        let metadata = fetch_metadata(&client, &metadata_url).await;
        assert_eq!(metadata["issuer"], "https://old.trycloudflare.com");

        // The Host header cannot redirect OAuth metadata away from the live URL.
        let spoofed = client
            .get(&metadata_url)
            .header("host", "evil.example.com")
            .send()
            .await
            .expect("spoofed request");
        let spoofed: serde_json::Value = spoofed.json().await.expect("spoofed json");
        assert_eq!(spoofed["issuer"], "https://old.trycloudflare.com");

        handle.public_url.set("https://new.trycloudflare.com");
        let metadata = fetch_metadata(&client, &metadata_url).await;
        assert_eq!(metadata["issuer"], "https://new.trycloudflare.com");
        assert_eq!(
            metadata["authorization_endpoint"],
            "https://new.trycloudflare.com/oauth/authorize"
        );
        assert_eq!(
            metadata["token_endpoint"],
            "https://new.trycloudflare.com/oauth/token"
        );

        let _ = handle.shutdown.send(());
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), handle.join).await;
    }

    async fn fetch_metadata(client: &reqwest::Client, url: &str) -> serde_json::Value {
        for _ in 0..100 {
            if let Ok(response) = client.get(url).send().await {
                if response.status().is_success() {
                    return response.json().await.expect("metadata json");
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("listener did not become ready");
    }
}
