use tauri::State;

use std::sync::LazyLock;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex as AsyncMutex;

use crate::app_state::AppState;
use crate::error::{AppError, AppResult};
use crate::mcp::upstream::UpstreamMcpManager;
use crate::platform::platform;
use crate::runtime::{
    await_listener_shutdown, port_busy_message, try_reclaim_previous_macos_app_port,
    wait_for_port_free, ServiceKind,
};
use crate::tunnel::{
    append_profile_log, current_running_url, maybe_start_for_runtime, stop_for_runtime,
    sync_managed_runtime_routes, TunnelServiceKind,
};
use crate::workspace::resources::{validate_service_start, WorkspaceService};
use crate::workspace::RuntimeStatusDto;

/// Serialize MCP/Actions restarts so secret-save and form-save cannot tear down
/// the same listener concurrently (that race could abort the process on Windows).
static RESTART_GATE: LazyLock<AsyncMutex<()>> = LazyLock::new(|| AsyncMutex::new(()));

fn profile_by_id(state: &AppState, id: &str) -> AppResult<crate::workspace::WorkspaceProfile> {
    state.with_workspaces(|store| {
        store
            .get(id)
            .cloned()
            .ok_or_else(|| AppError::Message(format!("workspace not found: {id}")))
    })
}

fn validate_start_resources(
    state: &AppState,
    id: &str,
    service: WorkspaceService,
) -> AppResult<()> {
    state.with_workspaces(|store| validate_service_start(store.list(), id, service))
}

fn persist_tunnel_url(
    state: &AppState,
    id: &str,
    kind: TunnelServiceKind,
    url: &str,
) -> AppResult<()> {
    if url.is_empty() {
        return Ok(());
    }

    state.with_workspaces(|store| {
        let Some(mut profile) = store.get(id).cloned() else {
            return Ok(());
        };

        match kind {
            TunnelServiceKind::Mcp => profile.tunnel.public_url = url.to_string(),
            TunnelServiceKind::Actions => profile.actions.public_url = url.to_string(),
        }

        store.update(profile)?;
        Ok(())
    })
}

async fn sync_tunnel_routes_from_runtime(state: &AppState) -> AppResult<()> {
    let active_keys = state.with_runtime(|runtime| Ok(runtime.active_tunnel_service_keys()))?;
    sync_managed_runtime_routes(active_keys).await
}

/// A live tunnel owns the public address. Quick Tunnels hand out a new
/// hostname on every launch, so the persisted URL can lag behind reality.
fn preferred_public_url(running_url: Option<&str>, profile: &crate::workspace::WorkspaceProfile) -> String {
    match running_url.map(crate::auth::normalize_public_url) {
        Some(url) if !url.is_empty() => url,
        _ => profile.effective_public_url(),
    }
}

/// Persist the URL shown to GPT and push it into the live MCP listener so its
/// OAuth metadata stops advertising a retired tunnel.
fn apply_authoritative_mcp_url(state: &AppState, id: &str, url: &str) -> AppResult<()> {
    let url = crate::auth::normalize_public_url(url);
    if url.is_empty() {
        return Ok(());
    }
    persist_tunnel_url(state, id, TunnelServiceKind::Mcp, &url)?;
    state.with_runtime(|runtime| {
        runtime.set_mcp_public_url(id, &url);
        Ok(())
    })
}

fn mcp_start_failure(
    state: &AppState,
    profile: &crate::workspace::WorkspaceProfile,
    message: String,
) -> AppResult<RuntimeStatusDto> {
    append_profile_log(&profile.id, "stderr.log", &format!("[start] {message}"));
    state.with_runtime(|runtime| {
        runtime.set_mcp_error(profile, message);
        Ok(runtime.mcp_status(profile))
    })
}

#[allow(clippy::collapsible_if)]
async fn ensure_port_available(port: u16, service_label: &str) -> AppResult<()> {
    let Some(pid) = platform().find_pid_listening_on_port(port)? else {
        return Ok(());
    };

    if crate::runtime::is_own_process(pid) {
        if wait_for_port_free(port, Duration::from_secs(3)).await {
            return Ok(());
        }
    }

    if try_reclaim_previous_macos_app_port(port) {
        return Ok(());
    }

    if let Some(pid) = platform().find_pid_listening_on_port(port)? {
        return Err(AppError::Message(port_busy_message(
            port,
            service_label,
            pid,
        )));
    }

    Ok(())
}

async fn stop_mcp_service(
    state: &AppState,
    id: &str,
    stop_tunnel: bool,
) -> AppResult<RuntimeStatusDto> {
    let profile = profile_by_id(state, id)?;
    let port = profile.runtime.local_port;
    let handle = state.with_runtime(|runtime| Ok(runtime.begin_stop(id, ServiceKind::Mcp)))?;
    await_listener_shutdown(handle, port).await;
    state.with_runtime(|runtime| {
        runtime.finish_stop(id, ServiceKind::Mcp);
        Ok(runtime.mcp_status(&profile))
    })?;
    // Restarting only the listener must keep the tunnel alive: recreating a
    // Quick Tunnel would rotate its hostname for no reason.
    if stop_tunnel {
        stop_for_runtime(&profile, TunnelServiceKind::Mcp).await?;
        sync_tunnel_routes_from_runtime(state).await?;
    }
    state.with_runtime(|runtime| Ok(runtime.mcp_status(&profile)))
}

async fn start_mcp_service(state: &AppState, id: &str) -> AppResult<RuntimeStatusDto> {
    validate_start_resources(state, id, WorkspaceService::Mcp)?;
    let mut profile = profile_by_id(state, id)?;

    // Consult the tunnel that is actually alive before spawning so a restarted
    // listener never boots against the hostname the tunnel already retired.
    let live_url = current_running_url(id, TunnelServiceKind::Mcp).await;
    if let Some(url) = live_url.as_deref() {
        profile.tunnel.public_url = crate::auth::normalize_public_url(url);
        apply_authoritative_mcp_url(state, id, &profile.tunnel.public_url)?;
    }

    if state.with_runtime(|runtime| Ok(runtime.is_running(id, ServiceKind::Mcp)))? {
        return state.with_runtime(|runtime| Ok(runtime.mcp_status(&profile)));
    }
    if let Err(error) = ensure_port_available(profile.runtime.local_port, "本地 MCP").await {
        return mcp_start_failure(state, &profile, error.to_string());
    }
    let core_tools = crate::tools::registry::exposed_tool_names(&profile.runtime.tool_profile);
    let upstream = match UpstreamMcpManager::start(&profile.runtime.upstream_mcps, &core_tools).await {
        Ok(upstream) => Arc::new(upstream),
        Err(error) => {
            let message = format!("本地 MCP 上游初始化失败：{error}");
            return mcp_start_failure(state, &profile, message);
        }
    };
    let started = match state.with_runtime(|runtime| {
        runtime.start_prepared_mcp(&profile, upstream.clone())
    }) {
        Ok(status) => status,
        Err(error) => {
            upstream.shutdown().await;
            return mcp_start_failure(state, &profile, error.to_string());
        }
    };
    if started.state != "running" {
        upstream.shutdown().await;
        return Ok(started);
    }
    sync_tunnel_routes_from_runtime(state).await?;

    let started_url = match maybe_start_for_runtime(&profile, TunnelServiceKind::Mcp).await {
        Ok(url) => url,
        Err(error) => {
            eprintln!("mcp tunnel auto-start failed for {id}: {error}");
            None
        }
    };
    let authoritative = preferred_public_url(started_url.as_deref(), &profile);
    apply_authoritative_mcp_url(state, id, &authoritative)?;

    let profile = profile_by_id(state, id)?;
    tokio::time::sleep(Duration::from_millis(250)).await;
    state.with_runtime(|runtime| {
        runtime.refresh_mcp(&profile);
        Ok(runtime.mcp_status(&profile))
    })
}

async fn stop_actions_service(state: &AppState, id: &str) -> AppResult<RuntimeStatusDto> {
    let profile = profile_by_id(state, id)?;
    let port = profile.actions.local_port;
    let handle = state.with_runtime(|runtime| Ok(runtime.begin_stop(id, ServiceKind::Actions)))?;
    await_listener_shutdown(handle, port).await;
    state.with_runtime(|runtime| {
        runtime.finish_stop(id, ServiceKind::Actions);
        Ok(runtime.actions_status(&profile))
    })?;
    stop_for_runtime(&profile, TunnelServiceKind::Actions).await?;
    sync_tunnel_routes_from_runtime(state).await?;
    state.with_runtime(|runtime| Ok(runtime.actions_status(&profile)))
}

async fn start_actions_service(state: &AppState, id: &str) -> AppResult<RuntimeStatusDto> {
    validate_start_resources(state, id, WorkspaceService::Actions)?;
    let profile = profile_by_id(state, id)?;
    ensure_port_available(profile.actions.local_port, "本地 Actions").await?;
    state.with_runtime(|runtime| runtime.start_actions(&profile))?;
    sync_tunnel_routes_from_runtime(state).await?;

    match maybe_start_for_runtime(&profile, TunnelServiceKind::Actions).await {
        Ok(Some(url)) => {
            persist_tunnel_url(state, id, TunnelServiceKind::Actions, &url)?;
        }
        Ok(None) => {}
        Err(error) => {
            eprintln!("actions tunnel auto-start failed for {id}: {error}");
        }
    }

    let profile = profile_by_id(state, id)?;
    tokio::time::sleep(Duration::from_millis(250)).await;
    state.with_runtime(|runtime| {
        runtime.refresh_actions(&profile);
        Ok(runtime.actions_status(&profile))
    })
}

/// Async stop→start for MCP. Used by the Tauri command and secret-change hooks.
/// The tunnel stays up across the restart so its address and OAuth audience
/// remain stable; it is only rotated when the listener moves to a new port.
pub(crate) async fn restart_mcp_by_id(
    state: &AppState,
    id: &str,
) -> AppResult<RuntimeStatusDto> {
    let _guard = RESTART_GATE.lock().await;
    let port = profile_by_id(state, id)?.runtime.local_port;
    let (was_running, running_port) = state.with_runtime(|runtime| {
        Ok((
            runtime.is_running(id, ServiceKind::Mcp),
            runtime.running_local_port(id, ServiceKind::Mcp),
        ))
    })?;
    if was_running {
        // A tunnel still forwards to the previous port; recreating it is the
        // only way to reach the new one.
        let port_changed = running_port.is_some_and(|bound| bound != port);
        let _ = stop_mcp_service(state, id, port_changed).await?;
    }
    start_mcp_service(state, id).await
}

/// Async stop→start for Actions. Used by the Tauri command and secret-change hooks.
pub(crate) async fn restart_actions_by_id(
    state: &AppState,
    id: &str,
) -> AppResult<RuntimeStatusDto> {
    let _guard = RESTART_GATE.lock().await;
    let was_running = state.with_runtime(|runtime| {
        Ok(runtime.is_running(id, ServiceKind::Actions))
    })?;
    if was_running {
        let _ = stop_actions_service(state, id).await?;
    }
    start_actions_service(state, id).await
}

#[tauri::command]
pub async fn start_runtime(state: State<'_, AppState>, id: String) -> AppResult<RuntimeStatusDto> {
    start_mcp_service(&state, &id).await
}

#[tauri::command]
pub async fn stop_runtime(state: State<'_, AppState>, id: String) -> AppResult<RuntimeStatusDto> {
    stop_mcp_service(&state, &id, true).await
}

#[tauri::command]
pub fn get_runtime_status(state: State<'_, AppState>, id: String) -> AppResult<RuntimeStatusDto> {
    let profile = profile_by_id(&state, &id)?;
    state.with_runtime(|runtime| {
        runtime.refresh_mcp(&profile);
        Ok(runtime.mcp_status(&profile))
    })
}

#[tauri::command]
pub async fn start_actions_runtime(
    state: State<'_, AppState>,
    id: String,
) -> AppResult<RuntimeStatusDto> {
    start_actions_service(&state, &id).await
}

#[tauri::command]
pub async fn stop_actions_runtime(
    state: State<'_, AppState>,
    id: String,
) -> AppResult<RuntimeStatusDto> {
    stop_actions_service(&state, &id).await
}

#[tauri::command]
pub fn get_actions_runtime_status(
    state: State<'_, AppState>,
    id: String,
) -> AppResult<RuntimeStatusDto> {
    let profile = profile_by_id(&state, &id)?;
    state.with_runtime(|runtime| {
        runtime.refresh_actions(&profile);
        Ok(runtime.actions_status(&profile))
    })
}

#[tauri::command]
pub async fn restart_runtime(
    state: State<'_, AppState>,
    id: String,
) -> AppResult<RuntimeStatusDto> {
    restart_mcp_by_id(&state, &id).await
}

#[tauri::command]
pub async fn restart_actions_runtime(
    state: State<'_, AppState>,
    id: String,
) -> AppResult<RuntimeStatusDto> {
    restart_actions_by_id(&state, &id).await
}

#[cfg(test)]
mod tests {
    use super::preferred_public_url;
    use crate::workspace::WorkspaceProfile;

    fn quick_tunnel_profile(public_url: &str) -> WorkspaceProfile {
        let mut profile = WorkspaceProfile::new("C:/workspace/test".into(), Some("test".into()));
        profile.tunnel.tunnel_type = "cloudflare".into();
        profile.tunnel.public_url = public_url.into();
        profile
    }

    #[test]
    fn running_tunnel_url_wins_over_the_cached_profile_url() {
        let profile = quick_tunnel_profile("https://old.trycloudflare.com");
        assert_eq!(
            preferred_public_url(Some("https://new.trycloudflare.com/"), &profile),
            "https://new.trycloudflare.com"
        );
    }

    #[test]
    fn cached_url_is_used_when_no_tunnel_is_running() {
        let profile = quick_tunnel_profile("https://old.trycloudflare.com/");
        assert_eq!(
            preferred_public_url(None, &profile),
            "https://old.trycloudflare.com"
        );
    }

    #[test]
    fn blank_running_url_falls_back_to_the_cached_url() {
        let profile = quick_tunnel_profile("https://old.trycloudflare.com");
        assert_eq!(
            preferred_public_url(Some("  "), &profile),
            "https://old.trycloudflare.com"
        );
    }
}
