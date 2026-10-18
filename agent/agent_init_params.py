
# Constructor params stored verbatim under the same name.
_PASSTHROUGH_PARAMS = (
    "model", "max_iterations", "save_trajectories", "verbose_logging", "quiet_mode",
    "tool_progress_mode", "ephemeral_system_prompt", "platform", "skip_context_files",
    "load_soul_identity", "pass_session_id", "log_prefix_chars",
    # OpenRouter provider preferences
    "providers_allowed", "providers_ignored", "providers_order", "provider_sort",
    "provider_require_parameters", "provider_data_collection", "openrouter_min_coding_score",
    # Toolset filtering
    "enabled_toolsets", "disabled_toolsets",
    # Model response configuration (None = provider/model default)
    "max_tokens", "reasoning_config", "service_tier",
    "side_agent",
)
# Gateway identity params stored as ``agent._<name>``. gateway_session_key is the stable
# per-chat key (e.g. agent:main:telegram:dm:123).
_GATEWAY_IDENTITY_PARAMS = (
    "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type", "thread_id",
    "gateway_session_key",
)
_CALLBACK_PARAMS = (
    "tool_progress_callback", "tool_start_callback", "tool_complete_callback",
    "tool_result_metadata_callback",
    "thinking_callback", "reasoning_callback", "clarify_callback",
    "read_terminal_callback", "read_preview_callback", "drive_preview_callback",
    "read_window_below_callback", "connection_callback", "tour_callback", "theme_callback",
    "setup_beat_callback", "step_callback", "stream_delta_callback",
    "interim_assistant_callback",
    "status_callback", "notice_callback", "notice_clear_callback",
    "event_callback", "reaction_callback", "tool_gen_callback",
)


def _normalize_run_budget_seconds(value) -> float | None:
    """Positive float or None (feature off). ``bool`` rejected: YAML ``true`` → 1s budget."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None  # NaN compares False → None

def _refuse_checkpoint_required_on_codex_app_server(
    checkpoint_required: bool, api_mode: str | None
) -> None:
    """Fail closed at init: the codex app-server compacts its own thread without a truthful
    pre-compaction boundary (default "native" mode), so a required checkpoint can't be
    guaranteed — the compress_context() guard alone cannot cover native turns."""
    if checkpoint_required and api_mode == "codex_app_server":
        raise RuntimeError(
            "BLOCKED_MISSING_PREREQUISITE: compression.checkpoint_required "
            "is incompatible with the codex_app_server API mode: the codex "
            "agent compacts its own thread without a truthful pre-compaction "
            "transcript boundary, so a required pre-compress checkpoint "
            "cannot be guaranteed. Disable compression.checkpoint_required "
            "or use a non-app-server API mode."
        )
