import json
import logging
from pathlib import Path
from typing import Optional

from cron.env_settings import cron_env_setting

logger = logging.getLogger("cron.scheduler")


def _load_prefill_messages(cfg: dict, job_id: str, hermes_home: Path) -> Optional[list]:
    """Prefill messages from env or config.yaml (top-level key canonical; agent.* is legacy)."""
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg.get("agent", {}), dict) else {}
    prefill_file = (
        cron_env_setting("HERMES_PREFILL_MESSAGES_FILE")
        or cfg.get("prefill_messages_file", "")
        or agent_cfg.get("prefill_messages_file", "")
    )
    if not prefill_file:
        return None
    pfpath = Path(prefill_file).expanduser()
    if not pfpath.is_absolute():
        pfpath = hermes_home / pfpath
    if not pfpath.exists():
        return None
    try:
        with open(pfpath, "r", encoding="utf-8-sig") as _pf:
            prefill_messages = json.load(_pf)
        return prefill_messages if isinstance(prefill_messages, list) else None
    except Exception as e:
        logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
        return None
