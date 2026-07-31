from pathlib import Path
from typing import Optional, Dict, Any, List
import os
from models import Session


class PermissionDecision:
    ALLOW_ONCE = "y"
    ALWAYS_TOOL = "a"
    ALWAYS_TARGET = "f"
    ALWAYS_ALL = "all"
    DENY = "n"


class PermissionManager:
    @staticmethod
    def extract_target_key(tool_name: str, raw_target: str) -> str:
        """Extract a clean target key for matching (e.g., 'npm' for bash command 'npm run dev')."""
        target = raw_target.strip()
        if tool_name == "bash":
            # Extract first command token (e.g., 'npm' from 'npm run dev')
            parts = target.split()
            return parts[0] if parts else target
        elif tool_name in ("read", "write", "edit", "grep"):
            # Standardize file/directory path
            try:
                return str(Path(target).resolve())
            except Exception:
                return target
        return target

    @classmethod
    def is_target_allowed(cls, tool_name: str, target: str, allowed_targets: Dict[str, List[str]]) -> bool:
        """Check if target matches any stored allowed target for the given tool."""
        if not allowed_targets or tool_name not in allowed_targets:
            return False

        saved_list = allowed_targets[tool_name]
        clean_target = cls.extract_target_key(tool_name, target)
        target_lower = clean_target.lower()

        for saved in saved_list:
            saved_lower = saved.lower()
            if tool_name == "bash":
                # Matches if command starts with saved prefix or token
                if target_lower.startswith(saved_lower) or target_lower == saved_lower:
                    return True
            elif tool_name in ("read", "write", "edit", "grep"):
                # Exact file path match or directory prefix match
                if target_lower == saved_lower or target_lower.startswith(saved_lower + os.sep):
                    return True
            else:
                if saved_lower in target_lower or target_lower.startswith(saved_lower):
                    return True

        return False

    @classmethod
    def check_permission(
        cls,
        session: Optional[Session],
        tool_name: str,
        target: str,
        autonomous_risk: bool = False
    ) -> bool:
        """
        Returns True if the tool execution is pre-approved without prompting.
        Returns False if interactive prompt confirmation is required.
        """
        # 1. Full Autonomous Risk Mode bypasses all checks
        if autonomous_risk:
            return True

        if not session or not hasattr(session, "permissions"):
            return False

        perms = session.permissions or {}

        # 2. Allow All Tools setting for workspace
        if perms.get("allow_all", False):
            return True

        # 3. Always Allow Tool setting (e.g., all 'read' or all 'web_search')
        if tool_name in perms.get("allowed_tools", []):
            return True

        # 4. Target-specific grant (e.g., specific file path or 'npm' command)
        allowed_targets = perms.get("allowed_targets", {})
        if cls.is_target_allowed(tool_name, target, allowed_targets):
            return True

        return False

    @classmethod
    def save_permission_grant(
        cls,
        memory: Any,
        session: Session,
        grant_type: str,
        tool_name: str,
        target: str
    ) -> None:
        """Persists granted permission into session.permissions and updates metadata.json."""
        if not session:
            return

        if not hasattr(session, "permissions") or session.permissions is None:
            session.permissions = {
                "allow_all": False,
                "allowed_tools": [],
                "allowed_targets": {}
            }

        perms = session.permissions

        if grant_type == PermissionDecision.ALWAYS_ALL:
            perms["allow_all"] = True

        elif grant_type == PermissionDecision.ALWAYS_TOOL:
            if tool_name not in perms["allowed_tools"]:
                perms["allowed_tools"].append(tool_name)

        elif grant_type == PermissionDecision.ALWAYS_TARGET:
            target_key = cls.extract_target_key(tool_name, target)
            if tool_name not in perms["allowed_targets"]:
                perms["allowed_targets"][tool_name] = []
            if target_key not in perms["allowed_targets"][tool_name]:
                perms["allowed_targets"][tool_name].append(target_key)

        # Persist updated session metadata to metadata.json on disk
        try:
            if memory and hasattr(session, "history_path") and session.history_path:
                metadata_file = session.history_path.parent / "metadata.json"
                memory.write_to_json(metadata_file, session)
        except Exception:
            pass
