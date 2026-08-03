from mistralai.client import Mistral
from openai import OpenAI
import os
import prompts
from enum import Enum
from dotenv import load_dotenv
from models import Role,Message, Session
from tools import TOOLS, execute_read, execute_write, execute_edit, execute_bash, execute_web_search, execute_grep
from console import get_console
from config import Config, estimate_cost, BUILTIN_PROVIDERS
from memory import Memory
from dataclasses import asdict
load_dotenv()
import json
import threading
from skills import Skills
from permissions import PermissionManager, PermissionDecision
from typing import Any, Optional
from interrupt import AgentInterrupted, interrupt_controller
from compaction import Compaction


def sanitize_api_messages(raw_messages: list[dict]) -> list[dict]:
    """
    Sanitizes message history to guarantee Mistral API message ordering rules:
    1. Every assistant message with tool_calls must be followed by matching tool response messages for each tool_call_id.
    2. Missing tool responses are filled with a placeholder response so tool_call counts match.
    3. Orphan tool response messages (without a preceding matching assistant tool call) are removed.
    """
    sanitized: list[dict] = []
    i = 0
    n = len(raw_messages)

    while i < n:
        msg = raw_messages[i]
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls") or []
            expected_ids = []
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    expected_ids.append(tc_id)

            sanitized.append(msg)
            i += 1

            tool_responses_by_id = {}
            while i < n and raw_messages[i].get("role") == "tool":
                tool_msg = raw_messages[i]
                t_id = tool_msg.get("tool_call_id")
                if t_id:
                    tool_responses_by_id[t_id] = tool_msg
                i += 1

            for t_id in expected_ids:
                if t_id in tool_responses_by_id:
                    sanitized.append(tool_responses_by_id[t_id])
                else:
                    sanitized.append({
                        "role": "tool",
                        "content": "Tool execution was interrupted.",
                        "tool_call_id": t_id
                    })
        elif role == "tool":
            i += 1
        else:
            sanitized.append(msg)
            i += 1

    return sanitized


def apply_sliding_window(raw_messages: list[dict], max_history: int = 20) -> list[dict]:
    """
    Applies a sliding window to conversation history:
    1. Always preserves the System prompt (index 0).
    2. Keeps the most recent messages up to max_history.
    3. Aligns the window start to a User turn to avoid breaking tool call contexts.
    4. Passes the result through sanitize_api_messages to guarantee API compliance.
    """
    if len(raw_messages) <= max_history:
        return sanitize_api_messages(raw_messages)

    sys_msg = None
    history = raw_messages

    if raw_messages and raw_messages[0].get("role") == "system":
        sys_msg = raw_messages[0]
        history = raw_messages[1:]

    target_count = max_history - (1 if sys_msg else 0)
    start_idx = max(0, len(history) - target_count)

    orig_start = start_idx
    while start_idx < len(history) and history[start_idx].get("role") != "user":
        start_idx += 1

    if start_idx >= len(history):
        start_idx = orig_start

    window = [sys_msg] + history[start_idx:] if sys_msg else history[start_idx:]
    return sanitize_api_messages(window)


class Agent:
    def __init__(self):
        config = Config()
        self.config = config
        self.console = get_console() 
        self.client = self.create_model()
        self.prompt = prompts.Prompt()
        self.memory: Memory = Memory() 
        self.memory.messages = [Message(role=Role.SYSTEM, content=self.prompt.prompts[0])]
        self._pending_prompt_tokens = 0
        self._pending_completion_tokens = 0
        self._pending_total_tokens = 0
        # None = auto skill selection via LLM; list = manual (/skills), skip LLM picker
        self.manual_skill_names: Optional[list[str]] = None

    def reset_conversation(self) -> None:
        """Drop the active session and start a fresh in-memory conversation."""
        self.memory.session = None
        if hasattr(self, "current_session"):
            self.current_session = None  # type: ignore[attr-defined]
        self._pending_prompt_tokens = 0
        self._pending_completion_tokens = 0
        self._pending_total_tokens = 0
        self.memory.messages = [
            Message(role=Role.SYSTEM, content=self.prompt.raw_system_prompt)
        ]
        # Keep manual skill preference across /new; re-apply into the fresh system prompt
        if self.manual_skill_names:
            self.apply_active_skills(list(self.manual_skill_names), announce=False)

    def apply_active_skills(self, skill_names: list[str], *, announce: bool = True) -> None:
        """Load skills into the system prompt (or reset to base when empty)."""
        if skill_names:
            active_skills = Skills.load_many(skill_names)
            sys_prompt = self.prompt.get_system_prompt(active_skills)
        else:
            sys_prompt = self.prompt.raw_system_prompt

        if self.memory.messages and self.memory.messages[0].role == Role.SYSTEM:
            self.memory.messages[0].content = sys_prompt
            if self.memory.session:
                self.memory.write_to_jsonl(
                    self.memory.session.history_path,
                    self.memory.messages,
                    mode="w",
                )

        if announce:
            if skill_names:
                self.console.console.print(
                    f"[bold cyan]Active Skills:[/bold cyan] "
                    f"[yellow]{', '.join(skill_names)}[/yellow]"
                )
            else:
                self.console.console.print(
                    "[dim]No skills loaded into the system prompt.[/dim]"
                )

    def _persist_session_usage(self) -> None:
        session = self.memory.session
        if not session:
            return
        meta_path = session.history_path.parent / "metadata.json"
        self.memory.write_to_json(meta_path, session)

    def _flush_pending_usage(self) -> None:
        session = self.memory.session
        if not session:
            return
        if not (
            self._pending_total_tokens
            or self._pending_prompt_tokens
            or self._pending_completion_tokens
        ):
            return
        session.prompt_tokens += self._pending_prompt_tokens
        session.completion_tokens += self._pending_completion_tokens
        session.total_tokens += self._pending_total_tokens
        session.estimated_cost_usd += estimate_cost(
            self._pending_prompt_tokens,
            self._pending_completion_tokens,
            self.config.input_price_per_mtok,
            self.config.output_price_per_mtok,
        )
        self._pending_prompt_tokens = 0
        self._pending_completion_tokens = 0
        self._pending_total_tokens = 0
        self._persist_session_usage()

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        if prompt == 0 and completion == 0 and total == 0:
            return
        session = self.memory.session
        if session is None:
            self._pending_prompt_tokens += prompt
            self._pending_completion_tokens += completion
            self._pending_total_tokens += total
            return
        session.prompt_tokens += prompt
        session.completion_tokens += completion
        session.total_tokens += total
        session.estimated_cost_usd += estimate_cost(
            prompt,
            completion,
            self.config.input_price_per_mtok,
            self.config.output_price_per_mtok,
        )
        self._persist_session_usage()

    def print_session_usage(self) -> None:
        session = self.memory.session
        if not session:
            return
        self.console.print_usage_summary(
            session.prompt_tokens,
            session.completion_tokens,
            session.total_tokens,
            session.estimated_cost_usd,
        )
    @property
    def model_name(self) -> str:
        model_str = str(self.config.model.value) if hasattr(self.config.model, "value") else str(self.config.model)
        return model_str

    def list_available_models(self) -> list[str]:
        """Fetch model IDs from the active provider's OpenAI-compatible /models endpoint."""
        if not self.config.api_key:
            raise RuntimeError("No API key configured. Run /login first.")
        base_url = self.config.base_url or BUILTIN_PROVIDERS.get(
            self.config.provider, {}
        ).get("base_url", "https://api.mistral.ai/v1")
        client = self.client or OpenAI(api_key=self.config.api_key, base_url=base_url)
        response = client.models.list()
        ids = []
        for m in getattr(response, "data", []) or []:
            mid = getattr(m, "id", None)
            if mid:
                ids.append(str(mid))
        return sorted(set(ids), key=str.lower)

    def apply_provider_runtime(self) -> None:
        """Reload config from auth.json and rebuild the OpenAI client."""
        self.config.reload_from_auth()
        self.client = self.create_model()


    def select_relevant_skills(self, user_query: str) -> list[str]:
        
        available = Skills.names()
        if not available:
            return []
        prompt_str = (
            f"You are a skill selection system for an AI coding assistant.\n"
            f"User Task: \"{user_query}\"\n\n"
            f"Available Skills: {available}\n\n"
            f"Select which skill names from the Available Skills list are relevant for fulfilling the user task.\n"
            f"Max 3 skills to be selected.\n"
            f"Return ONLY a JSON array of matching skill names, e.g. [\"react\", \"git\"]. If none are relevant, return []."
        )

        try:
            interrupt_controller.check()
            response = self._create_completion(
                messages=[{"role": "user", "content": prompt_str}],
                use_tools=False,
            )
            self._record_usage(response)
            content = (response.choices[0].message.content or "").strip()
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list):
                    return [s for s in parsed if isinstance(s, str) and s in available]
            return []
        except AgentInterrupted:
            raise
        except Exception:
            q_lower = user_query.lower()
            return [s for s in available if s.lower() in q_lower]

    def _append_message(self, msg: Message) -> None:
        """Append a message to in-memory history and persist to the session JSONL."""
        self.memory.messages.append(msg)
        if self.memory and self.memory.session:
            self.memory.write_to_jsonl(self.memory.session.history_path, [msg], mode="a")

    def _compaction(self) -> Compaction:
        return Compaction(
            compact_at_tokens=self.config.compact_at_tokens,
            keep_messages=self.config.compact_keep_messages,
            provider=self.config.provider,
        )

    def _build_api_messages(self) -> list[dict]:
        """System + optional summary + recent raw messages, then sliding window."""
        comp = self._compaction()
        working = comp.working_messages(self.memory.messages, self.memory.session)
        raw_dicts = [m.to_dict() for m in working]
        return apply_sliding_window(
            raw_dicts, max_history=self.config.max_history_messages
        )

    def _maybe_compact(self) -> None:
        """Auto path: only runs when enabled and over threshold."""
        self.run_compaction(force=False)

    def run_compaction(self, *, force: bool = False) -> str:
        """
        Summarize newly aged messages into session.compaction_summary.

        force=True: run even if under threshold / auto-disabled (manual /compact).
        Returns a short status string for the caller to show.
        """
        session = self.memory.session
        if not session:
            return "No active session. Send a task first (or /resume)."

        if not force and not self.config.compaction_enabled:
            return "Compaction is disabled. Enable it with /compaction or pass force."

        comp = self._compaction()
        if not force and not comp.should_compact(self.memory.messages, session):
            return "Below threshold — nothing to compact."

        # Over budget (or manual): may shrink keep window for few huge turns
        allow_shrink = force or comp.over_token_budget(self.memory.messages, session)
        keep = comp.effective_keep_messages(
            self.memory.messages,
            session,
            allow_shrink=allow_shrink,
        )
        planned = comp.plan_segment(
            self.memory.messages,
            session,
            keep_messages=keep,
        )
        if not planned:
            return (
                "Nothing new to fold — recent messages already cover the keep window "
                f"(keep={keep}, configured={self.config.compact_keep_messages})."
            )
        segment, new_until, keep_used = planned

        interrupt_controller.check()
        shrink_note = (
            f", keep shrunk {self.config.compact_keep_messages}→{keep_used}"
            if keep_used < self.config.compact_keep_messages
            else ""
        )
        self.console.console.print(
            "[dim]Compacting older context "
            f"({len(segment)} message(s) → summary{shrink_note})...[/dim]"
        )
        prompt = comp.build_prompt(session.compaction_summary, segment)
        try:
            response = self._create_completion(
                messages=[{"role": "user", "content": prompt}],
                use_tools=False,
            )
            self._record_usage(response)
            summary = (response.choices[0].message.content or "").strip()
        except AgentInterrupted:
            raise
        except Exception as e:
            self.console.print_error(
                f"Compaction failed: {e}",
                title="Compaction",
            )
            return f"Compaction failed: {e}"

        if not summary:
            return "Summarizer returned empty text — left context unchanged."

        session.compaction_summary = summary
        session.compacted_until = new_until
        self._persist_session_usage()
        msg = (
            f"Compacted through message {new_until} "
            f"({len(segment)} folded; keep={keep_used}; prior summary included)."
        )
        self.console.console.print(f"[dim]{msg}[/dim]")
        return msg

    def _record_error(self, error: BaseException, title: str = "LLM Error") -> Message:
        """Show an error in the console and persist it to conversation history."""
        detail = f"{type(error).__name__}: {error}"
        self.console.print_error(detail, title=title)
        err_msg = Message(
            role=Role.ASSISTANT,
            content=f"[{title}] {detail}",
        )
        self._append_message(err_msg)
        return err_msg

    def _finalize_pending_tool_calls(self) -> None:
        """
        If the last assistant turn requested tools but some responses are missing
        (e.g. interrupted mid-loop), write placeholder tool messages so history
        stays valid for the next API call.
        """
        messages = self.memory.messages
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role == Role.ASSISTANT and getattr(msg, "tool_calls", None):
                expected_ids: list[str] = []
                for tc in msg.tool_calls or []:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        expected_ids.append(tc_id)

                answered: set[str] = set()
                for later in messages[i + 1 :]:
                    if later.role != Role.TOOL:
                        break
                    tid = getattr(later, "tool_call_id", None)
                    if tid:
                        answered.add(tid)

                for t_id in expected_ids:
                    if t_id not in answered:
                        self._append_message(
                            Message(
                                role=Role.TOOL,
                                content="Tool execution was interrupted by the user.",
                                tool_call_id=t_id,
                            )
                        )
                return
            if msg.role == Role.USER:
                return

    def _handle_interrupt(self) -> None:
        """Persist interrupt marker and notify the user."""
        self._finalize_pending_tool_calls()
        interrupt_msg = Message(
            role=Role.USER,
            content="[Interrupted] Execution stopped by user.",
        )
        self._append_message(interrupt_msg)
        self.console.stop_loading()
        self.console.print_system_message(
            "Stopped current execution. Press Enter for a new task.",
            title="Interrupted",
        )

    def _create_completion(
        self,
        messages: list[dict],
        use_tools: bool = True,
    ) -> Any:
        """
        Run chat.completions.create in a worker thread so ESC can interrupt
        while waiting on the network response.
        On HTTP 429, rotate to the secondary API key (if configured) and retry once.
        """
        from config import rotate_provider_key

        def _is_rate_limit(exc: BaseException) -> bool:
            name = type(exc).__name__
            if "RateLimit" in name:
                return True
            status = getattr(exc, "status_code", None)
            if status == 429:
                return True
            resp = getattr(exc, "response", None)
            if resp is not None and getattr(resp, "status_code", None) == 429:
                return True
            msg = str(exc).lower()
            return "rate limit" in msg or "429" in msg

        def _run_once() -> tuple[Any, Optional[BaseException]]:
            result: dict[str, Any] = {"response": None, "error": None}

            def _run() -> None:
                try:
                    kwargs: dict[str, Any] = {
                        "model": self.model_name,
                        "messages": messages,
                    }
                    if use_tools:
                        kwargs["tools"] = TOOLS
                    if not self.client:
                        result["error"] = RuntimeError(
                            "No LLM client configured. Run /login first."
                        )
                        return
                    result["response"] = self.client.chat.completions.create(**kwargs)
                except Exception as exc:
                    result["error"] = exc

            worker = threading.Thread(target=_run, name="llm-completion", daemon=True)
            worker.start()
            while worker.is_alive():
                if interrupt_controller.interrupted:
                    raise AgentInterrupted("Execution stopped by user")
                worker.join(timeout=0.1)
            return result["response"], result["error"]

        response, error = _run_once()
        if error is None:
            return response

        if _is_rate_limit(error):
            new_key = rotate_provider_key(self.config.provider)
            if new_key:
                self.apply_provider_runtime()
                slot = "Primary" if self.config.active_key_index == 0 else "Secondary"
                self.console.console.print(
                    f"[dim]Switched to {slot} API key after rate limit, retrying...[/dim]"
                )
                response, error = _run_once()
                if error is None:
                    return response

        raise error

    def chat(self, user_query: str) -> Optional[Any]:
        interrupt_controller.start()
        try:
            # Skills: manual (/skills) skips the LLM skill-selection call
            available_skills = Skills.names()
            if available_skills:
                interrupt_controller.check()
                if self.manual_skill_names is not None:
                    selected_names = [
                        n for n in self.manual_skill_names if n in available_skills
                    ]
                else:
                    selected_names = self.select_relevant_skills(user_query)
                if selected_names:
                    # Auto mode: announce each turn; manual already announced in /skills
                    self.apply_active_skills(
                        selected_names,
                        announce=self.manual_skill_names is None,
                    )
                elif self.manual_skill_names is not None:
                    # Manual mode with empty selection: ensure base system prompt
                    self.apply_active_skills([], announce=False)

            user_msg = Message(role=Role.USER, content=user_query)
            self._append_message(user_msg)

            while True:
                interrupt_controller.check()
                self._maybe_compact()
                api_messages = self._build_api_messages()

                try:
                    res = self._create_completion(api_messages, use_tools=True)
                except AgentInterrupted:
                    raise
                except Exception as e:
                    self._record_error(e, title="LLM Error")
                    return None

                if not res or not res.choices:
                    self._record_error(
                        RuntimeError("LLM returned no choices"),
                        title="LLM Error",
                    )
                    return None

                self._record_usage(res)
                llm_res = res.choices[0]

                tool_calls_raw = llm_res.message.tool_calls
                if tool_calls_raw:
                    tool_calls_dicts = [tc.model_dump() for tc in tool_calls_raw]
                else:
                    tool_calls_dicts = None

                chat_msg = Message(
                    role=Role.ASSISTANT,
                    content=llm_res.message.content or "",
                    tool_calls=tool_calls_dicts,
                )
                self._append_message(chat_msg)

                if llm_res.message.tool_calls:  # type: ignore
                    for tool in llm_res.message.tool_calls:  # type: ignore
                        interrupt_controller.check()
                        tool_name = tool.function.name  # type: ignore
                        tool_arguments = tool.function.arguments  # type: ignore
                        self.console.print_tool_call(tool_name, tool_arguments)  # type: ignore
                        try:
                            fn_output = self.dispatch_tool_call(tool_name, tool_arguments)  # type: ignore
                        except AgentInterrupted:
                            raise
                        except Exception as e:
                            fn_output = f"Error executing tool {tool_name}: {str(e)}"
                            self.console.print_error(fn_output, title="Tool Error")
                        self.console.print_tool_result(fn_output)
                        tool_msg = Message(
                            role=Role.TOOL,
                            name=tool_name,
                            content=fn_output,
                            tool_call_id=tool.id,
                        )
                        self._append_message(tool_msg)
                else:
                    return res.choices[0]
        except AgentInterrupted:
            self._handle_interrupt()
            return None
        except KeyboardInterrupt:
            interrupt_controller.trigger()
            self._handle_interrupt()
            return None
        finally:
            interrupt_controller.stop()

    def send(self, user_query):
        return self.chat(user_query)

    def create_model(self):
        if not self.config.api_key:
            return None
        endpoint = (
            self.config.base_url
            or BUILTIN_PROVIDERS.get(self.config.provider, {}).get("base_url")
            or "https://api.mistral.ai/v1"
        )
        return OpenAI(
            api_key=self.config.api_key,
            base_url=endpoint,
        )
    def check_and_request_permission(self, tool_name: str, target: str, action_details: str) -> bool:
        """Checks if permission is pre-approved, otherwise prompts the user with persistent options."""
        if PermissionManager.check_permission(self.memory.session, tool_name, target, self.config.autonomous_risk):
            return True

        # Pause ESC capture so the permission prompt receives keystrokes
        interrupt_controller.pause()
        try:
            interrupt_controller.check()
            choice = self.console.confirm_permission_extended(tool_name, target, action_details)
        finally:
            interrupt_controller.resume()

        if choice in (PermissionDecision.ALLOW_ONCE, PermissionDecision.ALWAYS_TOOL, PermissionDecision.ALWAYS_TARGET, PermissionDecision.ALWAYS_ALL):
            if choice != PermissionDecision.ALLOW_ONCE and self.memory and self.memory.session:
                PermissionManager.save_permission_grant(self.memory, self.memory.session, choice, tool_name, target)
            return True
        return False

    def dispatch_tool_call(self, tool_name: str, function_arguments: str):
        args = json.loads(function_arguments)
        if tool_name == "read":
            path = args.get("path", "")
            if not self.check_and_request_permission(tool_name, path, f"Agent wants to read {path}"):
                return "User permission denied"
            return execute_read(path)

        elif tool_name == "write":
            path = args.get("path", "")
            if not self.check_and_request_permission(tool_name, path, f"Agent wants to write to {path}"):
                return "User permission denied"
            return execute_write(path, args.get("content", ""))

        elif tool_name == "edit":
            path = args.get("path", "")
            if not self.check_and_request_permission(tool_name, path, f"Agent wants to edit {path}"):
                return "User permission denied"
            return execute_edit(path, args.get("edits", []))

        elif tool_name == "bash":
            cmd = args.get("command", "")
            timeout = args.get("timeout", 30)
            is_bg = args.get("is_background", False)
            bg_note = " (background)" if is_bg else ""
            if not self.check_and_request_permission(tool_name, cmd, f"Agent wants to run bash{bg_note}: {cmd}"):
                return "User permission denied"
            return execute_bash(cmd, timeout=timeout, is_background=is_bg)

        elif tool_name == "web_search":
            query = args.get("query", "")
            max_results = args.get("max_results", 5)
            if not self.check_and_request_permission(tool_name, query, f"Agent wants to run web_search: '{query}'"):
                return "User permission denied"
            return execute_web_search(query, max_results=max_results)

        elif tool_name == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", ".") or "."
            glob = args.get("glob", "") or ""
            case_insensitive = bool(args.get("case_insensitive", False))
            max_results = args.get("max_results", 50)
            if not self.check_and_request_permission(
                tool_name, path, f"Agent wants to grep '{pattern}' in {path}"
            ):
                return "User permission denied"
            return execute_grep(
                pattern=pattern,
                path=path,
                glob=glob,
                case_insensitive=case_insensitive,
                max_results=max_results,
            )

        else:
            return f"Unknown tool: {tool_name}"


