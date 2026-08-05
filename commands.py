"""Slash-command handlers for the PI agent REPL."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from llm import Agent
from tokenizer import count_messages, tokens_by_role

CommandResult = Union[bool, str, None]


class Commands:
    """All user-facing slash commands live here."""

    ALIASES: Dict[str, str] = {
        "exit": "/exit",
        "quit": "/exit",
        "help": "/help",
        "token": "/tokens",
        "tokenizer": "/tokens",
        "price": "/prices",
        "pricing": "/prices",
        "max_history": "/history",
    }

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.commands = self.get_methods()
        self.agent.console.set_slash_commands(self.command_names())

    def command_names(self) -> List[str]:
        names: List[str] = []
        for entry in self.commands:
            names.extend(entry.keys())
        # Include bare aliases for completion
        names.extend(["exit", "quit"])
        # stable unique order
        seen: set[str] = set()
        ordered: List[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        return ordered

    def command_help_rows(self) -> List[tuple[str, str]]:
        rows: List[tuple[str, str]] = []
        for entry in self.commands:
            for name, fn in entry.items():
                doc = (inspect.getdoc(fn) or "").strip().split("\n")[0]
                rows.append((name, doc or name))
        rows.append(("exit / quit", "End the session"))
        return rows

    def router(self, command: str) -> CommandResult:
        raw = (command or "").strip()
        if not raw:
            return None

        key = raw.lower()
        key = self.ALIASES.get(key, key)

        if not key.startswith("/"):
            return None

        for entry in self.commands:
            handler = entry.get(key)
            if handler is not None:
                return handler()
        return None

    def get_methods(self) -> List[Dict[str, Callable[..., Any]]]:
        """Auto-register public methods as /<name> commands."""
        methods: List[Dict[str, Callable[..., Any]]] = []
        skip = {"router", "command_names", "command_help_rows", "get_methods"}

        for attr_name in dir(self):
            if attr_name.startswith("_") or attr_name in skip:
                continue
            if attr_name.startswith("get"):
                continue

            attr = getattr(self, attr_name)
            if inspect.ismethod(attr):
                methods.append({f"/{attr_name}": attr})

        return methods

    def help(self) -> bool:
        """Show this help"""
        self.agent.console.print_help(self.command_help_rows())
        return True

    def clear(self) -> bool:
        """Clear the screen"""
        agent = self.agent
        agent.console.clear_screen()
        agent.console.print_welcome(
            agent.console._session_title,
            agent.console._session_workspace,
            is_auth=agent.config.api_key,
            provider=agent.config.provider,
            model=str(agent.config.model),
        )
        return True

    def new(self) -> bool:
        """Start a new conversation session without quitting"""
        agent = self.agent
        console = agent.console
        had_session = agent.memory.session is not None
        prev_title = (
            agent.memory.session.title if agent.memory.session else None
        )

        agent.reset_conversation()
        console._session_title = " "
        console._session_workspace = ""
        console._last_assistant_message = ""
        console.clear_screen()
        console.print_welcome(
            is_auth=agent.config.api_key,
            provider=agent.config.provider,
            model=str(agent.config.model),
        )
        if had_session:
            note = f"Previous session left as-is"
            if prev_title:
                note += f" ({prev_title})"
            note += ".\nNew session ready — send a task to begin."
        else:
            note = "New session ready — send a task to begin."
        console.print_system_message(note, title="New")
        return True

    def quiet(self) -> bool:
        """Collapse tool output to one-liners"""
        console = self.agent.console
        console.quiet = True
        console.verbose = False
        console.print_system_message("Tool output collapsed (quiet mode).", title="Mode")
        return True

    def verbose(self) -> bool:
        """Show full tool output"""
        console = self.agent.console
        console.quiet = False
        console.verbose = True
        console.print_system_message("Full tool output enabled (verbose mode).", title="Mode")
        return True

    def copy(self) -> bool:
        """Copy last assistant reply to clipboard"""
        console = self.agent.console
        if not console._last_assistant_message:
            console.print_error("No assistant message to copy yet.", title="Copy")
        elif console._copy_to_clipboard(console._last_assistant_message):
            console.print_system_message(
                "Last assistant reply copied to clipboard.", title="Copy"
            )
        else:
            console.print_error("Could not access the system clipboard.", title="Copy")
        return True

    def resume(self) -> bool:
        """Resume a previous session"""
        agent = self.agent
        old_sessions = agent.memory.load_old_sessions()
        if not old_sessions:
            agent.console.print_system_message("No previous sessions found.", "Resume")
            return True
        selected_session = agent.console.interactive_select(old_sessions)
        agent.console.clear_screen()
        old_chats = agent.memory.load_session_chat(
            selected_session.history_path,
            system_prompt=agent.prompt.raw_system_prompt,
        )
        agent.memory.session = selected_session
        agent.memory.messages = old_chats

        # Immediately calculate and sync context state for the resumed session
        comp = agent._compaction()
        working_tokens = comp.working_token_count(old_chats, selected_session)
        cw = agent.get_model_context_window()
        agent.console.set_context_state(working_tokens, cw)

        agent.console.print_welcome(
            selected_session.title,
            str(selected_session.workspace),
            is_auth=agent.config.api_key,
            provider=agent.config.provider,
            model=str(agent.config.model),
        )
        agent.console.print_chat_history(old_chats)
        agent.console.print_system_message(f"Resumed session: {selected_session.title}")
        return True

    def skills(self) -> bool:
        """Manually pick skills for this session (skips auto skill selection)"""
        from skills import Skills

        agent = self.agent
        console = agent.console
        Skills.refresh()
        names = Skills.names()
        if not names:
            console.print_system_message(
                "No skills found.\n"
                "Looked in:\n  "
                + "\n  ".join(str(p) for p in Skills.search_dirs()),
                title="Skills",
            )
            return True

        if agent.manual_skill_names is None:
            mode_note = "Mode: auto (agent picks each turn)"
            preselected: list[str] = []
        else:
            mode_note = f"Mode: manual ({len(agent.manual_skill_names)} selected)"
            preselected = list(agent.manual_skill_names)

        console.print_system_message(
            f"{mode_note}\n"
            f"Available: {len(names)} skill(s) in this workspace.",
            title="Skills",
        )

        result = console.interactive_multi_pick(
            names,
            title="Select skills (space to toggle)",
            preselected=preselected,
        )
        if result is None:
            console.print_system_message("Skills selection cancelled.", title="Skills")
            return True

        mode, selected = result
        if mode == "auto":
            agent.manual_skill_names = None
            # Reset system prompt to base; auto picker will inject next turn
            agent.apply_active_skills([], announce=False)
            console.print_system_message(
                "Skill mode: auto — agent will pick skills each turn.",
                title="Skills",
            )
            return True

        agent.manual_skill_names = selected
        agent.apply_active_skills(selected, announce=False)
        if selected:
            console.print_system_message(
                "Skill mode: manual — auto selection disabled.\n"
                f"Active: {', '.join(selected)}",
                title="Skills",
            )
        else:
            console.print_system_message(
                "Skill mode: manual — no skills selected.\n"
                "Auto selection is disabled until you press 'a' in /skills.",
                title="Skills",
            )
        return True

    def login(self) -> bool:
        """Set Primary/Secondary API key (supports rate-limit failover)"""
        from config import get_provider_settings, set_provider_key

        agent = self.agent
        console = agent.console
        provider = agent.config.provider
        settings = get_provider_settings(provider)
        key_count = int(settings.get("key_count", 0) or 0)
        active_idx = int(settings.get("active_key_index", 0) or 0)

        slot_labels = [
            f"Primary{'  (set)' if key_count >= 1 else '  (empty)'}"
            f"{'  [active]' if active_idx == 0 and key_count >= 1 else ''}",
            f"Secondary{'  (set)' if key_count >= 2 else '  (empty)'}"
            f"{'  [active]' if active_idx == 1 and key_count >= 2 else ''}",
        ]
        picked = console.interactive_pick(
            slot_labels,
            title=f"Select API key slot for '{provider}'",
            current=slot_labels[active_idx] if key_count else None,
        )
        if not picked:
            console.print_system_message("Login cancelled.", title="Auth")
            return True
        slot = 0 if picked.startswith("Primary") else 1
        slot_name = "Primary" if slot == 0 else "Secondary"

        api_key = console.get_api_key(f"{provider} ({slot_name})")
        if not api_key:
            return True

        try:
            set_provider_key(provider, slot, api_key, make_active_slot=True)
            agent.apply_provider_runtime()
            updated = get_provider_settings(provider)
            console.print_system_message(
                f"{slot_name} API key saved for '{provider}'.\n"
                f"Keys configured: {updated.get('key_count', 0)}/2\n"
                f"Active slot: {'Primary' if updated.get('active_key_index', 0) == 0 else 'Secondary'}",
                title="Auth Success",
            )
        except Exception as e:
            console.print_error(f"Failed to save credentials: {e}", title="Auth Error")
        return True

    def tavily(self) -> bool:
        """Set Tavily API key for web_search"""
        from config import set_tavily_api_key, get_tavily_api_key

        agent = self.agent
        console = agent.console
        existing = get_tavily_api_key()
        if existing:
            console.print_system_message(
                "A Tavily API key is already configured (will be overwritten if you continue).",
                title="Tavily",
            )

        api_key = console.get_api_key("Tavily")
        if not api_key:
            return True

        try:
            set_tavily_api_key(api_key)
            agent.config.tavily_api_key = api_key
            console.print_system_message(
                "Tavily API key saved to auth.json.",
                title="Tavily",
            )
        except Exception as e:
            console.print_error(f"Failed to save Tavily API key: {e}", title="Tavily")
        return True

    def provider(self) -> bool:
        """Change LLM provider (mistral / groq / custom)"""
        from config import (
            BUILTIN_PROVIDERS,
            list_provider_names,
            set_active_provider,
            upsert_provider_settings,
            get_provider_settings,
        )

        agent = self.agent
        console = agent.console
        ADD_CUSTOM = "+ Add custom provider"

        names = list_provider_names()
        labels = []
        for name in names:
            settings = get_provider_settings(name)
            kind = "custom" if settings.get("is_custom") else "built-in"
            url = settings.get("base_url") or ""
            short_url = url if len(url) <= 40 else url[:37] + "..."
            labels.append(f"{name}  [{kind}]  {short_url}")
        labels.append(ADD_CUSTOM)

        # Map label -> name for selection
        label_to_name = {labels[i]: names[i] for i in range(len(names))}
        label_to_name[ADD_CUSTOM] = ADD_CUSTOM

        current_label = None
        for label, name in label_to_name.items():
            if name == agent.config.provider:
                current_label = label
                break

        picked = console.interactive_pick(
            labels,
            title=f"Select provider (active: {agent.config.provider})",
            current=current_label,
        )
        if not picked:
            console.print_system_message("Provider selection cancelled.", title="Provider")
            return True

        if picked == ADD_CUSTOM or label_to_name.get(picked) == ADD_CUSTOM:
            name = console.prompt_text("Custom provider name (e.g. ollama)")
            if not name:
                console.print_system_message("Cancelled.", title="Provider")
                return True
            name = name.strip().lower().replace(" ", "-")
            if name in BUILTIN_PROVIDERS:
                console.print_error(
                    f"'{name}' is a built-in provider name. Choose another.",
                    title="Provider",
                )
                return True
            base_url = console.prompt_text(
                "OpenAI-compatible base URL",
                default="http://localhost:11434/v1",
            )
            if not base_url:
                console.print_system_message("Cancelled.", title="Provider")
                return True
            api_key = console.get_api_key(name) or "no-key"
            upsert_provider_settings(
                name,
                api_key=api_key,
                base_url=base_url.rstrip("/"),
                is_custom=True,
                make_active=True,
            )
            agent.apply_provider_runtime()
            console.print_system_message(
                f"Custom provider '{name}' saved and activated.\n"
                f"Endpoint: {base_url}\n"
                f"Tip: run /model to pick a model.",
                title="Provider",
            )
            return True

        provider_name = label_to_name[picked]
        set_active_provider(provider_name)
        agent.apply_provider_runtime()

        settings = get_provider_settings(provider_name)
        msg = (
            f"Active provider: {provider_name}\n"
            f"Model: {settings.get('model') or '(none)'}\n"
            f"Endpoint: {settings.get('base_url') or '(none)'}"
        )
        if not settings.get("api_key"):
            msg += "\nNo API key yet — run /login to authenticate."
        console.print_system_message(msg, title="Provider")
        return True

    def model(self) -> bool:
        """Change model for the active provider (search or type a custom name)"""
        from config import upsert_provider_settings, get_provider_settings

        agent = self.agent
        console = agent.console
        provider = agent.config.provider
        CUSTOM = "+ Enter model name manually..."

        if not agent.config.api_key:
            console.print_error(
                "No API key for the active provider. Run /login first.",
                title="Model",
            )
            return True
        if not agent.config.base_url:
            console.print_error(
                "No base URL configured for this provider.",
                title="Model",
            )
            return True

        console.print_system_message(
            f"Fetching models from {provider} ({agent.config.base_url})...",
            title="Model",
        )
        models_info: list[dict[str, Any]] = []
        try:
            models_info = agent.list_available_models_info()
        except Exception as e:
            console.print_error(
                f"Failed to list models: {e}\nYou can still enter a model name manually.",
                title="Model",
            )

        current = str(agent.config.model)
        if not models_info:
            console.print_system_message(
                "No models listed by the provider — enter a model id manually.",
                title="Model",
            )
            picked = console.prompt_text("Model name", default=current or "")
            if not picked:
                console.print_system_message("Model selection cancelled.", title="Model")
                return True
        else:
            options = []
            option_to_id = {}
            current_option = None
            for info in models_info:
                mid = info["id"]
                label = f"{mid}  [{info['formatted_context']} context]"
                options.append(label)
                option_to_id[label] = mid
                if mid == current:
                    current_option = label

            picked_option = console.interactive_pick(
                options,
                title=f"Select model ({provider}) — {len(options)} available",
                current=current_option,
                searchable=True,
                custom_option=CUSTOM,
            )
            if not picked_option:
                console.print_system_message("Model selection cancelled.", title="Model")
                return True
            if picked_option == CUSTOM:
                picked = console.prompt_text("Model name", default=current or "")
                if not picked:
                    console.print_system_message("Model selection cancelled.", title="Model")
                    return True
            else:
                picked = option_to_id.get(picked_option, picked_option.split(" ")[0])

        picked = picked.strip()
        upsert_provider_settings(provider, model=picked, make_active=True)
        agent.config.model = picked
        cw = agent.get_model_context_window(picked)
        from llm import format_tokens

        console.print_system_message(
            f"Model set to '{picked}' for provider '{provider}' ({format_tokens(cw)} context window).",
            title="Model",
        )
        return True

    def history(self) -> bool:
        """Set max messages kept in the LLM context window"""
        from config import update_app_settings

        agent = self.agent
        console = agent.console
        current = agent.config.max_history_messages
        console.print_system_message(
            f"Current MAX_HISTORY_MESSAGES: {current}\n"
            "This caps how many recent messages are sent to the model each turn.",
            title="History",
        )
        raw = console.prompt_text(
            "Max history messages (>= 1)",
            default=str(current),
        )
        if raw is None:
            console.print_system_message("Cancelled.", title="History")
            return True
        try:
            value = int(raw.strip())
            settings = update_app_settings(max_history_messages=value)
        except ValueError as e:
            console.print_error(f"Invalid value: {e}", title="History")
            return True

        agent.config.apply_app_settings(settings)
        console.print_system_message(
            f"Max history messages set to {agent.config.max_history_messages}.\n"
            "Saved to auth.json (app_settings).",
            title="History",
        )
        return True

    def max_tokens(self) -> bool:
        """Set max generation tokens for LLM requests (or clear it)"""
        from config import update_app_settings

        agent = self.agent
        console = agent.console
        current = agent.config.max_tokens
        current_str = str(current) if current is not None else "None (provider default)"

        console.print_system_message(
            f"Current MAX_TOKENS limit: {current_str}\n"
            "This limits the maximum completion tokens requested from the model.",
            title="Max Tokens",
        )
        raw = console.prompt_text(
            "Enter max tokens value (>= 1) or 'none'/'clear' to remove limit",
            default="" if current is None else str(current),
        )
        if raw is None:
            console.print_system_message("Cancelled.", title="Max Tokens")
            return True

        cleaned = raw.strip().lower()
        if cleaned in ("none", "clear", ""):
            settings = update_app_settings(clear_max_tokens=True)
            agent.config.apply_app_settings(settings)
            console.print_system_message(
                "Max tokens limit removed. Provider defaults will be used.\n"
                "Saved to auth.json (app_settings).",
                title="Max Tokens",
            )
            return True

        try:
            value = int(cleaned)
            settings = update_app_settings(max_tokens=value)
        except ValueError as e:
            console.print_error(f"Invalid value: {e}", title="Max Tokens")
            return True

        agent.config.apply_app_settings(settings)
        console.print_system_message(
            f"Max tokens set to {agent.config.max_tokens}.\n"
            "Saved to auth.json (app_settings).",
            title="Max Tokens",
        )
        return True

    def compact(self) -> bool:
        """Run compaction now (manual), even if under the auto threshold"""
        from llm import format_tokens

        agent = self.agent
        console = agent.console
        if not agent.config.api_key:
            console.print_error(
                "No API key configured. Run /login first.",
                title="Compact",
            )
            return True
        if not agent.memory.session:
            console.print_system_message(
                "No active session yet. Send a task (or /resume) first.",
                title="Compact",
            )
            return True

        session = agent.memory.session
        cw = agent.get_model_context_window()
        cw_fmt = format_tokens(cw)
        keep_recent = agent.config.keep_recent_tokens
        keep_pct = round((keep_recent / cw) * 100, 1) if cw > 0 else 15.0
        console.print_system_message(
            f"Manual compaction\n"
            f"Keep recent tokens: {keep_recent:,} (~{keep_pct}% of {cw_fmt})\n"
            f"Already compacted until index: {session.compacted_until}\n"
            f"Has prior summary: {'yes' if session.compaction_summary else 'no'}",
            title="Compact",
        )
        try:
            status = agent.run_compaction(force=True)
        except Exception as e:
            console.print_error(str(e), title="Compact")
            return True

        console.print_system_message(status, title="Compact")
        if session.compaction_summary:
            preview = session.compaction_summary.strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:237] + "..."
            console.print_system_message(
                f"Summary preview:\n{preview}",
                title="Compact",
            )
        return True

    def compaction(self) -> bool:
        """Configure compaction (enabled, trigger threshold, keep recent tokens)"""
        from config import update_app_settings
        from llm import format_tokens

        agent = self.agent
        console = agent.console
        enabled = agent.config.compaction_enabled
        at_tokens = agent.config.compact_at_tokens
        keep_recent_tokens = agent.config.keep_recent_tokens

        cw = agent.get_model_context_window()
        cw_fmt = format_tokens(cw)
        at_pct = round((at_tokens / cw) * 100, 1) if cw > 0 else 75.0
        keep_pct = round((keep_recent_tokens / cw) * 100, 1) if cw > 0 else 15.0

        RUN_LABEL = "Run compaction now"
        ENABLED_LABEL = f"Enabled  {'on' if enabled else 'off'}"
        TOKENS_LABEL = f"Compact trigger threshold  {at_tokens:,} tokens ({at_pct}% of {cw_fmt} context)"
        KEEP_TOKENS_LABEL = f"Keep recent tokens  {keep_recent_tokens:,} tokens ({keep_pct}% of {cw_fmt} context)"

        console.print_system_message(
            f"Compaction summarizes older turns when working context grows large.\n"
            f"Active Model Context Window: {cw_fmt} ({cw:,} tokens)\n"
            f"Enabled: {enabled}\n"
            f"Trigger Threshold: {at_tokens:,} tokens (~{at_pct}% of context)\n"
            f"Keep Recent Tokens: {keep_recent_tokens:,} tokens (~{keep_pct}% of context)\n"
            "Full history stays on disk; older context is summarized dynamically.\n"
            "Tip: /compact forces a compaction run immediately.",
            title="Compaction",
        )

        picked = console.interactive_pick(
            [RUN_LABEL, ENABLED_LABEL, TOKENS_LABEL, KEEP_TOKENS_LABEL],
            title="Compaction",
        )
        if not picked:
            console.print_system_message("Cancelled.", title="Compaction")
            return True

        if picked == RUN_LABEL:
            return self.compact()

        kwargs: Dict[str, Any] = {}
        try:
            if picked == ENABLED_LABEL:
                choice = console.interactive_pick(
                    ["on", "off"],
                    title="Compaction enabled",
                    current="on" if enabled else "off",
                )
                if not choice:
                    console.print_system_message("Cancelled.", title="Compaction")
                    return True
                kwargs["compaction_enabled"] = choice == "on"

            elif picked == TOKENS_LABEL:
                MODE_PCT = f"Set by Percentage (%)  [Active model: {cw_fmt} context]"
                MODE_EXACT = "Set by Exact Token Count"
                mode = console.interactive_pick(
                    [MODE_PCT, MODE_EXACT],
                    title=f"Set Trigger Threshold (Context: {cw_fmt})",
                )
                if not mode:
                    console.print_system_message("Cancelled.", title="Compaction")
                    return True

                if mode == MODE_PCT:
                    raw_pct = console.prompt_text(
                        f"Enter compaction trigger threshold percentage (1% - 95% of {cw_fmt})",
                        default=str(int(at_pct)),
                    )
                    if raw_pct is None:
                        console.print_system_message("Cancelled.", title="Compaction")
                        return True
                    clean_pct = float(raw_pct.strip().replace("%", ""))
                    if not (1 <= clean_pct <= 95):
                        raise ValueError("Percentage must be between 1% and 95%.")
                    target_tokens = int(cw * (clean_pct / 100.0))
                    target_tokens = max(1000, target_tokens)
                    kwargs["compact_at_tokens"] = target_tokens
                    console.print_system_message(
                        f"Calculated trigger threshold: {clean_pct}% of {cw_fmt} = {target_tokens:,} tokens.",
                        title="Threshold Set",
                    )
                else:
                    raw = console.prompt_text(
                        "Compact when working context reaches (tokens, >= 1000)",
                        default=str(at_tokens),
                    )
                    if raw is None:
                        console.print_system_message("Cancelled.", title="Compaction")
                        return True
                    kwargs["compact_at_tokens"] = int(raw.strip().replace(",", ""))

            elif picked == KEEP_TOKENS_LABEL:
                MODE_PCT = f"Set by Percentage (%)  [Active model: {cw_fmt} context]"
                MODE_EXACT = "Set by Exact Token Count"
                mode = console.interactive_pick(
                    [MODE_PCT, MODE_EXACT],
                    title=f"Set Recent Tokens to Keep (Context: {cw_fmt})",
                )
                if not mode:
                    console.print_system_message("Cancelled.", title="Compaction")
                    return True

                if mode == MODE_PCT:
                    raw_pct = console.prompt_text(
                        f"Enter recent tokens to keep percentage (1% - 90% of {cw_fmt})",
                        default=str(int(keep_pct)),
                    )
                    if raw_pct is None:
                        console.print_system_message("Cancelled.", title="Compaction")
                        return True
                    clean_pct = float(raw_pct.strip().replace("%", ""))
                    if not (1 <= clean_pct <= 90):
                        raise ValueError("Percentage must be between 1% and 90%.")
                    target_tokens = int(cw * (clean_pct / 100.0))
                    target_tokens = max(1000, target_tokens)
                    kwargs["keep_recent_tokens"] = target_tokens
                    console.print_system_message(
                        f"Calculated keep recent tokens: {clean_pct}% of {cw_fmt} = {target_tokens:,} tokens.",
                        title="Keep Tokens Set",
                    )
                else:
                    raw = console.prompt_text(
                        "Keep this many recent tokens raw (>= 1000)",
                        default=str(keep_recent_tokens),
                    )
                    if raw is None:
                        console.print_system_message("Cancelled.", title="Compaction")
                        return True
                    kwargs["keep_recent_tokens"] = int(raw.strip().replace(",", ""))

            settings = update_app_settings(**kwargs)
        except ValueError as e:
            console.print_error(f"Invalid value: {e}", title="Compaction")
            return True

        agent.config.apply_app_settings(settings)
        new_at_tokens = agent.config.compact_at_tokens
        new_keep_tokens = agent.config.keep_recent_tokens
        new_at_pct = round((new_at_tokens / cw) * 100, 1) if cw > 0 else 75.0
        new_keep_pct = round((new_keep_tokens / cw) * 100, 1) if cw > 0 else 15.0

        console.print_system_message(
            f"Enabled: {agent.config.compaction_enabled}\n"
            f"Trigger Threshold: {new_at_tokens:,} tokens (~{new_at_pct}% of {cw_fmt} context)\n"
            f"Keep Recent Tokens: {new_keep_tokens:,} tokens (~{new_keep_pct}% of {cw_fmt} context)\n"
            "Saved to auth.json (app_settings).",
            title="Compaction",
        )
        return True

    def prices(self) -> bool:
        """Set estimated input/output USD price per million tokens"""
        from config import update_app_settings

        agent = self.agent
        console = agent.console
        inp = agent.config.input_price_per_mtok
        out = agent.config.output_price_per_mtok

        INPUT_LABEL = f"Input  ${inp:.4f} / MTok"
        OUTPUT_LABEL = f"Output ${out:.4f} / MTok"
        BOTH_LABEL = "Edit both"

        picked = console.interactive_pick(
            [INPUT_LABEL, OUTPUT_LABEL, BOTH_LABEL],
            title="Cost estimate rates ($ per million tokens)",
        )
        if not picked:
            console.print_system_message("Cancelled.", title="Prices")
            return True

        kwargs: Dict[str, float] = {}
        try:
            if picked in (INPUT_LABEL, BOTH_LABEL):
                raw_in = console.prompt_text(
                    "Input price per MTok (USD)",
                    default=f"{inp:.4f}",
                )
                if raw_in is None:
                    console.print_system_message("Cancelled.", title="Prices")
                    return True
                kwargs["input_price_per_mtok"] = float(raw_in.strip())

            if picked in (OUTPUT_LABEL, BOTH_LABEL):
                raw_out = console.prompt_text(
                    "Output price per MTok (USD)",
                    default=f"{out:.4f}",
                )
                if raw_out is None:
                    console.print_system_message("Cancelled.", title="Prices")
                    return True
                kwargs["output_price_per_mtok"] = float(raw_out.strip())

            settings = update_app_settings(**kwargs)
        except ValueError as e:
            console.print_error(f"Invalid value: {e}", title="Prices")
            return True

        agent.config.apply_app_settings(settings)
        console.print_system_message(
            f"Input:  ${agent.config.input_price_per_mtok:.4f} / MTok\n"
            f"Output: ${agent.config.output_price_per_mtok:.4f} / MTok\n"
            "Saved to auth.json (app_settings). Used for session cost estimates.",
            title="Prices",
        )
        return True

    def _history_path(self) -> Optional[Path]:
        session = self.agent.memory.session
        if not session:
            return None
        path = Path(session.history_path)
        if path.is_file():
            return path
        candidate = path.parent / "conversation_history.jsonl"
        if candidate.is_file():
            return candidate
        return path if path.exists() else None

    def _load_history_messages(self) -> Tuple[List[Any], Optional[Path]]:
        path = self._history_path()
        if path is None:
            return [], None
        if not path.is_file():
            return [], path
        return self.agent.memory.read_from_jsonl(path), path

    def tokens(self) -> bool:
        """Count tokens in the current session conversation_history.jsonl"""
        agent = self.agent
        console = agent.console
        session = agent.memory.session

        if not session:
            console.print_error("No active session yet.", title="Tokens")
            return True

        messages, history_path = self._load_history_messages()
        if history_path is None:
            console.print_error(
                "Session has no conversation_history.jsonl path.",
                title="Tokens",
            )
            return True
        if not history_path.is_file():
            console.print_error(
                f"History file not found:\n{history_path}",
                title="Tokens",
            )
            return True
        if not messages:
            console.print_system_message(
                f"History is empty.\n{history_path}",
                title="Tokens",
            )
            return True

        try:
            counts, total, tokenizer_id = count_messages(
                messages,
                provider=agent.config.provider,
            )
        except Exception as e:
            console.print_error(str(e), title="Tokens")
            return True

        by_role = tokens_by_role(messages, counts)
        role_lines = "\n".join(
            f"  {role}: {stats['tokens']:,} tokens across {stats['messages']} message(s)"
            for role, stats in sorted(by_role.items())
        )
        api_line = (
            f"API-reported (session metadata): {session.total_tokens:,} "
            f"(in {session.prompt_tokens:,} / out {session.completion_tokens:,} / cached {session.cached_tokens:,})"
        )
        console.print_system_message(
            f"Session: {session.title}\n"
            f"History: {history_path}\n"
            f"Tokenizer: {tokenizer_id}\n"
            f"Messages: {len(messages)}\n"
            f"Local total: {total:,} tokens\n"
            f"By role:\n{role_lines}\n"
            f"{api_line}",
            title="Tokens",
        )
        return True

    def exit(self) -> str:
        """End the session"""
        self.agent.console.print_system_message("Goodbye!", "Exit")
        return "exit"
