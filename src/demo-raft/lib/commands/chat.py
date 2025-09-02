#!/usr/bin/env python3
"""
Chat command for RAFT CLI - interactive chat TUI.

Provides a small Rich-based TUI to chat with the configured finetune and
baseline endpoints. Chat history is persisted across restarts in
'.raft_chats.json' stored at the root of the demo-raft folder.

This module re-uses the create_client helper from infra/tests/utils.py.
It expects environment variables with prefixes FINETUNE_* or BASELINE_*.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import rich_click as click
from rich.table import Table
from rich.prompt import Prompt
import os

from lib.shared import console, setup_environment, logger

# Optional LangChain schema imports (not required, used for message types)
try:
    from langchain.schema import HumanMessage, AIMessage, SystemMessage  # type: ignore
except Exception:
    HumanMessage = AIMessage = SystemMessage = None  # type: ignore

# Reuse the test helper to create an OpenAI/Azure client
try:
    from infra.tests.utils import create_client  # type: ignore
except Exception:
    create_client = None  # type: ignore

# Persisted chats file (placed at the demo-raft root)
CHATS_FILE: Path = Path(__file__).resolve().parents[2] / ".raft_chats.json"

# Default assistant/system prompt used by interactive components
SYSTEM_PROMPT = (
    "The following is a conversation with an AI assistant. "
    "The assistant is helpful, clever, friendly and gives concise and accurate answers."
)


def load_chats() -> Dict[str, Any]:
    if not CHATS_FILE.exists():
        return {}
    try:
        with open(CHATS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_chats(chats: Dict[str, Any]) -> None:
    try:
        tmp = CHATS_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(chats, fh, indent=2, ensure_ascii=False)
        tmp.replace(CHATS_FILE)
    except Exception as e:
        console.print(f"[red]Failed to save chats: {e}[/red]")


def create_new_chat(name: str, initial_endpoint: str = "finetune") -> Dict[str, Any]:
    chats = load_chats()
    chat_id = uuid.uuid4().hex
    chat = {
        "id": chat_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "messages": [],
        "endpoint": (initial_endpoint or "finetune").lower(),
    }
    chats[chat_id] = chat
    save_chats(chats)
    logger.debug("Created chat id=%s name=%s endpoint=%s", chat_id, name, chat["endpoint"])
    return chat


def find_chat_by_name(chats: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for c in chats.values():
        if c.get("name") == name or c.get("id") == name:
            return c
    return None


def _safe_create_client_for(endpoint: str):
    if not create_client:
        raise click.ClickException(
            "create_client is not importable. Ensure infra/tests/utils.py is available on PYTHONPATH."
        )

    # Ensure environment variables are loaded
    setup_environment()

    ep = endpoint.lower()
    if ep == "finetune":
        pref = "FINETUNE"
    elif ep == "baseline":
        pref = "BASELINE"
    else:
        raise click.ClickException("Unsupported endpoint. Use 'finetune' or 'baseline'.")

    # Log which env keys exist for the chosen prefix (keys only, not values)
    env_keys = [k for k in os.environ.keys() if k.startswith(pref + "_")]
    logger.debug("_safe_create_client_for: trying prefix=%s for endpoint=%s; env_keys=%s", pref, endpoint, env_keys)

    try:
        client, model = create_client(pref)
        logger.debug("create_client returned client=%s model=%s for prefix=%s", type(client), model, pref)
        return (client, model)
    except Exception as e:
        logger.exception("create_client failed for prefix %s: %s", pref, e)
        raise click.ClickException(f"Failed to create client for endpoint '{endpoint}' using prefix '{pref}': {e}")


def _format_message(msg: Dict[str, Any]) -> str:
    role = msg.get("role")
    content = msg.get("content")
    endpoint = msg.get("endpoint")
    if role == "user":
        return f"[cyan]You[/cyan]: {content}"
    elif role == "assistant":
        tag = f"[magenta]{endpoint}[/magenta]" if endpoint else "[green]Assistant[/green]"
        return f"{tag}: {content}"
    elif role == "system":
        return f"[dim]System:[/dim] {content}"
    else:
        return f"[dim]{role}[/dim]: {content}"


def open_chat_tui(chat: Dict[str, Any]) -> None:
    current_endpoint = chat.get("endpoint", "finetune").lower()
    console.rule(f" Chat: [bold]{chat.get('name')}[/bold] • Current endpoint: [cyan]{current_endpoint}[/cyan] ")
    console.print("[dim]Tip: use /switch to change endpoint without leaving the chat. Type /help for more commands.[/dim]")

    while True:
        # Display only the last 12 messages to keep the UI compact
        console.print()
        msgs = chat.get("messages", [])
        for m in msgs[-12:]:
            console.print(_format_message(m))

        try:
            user_input = console.input("\n[bold cyan]You[/bold cyan]> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Interrupted - exiting chat and saving...[/dim]")
            chats = load_chats()
            chats[chat["id"]] = chat
            save_chats(chats)
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.strip().split()
            cmd = parts[0][1:].lower()
            if cmd in ("exit", "quit"):
                chats = load_chats()
                chats[chat["id"]] = chat
                save_chats(chats)
                console.print("Exiting chat.")
                break
            elif cmd == "switch":
                if len(parts) > 1:
                    new_ep = parts[1].lower()
                else:
                    new_ep = "baseline" if current_endpoint == "finetune" else "finetune"
                current_endpoint = new_ep
                chat["endpoint"] = current_endpoint
                logger.debug("Switched endpoint for chat %s to %s", chat.get("id"), current_endpoint)
                console.print(f"Switched endpoint to [bold]{current_endpoint}[/bold]")
                # persist the endpoint change immediately
                chats = load_chats()
                chats[chat["id"]] = chat
                save_chats(chats)
                continue
            elif cmd == "history":
                console.print("[bold]Full chat history:[/bold]")
                for m in msgs:
                    console.print(_format_message(m))
                continue
            elif cmd == "save":
                chats = load_chats()
                chats[chat["id"]] = chat
                save_chats(chats)
                console.print("Chat saved.")
                continue
            elif cmd == "help":
                console.print("Commands: /switch [finetune|baseline], /history, /save, /exit, /help")
                continue
            else:
                console.print(f"Unknown command '/{cmd}'. Type /help for commands.")
                continue

        # Normal user message flow
        user_msg = {"role": "user", "content": user_input, "ts": datetime.utcnow().isoformat() + "Z"}
        chat.setdefault("messages", []).append(user_msg)

        # Persist the user's message immediately
        chats = load_chats()
        chats[chat["id"]] = chat
        save_chats(chats)

        # Prepare messages to send to the model (send full conversation for context)
        msgs_for_api = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in chat.get("messages", [])]

        # Prepend the system prompt if not already present
        if not msgs_for_api or msgs_for_api[0].get("role") != "system":
            msgs_for_api.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            logger.debug("Prepended system prompt to messages for API (length now %d)", len(msgs_for_api))

        # Call the selected endpoint using the shared create_client helper
        try:
            client, model = _safe_create_client_for(current_endpoint)
            logger.debug("Sending %d messages to model %s on endpoint %s", len(msgs_for_api), model, current_endpoint)

            # The tests helper returns an OpenAI/AzureOpenAI client which exposes chat.completions.create
            response = client.chat.completions.create(model=model, messages=msgs_for_api)

            logger.debug("Received response object: %s", type(response))

            assistant_content = None
            try:
                assistant_content = response.choices[0].message.content
            except Exception:
                # Fallbacks for different response shapes
                try:
                    assistant_content = response.choices[0].text
                except Exception:
                    assistant_content = None

            if not assistant_content:
                logger.debug("No assistant content found in response: %s", response)
                console.print("[red]No reply from model (empty response).[/red]")
                continue

            # Log only a truncated preview to avoid spamming secrets
            preview = (assistant_content[:200] + '...') if len(assistant_content) > 200 else assistant_content
            logger.debug("Assistant reply preview: %s", preview)

            assistant_msg = {
                "role": "assistant",
                "content": assistant_content,
                "endpoint": current_endpoint,
                "ts": datetime.utcnow().isoformat() + "Z",
            }

            chat.setdefault("messages", []).append(assistant_msg)

            # Persist after receiving reply
            chats = load_chats()
            chats[chat["id"]] = chat
            save_chats(chats)

            console.print(f"\n[green]{assistant_content}[/green]\n")

        except click.ClickException as ce:
            logger.exception("ClickException while calling model: %s", ce)
            console.print(f"[red]{ce}[/red]")
        except Exception as e:
            logger.exception("Unexpected error while calling model: %s", e)
            console.print(f"[red]Error calling model: {e}[/red]")


# Click command group and subcommands
@click.command("chat")
def chat() -> None:
    """Start the interactive chat manager and TUI.

    All discussion creation, listing, selection and endpoint switching is
    handled inside the TUI so there is only a single `raft chat` entrypoint.
    """
    while True:
        chats = load_chats()
        console.rule("[bold]RAFT Chat Manager[/bold]")

        if not chats:
            console.print("No chats found. Create a new chat now.")
            name = console.input("Chat name: ")
            endpoint = Prompt.ask("Initial endpoint", choices=["finetune", "baseline"], default="finetune")
            chat_obj = create_new_chat(name, endpoint)
            setup_environment()
            open_chat_tui(chat_obj)
            continue

        # Display saved chats
        table = Table(title="Saved Chats")
        table.add_column("#", style="dim", width=3)
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Endpoint")
        table.add_column("Messages", justify="right")

        ordered = list(chats.items())
        for idx, (cid, c) in enumerate(ordered, start=1):
            table.add_row(str(idx), cid, c.get("name", ""), c.get("endpoint", ""), str(len(c.get("messages", []))))

        console.print(table)
        console.print("[bold]Actions:[/bold] [green]open <#>[/green], [green]new[/green], [green]delete <#>[/green], [green]quit[/green]")

        action = console.input("Action> ").strip()
        if not action:
            continue
        cmd = action.split()
        verb = cmd[0].lower()

        if verb in ("q", "quit", "exit"):
            break

        if verb in ("n", "new"):
            name = console.input("Chat name: ")
            endpoint = Prompt.ask("Initial endpoint", choices=["finetune", "baseline"], default="finetune")
            chat_obj = create_new_chat(name, endpoint)
            setup_environment()
            open_chat_tui(chat_obj)
            continue

        if verb == "open" or verb.isdigit():
            # support either 'open 2' or just '2'
            try:
                idx = int(cmd[1]) if verb == "open" and len(cmd) > 1 else int(verb)
            except Exception:
                console.print("[red]Invalid selection[/red]")
                continue
            if idx < 1 or idx > len(ordered):
                console.print("[red]Selection out of range[/red]")
                continue
            _, chat_obj = ordered[idx - 1]
            setup_environment()
            open_chat_tui(chat_obj)
            continue

        if verb == "delete":
            try:
                idx = int(cmd[1])
            except Exception:
                console.print("[red]Invalid index for delete[/red]")
                continue
            if idx < 1 or idx > len(ordered):
                console.print("[red]Selection out of range[/red]")
                continue
            cid, c = ordered[idx - 1]
            confirm = Prompt.ask(f"Delete chat '{c.get('name')}'?", choices=["y", "n"], default="n")
            if confirm == "y":
                chats = load_chats()
                chats.pop(cid, None)
                save_chats(chats)
                console.print("Chat deleted.")
            continue

        console.print("[red]Unknown action[/red]")
