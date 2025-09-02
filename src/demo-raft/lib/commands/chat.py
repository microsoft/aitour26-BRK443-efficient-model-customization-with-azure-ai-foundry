#!/usr/bin/env python3
"""Interactive chat TUI for RAFT

- Provides a `chat` click command that launches a Rich based TUI
- Wraps the existing `create_client` helper into a LangChain-compatible LLM
- Uses LangChain `ConversationBufferMemory` for simple memory and conversation management
- When creating a new conversation the user can pick the env prefix: BASELINE or FINETUNE
"""
import logging
from typing import Any, Optional
import rich_click as click
from rich.panel import Panel
from rich.table import Table

# Use shared helpers and console/logger configured elsewhere in the CLI
from lib.shared import setup_environment, console, logger

# Use the chat-model base and schema types from LangChain for correct integration
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    ChatResult,
    ChatGeneration,
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    # ToolMessage may not be present in some LangChain builds; handle by name at runtime
)
from langchain.memory import ConversationBufferMemory

from infra.tests.utils import create_client

logger = logging.getLogger("raft_cli.chat")


class RaftLangChainLLM(BaseChatModel):
    """LangChain ChatModel wrapper around the project's create_client helper.

    Implements `_generate` to conform with LangChain chat model integration
    (see LangChain docs in `langchain_llms.txt`).
    """

    env_prefix: str = "BASELINE"
    _client: Any = None
    _model: Optional[str] = None

    def __init__(self, **data):
        # Let pydantic / BaseChatModel populate declared fields (env_prefix etc.)
        super().__init__(**data)
        try:
            client, model = create_client(self.env_prefix)
            self._client = client
            self._model = model
            logger.debug("Initialized RaftLangChainLLM with env_prefix=%s model=%s", self.env_prefix, self._model)
        except Exception as e:
            logger.debug("Failed to initialize underlying client for env_prefix=%s: %s", self.env_prefix, e)

    def _messages_to_sdk(self, messages: list[BaseMessage]):
        sdk = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                # Fallback: detect ToolMessage by class name or other heuristics
                cls_name = getattr(m, "__class__", type(m)).__name__
                if cls_name == "ToolMessage":
                    role = "tool"
                else:
                    role = getattr(m, "type", "user") or "user"
            # message content can be a string or structured blocks; keep as-is
            content = getattr(m, "content", "")
            sdk.append({"role": role, "content": content})
        return sdk

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None, **kwargs) -> ChatResult:
        client = self._client
        model = self._model
        if client is None or model is None:
            # Attempt to (re)create client on-demand
            client, model = create_client(self.env_prefix)

        sdk_messages = self._messages_to_sdk(messages)
        response = client.chat.completions.create(model=model, messages=sdk_messages)
        if not response or not getattr(response, "choices", None):
            raise ValueError("No response from model")

        choice = response.choices[0]
        message_obj = getattr(choice, "message", None)
        response_text = message_obj.content if message_obj else str(choice)

        ai_message = AIMessage(content=response_text)
        gen = ChatGeneration(message=ai_message)
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "raft_chat"


class Conversation:
    def __init__(self, name: str, env_prefix: str = "BASELINE"):
        self.name = name
        self.env_prefix = env_prefix
        # LLM wrapper (builds client lazily during initialization)
        self.llm = RaftLangChainLLM(env_prefix=env_prefix)
        # LangChain memory (we will update it when sending messages)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Simple local history for composing chat messages to the SDK
        self.history = []  # list of {"role": "user"|"assistant", "content": str}

    def chat(self, text: str) -> str:
        # Append user message
        self.history.append({"role": "user", "content": text})

        # Ensure client is available
        client = getattr(self.llm, "_client", None)
        model = getattr(self.llm, "_model", None)
        if client is None or model is None:
            # Try to create client on-demand
            client, model = create_client(self.env_prefix)

        # Compose messages from history for the chat SDK
        messages = [{"role": m["role"], "content": m["content"]} for m in self.history]

        response = client.chat.completions.create(model=model, messages=messages)
        if not response or not getattr(response, "choices", None):
            raise ValueError("No response from model")

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        response_text = message.content if message else str(choice)

        # Append assistant message to history and update LangChain memory
        self.history.append({"role": "assistant", "content": response_text})
        try:
            # Save to LangChain memory in the expected shape
            self.memory.save_context({"input": text}, {"output": response_text})
        except Exception:
            # Memory implementations may differ between LangChain versions; ignore if not supported
            pass

        return response_text


class ConversationManager:
    def __init__(self):
        self._conversations = {}

    def create(self, name: str, env_prefix: str = "BASELINE") -> Conversation:
        if name in self._conversations:
            raise ValueError(f"Conversation '{name}' already exists")
        conv = Conversation(name=name, env_prefix=env_prefix)
        self._conversations[name] = conv
        return conv

    def list(self):
        return list(self._conversations.keys())

    def get(self, name: str) -> Conversation:
        return self._conversations.get(name)


@click.command()
@click.option("--start", is_flag=True, help="Start the chat UI immediately")
def chat(start: bool):
    """Start an interactive chat TUI backed by LangChain and the RAFT OpenAI/Azure clients.

    Conversations may be created to use either the BASELINE or FINETUNE env prefix
    (these correspond to the prefixes expected by the create_client helper).
    """
    # Load environment variables like other commands do
    setup_environment()

    manager = ConversationManager()

    console.print(Panel("[bold blue]RAFT Chat TUI[/bold blue]\nCreate or open conversations and chat interactively", expand=False))

    active = None

    while True:
        # If there's no active conversation show the management menu
        if active is None:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#")
            table.add_column("Conversation")
            for idx, name in enumerate(manager.list(), start=1):
                table.add_row(str(idx), name)
            console.print(table)

            console.print("[b]n[/b] New conversation    [b]o[/b] Open conversation    [b]q[/b] Quit")
            choice = console.input("Select> ").strip().lower()

            if choice in ("q", "quit", "exit"):
                console.print("Exiting chat")
                return

            if choice in ("n", "new"):
                name = console.input("Conversation name: ").strip()
                if not name:
                    console.print("[red]Conversation name is required[/red]")
                    continue
                env_prefix = click.prompt("Env prefix", type=click.Choice(["BASELINE", "FINETUNE"]), default="BASELINE")
                try:
                    active = manager.create(name, env_prefix)
                    console.print(f"Created conversation [bold]{name}[/bold] using env prefix [green]{env_prefix}[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to create conversation: {e}[/red]")
                    active = None
                    continue

            elif choice in ("o", "open"):
                names = manager.list()
                if not names:
                    console.print("[yellow]No conversations available - create one first[/yellow]")
                    continue
                sel = console.input("Enter conversation name or index: ").strip()
                selected = None
                if sel.isdigit():
                    idx = int(sel) - 1
                    if 0 <= idx < len(names):
                        selected = names[idx]
                else:
                    if sel in names:
                        selected = sel
                if not selected:
                    console.print("[red]Invalid selection[/red]")
                    continue
                active = manager.get(selected)
                console.print(f"Opened conversation [bold]{active.name}[/bold]")
            else:
                console.print("[red]Unknown command[/red]")
                continue

        # Active conversation chat loop
        console.rule(f"[bold cyan]Conversation: {active.name}[/bold cyan] ([green]{active.env_prefix}[/green])")
        console.print("Type your message and press Enter. Type 'back' to return to conversations, 'quit' to exit.")
        while True:
            try:
                user_input = console.input("[bold green]You[/bold green]> ").rstrip()
            except (KeyboardInterrupt, EOFError):
                console.print("\nExiting chat")
                return

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                console.print("Exiting chat")
                return
            if user_input.lower() in ("back", "return"):
                active = None
                break

            with console.status("Thinking..."):
                try:
                    resp = active.chat(user_input)
                except Exception as e:
                    console.print(f"[red]Error from model: {e}[/red]")
                    resp = None

            if resp is not None:
                console.print(Panel(resp, title="Assistant", expand=False))


# Expose the command for raft.py to import


# legacy click integration expects a symbol named `chat`


if __name__ == '__main__':
    chat()
