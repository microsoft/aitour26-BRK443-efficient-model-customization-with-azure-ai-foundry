"""
Interactive Chat Command (uses configurable env prefix via helper)

This command uses the repository helper `create_langchain_chat_model` with a
configurable env_prefix (default: BASELINE) to create a LangChain chat client.
The helper handles both OpenAI and Azure OpenAI paths based on env vars like
`<PREFIX>_AZURE_OPENAI_ENDPOINT` / `<PREFIX>_AZURE_OPENAI_DEPLOYMENT` etc.

Assumes langchain v0.3.27 and the project's `lib.utils.raft_llm.create_langchain_chat_model`.
"""

import os

import rich_click as click

from lib.shared import setup_environment, console, logger
from lib.utils.raft_llm import create_langchain_chat_model


@click.command()
@click.option("--env-prefix", default="BASELINE", help="Environment prefix used by create_langchain_chat_model (e.g. BASELINE, FINETUNE, STUDENT)")
@click.option("--deployment", "-d", default=None, help="Override the <PREFIX>_AZURE_OPENAI_DEPLOYMENT from the environment")
@click.option("--temperature", "-t", default=0.0, type=float, help="Sampling temperature (honored by the helper client if applicable)")
@click.option("--system-prompt", default="You are a helpful assistant.", help="System prompt for the chat")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
def chat(env_prefix: str, deployment: str, temperature: float, system_prompt: str, verbose: bool):
    """
    Start an interactive chat using a LangChain model created by
    `create_langchain_chat_model(env_prefix)`.

    The helper will read <PREFIX>_* environment variables and return a
    configured LangChain chat client and the model/deployment name.
    """
    if verbose:
        logger.setLevel("DEBUG")
        logger.debug("Verbose logging enabled")

    # Ensure environment files are loaded
    setup_environment()

    prefix = (env_prefix or "BASELINE").upper()

    # If the user provided a deployment override, set the appropriate env var so
    # the helper will pick it up for the selected prefix.
    if deployment:
        os.environ[f"{prefix}_AZURE_OPENAI_DEPLOYMENT"] = deployment

    # Create the LangChain chat model using the project's helper
    try:
        llm, model = create_langchain_chat_model(prefix)
    except Exception as e:
        raise click.ClickException(f"Failed to create chat model for prefix '{prefix}': {e}") from e

    console.print(f"🟢 Starting chat with deployment [bold]{model}[/bold] using prefix [bold]{prefix}[/bold]")

    # Import message types from langchain (assume available)
    from langchain.schema import HumanMessage, SystemMessage, AIMessage

    # Start conversation
    messages = [SystemMessage(content=system_prompt)]
    console.print("\nType messages and press Enter. Type '/exit' or Ctrl+C to quit.\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
            if not user_input:
                continue

            if user_input.strip().lower() in ("/exit", "exit", "quit"):
                console.print("👋 Exiting chat.")
                break

            # Append user message
            messages.append(HumanMessage(content=user_input))

            # Try the high-level predict_messages API if available
            assistant_content = None
            try:
                response = llm(messages)
                assistant_content = response.content
            except Exception as e:
                logger.error(f"Error generating assistant response: {e}")
                console.print_exception()
                break

            console.print(f"[bold green]Assistant:[/bold green] {assistant_content}\n")

            # Append assistant message
            messages.append(AIMessage(content=assistant_content))

        except KeyboardInterrupt:
            console.print("\n👋 Exiting chat.")
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            console.print_exception()
            break
