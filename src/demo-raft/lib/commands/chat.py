"""
Interactive Chat Command (uses BASELINE env prefix via helper)

This command uses the repository helper `create_langchain_chat_model` with the
`BASELINE` env_prefix to create a LangChain chat client. That helper handles
both OpenAI and Azure OpenAI paths based on env vars like
`BASELINE_AZURE_OPENAI_ENDPOINT` / `BASELINE_AZURE_OPENAI_DEPLOYMENT` etc.

Assumes langchain v0.3.27 and the project's `lib.utils.raft_llm.create_langchain_chat_model`.
"""

import os

import rich_click as click

from lib.shared import setup_environment, console, logger
from lib.utils.raft_llm import create_langchain_chat_model


@click.command()
@click.option("--deployment", "-d", default=None, help="Override the BASELINE_AZURE_OPENAI_DEPLOYMENT from the environment")
@click.option("--temperature", "-t", default=0.0, type=float, help="Sampling temperature (honored by the helper client if applicable)")
@click.option("--system-prompt", default="You are a helpful assistant.", help="System prompt for the chat")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
def chat(deployment: str, temperature: float, system_prompt: str, verbose: bool):
    """
    Start an interactive chat using a LangChain model created by
    `create_langchain_chat_model(env_prefix="BASELINE").`

    The helper will read BASELINE_* environment variables and return a
    configured LangChain chat client and the model/deployment name.
    """
    if verbose:
        logger.setLevel("DEBUG")
        logger.debug("Verbose logging enabled")

    # Ensure environment files are loaded
    setup_environment()

    # If the user provided a deployment override, set the BASELINE env var so
    # the helper will pick it up.
    if deployment:
        os.environ["BASELINE_AZURE_OPENAI_DEPLOYMENT"] = deployment

    # Create the LangChain chat model using the project's helper
    try:
        llm, model = create_langchain_chat_model("BASELINE")
    except Exception as e:
        raise click.ClickException(f"Failed to create baseline chat model: {e}") from e

    console.print(f"🟢 Starting chat with baseline deployment [bold]{model}[/bold]")

    # Start conversation
    try:
        # import message types from langchain for structured messages
        from langchain.schema import HumanMessage, SystemMessage, AIMessage
    except Exception:
        # If langchain isn't available, still allow the client to be called with raw strings
        HumanMessage = SystemMessage = AIMessage = None

    messages = [SystemMessage(content=system_prompt) if SystemMessage else {"role": "system", "content": system_prompt}]
    console.print("\nType messages and press Enter. Type '/exit' or Ctrl+C to quit.\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
            if not user_input:
                continue

            if user_input.strip().lower() in ("/exit", "exit", "quit"):
                console.print("👋 Exiting chat.")
                break

            # Build message depending on whether langchain message classes are available
            if HumanMessage:
                messages.append(HumanMessage(content=user_input))
            else:
                messages.append({"role": "user", "content": user_input})

            # Try the high-level predict_messages API if available
            assistant_content = None
            try:
                if hasattr(llm, "predict_messages"):
                    response = llm.predict_messages(messages)
                    assistant_content = getattr(response, "content", str(response))
                else:
                    # Fallback to calling the model directly
                    result = llm(messages)
                    if hasattr(result, "generations") and result.generations:
                        gen = result.generations[0][0]
                        if hasattr(gen, "text") and gen.text:
                            assistant_content = gen.text
                        elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                            assistant_content = gen.message.content
                    if assistant_content is None:
                        assistant_content = str(result)
            except Exception as e:
                logger.error(f"Error generating assistant response: {e}")
                console.print_exception()
                break

            console.print(f"[bold green]Assistant:[/bold green] {assistant_content}\n")

            if AIMessage:
                messages.append(AIMessage(content=assistant_content))
            else:
                messages.append({"role": "assistant", "content": assistant_content})

        except KeyboardInterrupt:
            console.print("\n👋 Exiting chat.")
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            console.print_exception()
            break
