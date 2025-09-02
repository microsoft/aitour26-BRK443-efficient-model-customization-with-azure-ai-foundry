"""
Interactive Chat Command (uses configurable env prefix + optional Azure AI Search retriever)

This command uses the repository helper `create_langchain_chat_model` with a
configurable env_prefix (default: BASELINE) to create a LangChain chat client.
Optionally, it can use `AzureAISearchRetriever` (from `langchain-community`) to
retrieve relevant documents from an Azure AI Search index and include them as
context for each user query.

Assumes langchain v0.3.27 and the project's `lib.utils.raft_llm.create_langchain_chat_model`.
"""

import os

import rich_click as click

from lib.shared import setup_environment, console, logger
from lib.utils.raft_llm import create_langchain_chat_model


@click.command()
@click.option("--env-prefix", default="BASELINE", help="Environment prefix used by create_langchain_chat_model (e.g. BASELINE, FINETUNE, STUDENT)")
@click.option("--use-search", is_flag=True, help="Enable Azure AI Search retriever to augment user queries with retrieved context")
@click.option("--search-index", default=None, help="Azure AI Search index name (overrides AZURE_AI_SEARCH_INDEX_NAME env var)")
@click.option("--search-top-k", default=3, type=int, help="Number of search results to retrieve and include as context")
@click.option("--deployment", "-d", default=None, help="Override the <PREFIX>_AZURE_OPENAI_DEPLOYMENT from the environment")
@click.option("--temperature", "-t", default=0.0, type=float, help="Sampling temperature (honored by the helper client if applicable)")
@click.option("--system-prompt", default="The following is a conversation with an AI assistant. The assistant is helpful, clever, friendly and gives concise and accurate answers.", help="System prompt for the chat")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
def chat(env_prefix: str, use_search: bool, search_index: str, search_top_k: int, deployment: str, temperature: float, system_prompt: str, verbose: bool):
    """
    Start an interactive chat using a LangChain model created by
    `create_langchain_chat_model(env_prefix)`.

    Optionally uses Azure AI Search retriever to fetch relevant documents and
    include them as context for each user query.
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

    # Prepare optional Azure AI Search retriever
    retriever = None
    index_name = None
    if use_search:
        try:
            from langchain_community.retrievers import AzureAISearchRetriever
        except Exception as e:
            raise click.ClickException(
                "langchain-community and Azure Search packages are required for --use-search.\n"
                "Install with: pip install langchain-community azure-search-documents azure-identity"
            ) from e

        index_name = search_index or os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
        if not index_name:
            raise click.ClickException(
                "Azure AI Search index name not provided. Set AZURE_AI_SEARCH_INDEX_NAME or pass --search-index.")

        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        ad_token = token_provider()

        # Create retriever - service name and api key are picked up from env vars
        retriever = AzureAISearchRetriever(
            content_key="content", 
            top_k=search_top_k, 
            index_name=index_name,
            api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
            #azure_ad_token=ad_token
        )
        logger.debug("Created AzureAISearchRetriever for index %s (top_k=%s)", index_name, search_top_k)
        logger.info("Using Azure AI Search retriever for index: %s", index_name)

    # Import message types from langchain (assume available)
    from langchain.schema import HumanMessage, SystemMessage, AIMessage

    # Start conversation state
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

            # If search is enabled, retrieve documents for the user query
            context_text = None
            if retriever:
                try:
                    logger.info("Retriever invoked for index %s with query: %s", index_name, user_input[:120])
                    # Try common retriever APIs
                    docs = retriever.invoke(user_input)

                    # Build a simple context string from retrieved documents
                    if docs:
                        pieces = []
                        # Indicate retrieval to the user and show short previews
                        total_docs = len(docs)
                        shown = min(total_docs, search_top_k)
                        console.print(f"🔎 Retrieved {total_docs} documents (showing top {shown})")
                        logger.info("Retriever returned %s documents for query '%s'", total_docs, user_input[:120])
                        for i, d in enumerate(docs[:search_top_k], start=1):
                            content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                            # Escape any existing closing DOCUMENT tags to avoid accidental termination
                            # Wrap each document in <DOCUMENT>...</DOCUMENT>
                            pieces.append(f"<DOCUMENT>{content}</DOCUMENT>")
                            metadata = getattr(d, "metadata", {}) or {}
                            src = metadata.get("source") or metadata.get("id") or metadata.get("doc_id") or metadata.get("url") or "unknown"
                            score = metadata.get("score") or getattr(d, "score", None)
                            preview = content.replace("\n", " ")[:240]
                            if score is not None:
                                console.print(f"  • [{i}] {src} (score={score}) — {preview}")
                            else:
                                console.print(f"  • [{i}] {src} — {preview}")

                        # Join wrapped documents using a single newline between documents
                        context_text = "\n".join(pieces)
                        console.print(f"🔎 Retrieved documents:\n{context_text}")
                    else:
                        console.print("🔎 No relevant documents found.")
                        logger.info("Retriever returned no documents for query '%s'", user_input[:120])
                except Exception as e:
                    logger.error("Search retrieval failed: %s", e)
                    console.print(f"🔎 Search retrieval failed: {e}")
                    context_text = None

            # Build message list to send to the model. Do not permanently inject
            # the retrieved context into `messages` history; include it only for
            # this turn so the model sees the context but history remains clean.
            call_messages = list(messages)
            if context_text:
                call_messages.append(SystemMessage(content=f"Relevant documents:\n{context_text}"))

            # Append the user message
            call_messages.append(HumanMessage(content=user_input))

            # Get model response
            assistant_content = None
            try:
                response = llm.invoke(call_messages)
                assistant_content = response.content
            except Exception as e:
                logger.error(f"Error generating assistant response: {e}")
                console.print_exception()
                break

            console.print(f"[bold green]Assistant:[/bold green] {assistant_content}\n")

            # Append assistant message to conversation history
            messages.append(AIMessage(content=assistant_content))

        except KeyboardInterrupt:
            console.print("\n👋 Exiting chat.")
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            console.print_exception()
            break
