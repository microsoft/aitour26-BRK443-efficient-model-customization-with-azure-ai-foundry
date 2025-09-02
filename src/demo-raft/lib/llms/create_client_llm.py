"""Small adapter class that wraps the test helper `create_client` and exposes a simple chat interface.

This keeps model invocation in one place so chat code can call `llm.chat(messages, endpoint)` instead of
calling the low-level OpenAI/Azure client directly.
"""
from typing import List, Dict, Tuple
import logging

try:
    from infra.tests.utils import create_client
except Exception:
    create_client = None

logger = logging.getLogger("raft_cli")


class CreateClientLLM:
    """Adapter for calling the OpenAI/Azure client returned by create_client.

    This implementation lazily creates and caches a client/model pair for the
    requested endpoint on first use, so subsequent `chat` calls reuse the
    cached client without calling `create_client` repeatedly.

    chat(messages, endpoint) -> str
    """

    def __init__(self):
        # clients are stored by lowercased endpoint key: 'finetune' or 'baseline'
        self._clients = {}
        if create_client is None:
            logger.warning("CreateClientLLM: create_client helper not available; clients will be created lazily when possible")

    def _create_and_cache(self, pref_upper: str):
        """Create a client/model for the given uppercase prefix and cache it."""
        try:
            client, model = create_client(pref_upper)
            self._clients[pref_upper.lower()] = (client, model)
            logger.debug("CreateClientLLM: created client for prefix=%s model=%s", pref_upper, model)
            return client, model
        except Exception as e:
            logger.exception("CreateClientLLM: failed to create client for prefix=%s: %s", pref_upper, e)
            raise

    def chat(self, messages: List[Dict[str, str]], endpoint: str) -> str:
        if create_client is None:
            raise RuntimeError("create_client helper not available; cannot create client")

        env_prefix = endpoint.lower()
        # Lazily construct a client/model pair for this endpoint if not already cached
        if env_prefix not in self._clients:
            pref_upper = env_prefix.upper()
            client, model = self._create_and_cache(pref_upper)
        else:
            client, model = self._clients[env_prefix]

        logger.debug("CreateClientLLM.chat: calling model=%s for endpoint=%s", model, env_prefix)
        response = client.chat.completions.create(model=model, messages=messages)

        # try common response shapes
        assistant_content = None
        try:
            assistant_content = response.choices[0].message.content
        except Exception:
            try:
                assistant_content = response.choices[0].text
            except Exception:
                assistant_content = None

        if assistant_content is None:
            raise RuntimeError("No assistant content in response")

        return assistant_content
