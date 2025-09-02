from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider
from os import getenv
import os
import logging
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

logger = logging.getLogger("raft_cli")


def create_langchain_chat_model(env_prefix):
    logger.debug("create_langchain_chat_model called with env_prefix=%s", env_prefix)

    # OpenAI API
    base_url = getenv(f"{env_prefix}_OPENAI_BASE_URL")

    # Azure OpenAI API
    endpoint = getenv(f"{env_prefix}_AZURE_OPENAI_ENDPOINT")

    logger.debug("env keys present for prefix %s: %s", env_prefix, [k for k in os.environ.keys() if k.startswith(env_prefix + "_")])

    if base_url:
        model = getenv(f"{env_prefix}_OPENAI_DEPLOYMENT")
        api_key = getenv(f"{env_prefix}_OPENAI_API_KEY")
        logger.debug("Detected OpenAI path for prefix %s: base_url=%s, model=%s, api_key_set=%s", env_prefix, base_url, model, bool(api_key))
        assert model
        assert api_key

        client = ChatOpenAI(
            base_url = base_url,
            api_key = api_key,
            model = model
            )
        logger.debug("Created OpenAI client for prefix %s", env_prefix)
    elif endpoint:
        model = getenv(f"{env_prefix}_AZURE_OPENAI_DEPLOYMENT")
        version = getenv(f"{env_prefix}_OPENAI_API_VERSION")
        logger.debug("Detected Azure path for prefix %s: endpoint=%s, model=%s, version=%s", env_prefix, endpoint, model, version)
        assert model
        assert version

        # Authenticate using the default Azure credential chain
        azure_credential = DefaultAzureCredential()
        logger.debug("Authenticated with DefaultAzureCredential for prefix %s", env_prefix)

        client = AzureChatOpenAI(
            api_version=version,
            azure_endpoint=endpoint,
            azure_ad_token_provider = get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            ),
            deployment_name=model
        )
        logger.debug("Created AzureChatOpenAI client for prefix %s", env_prefix)
    else:
        logger.error("No OpenAI or Azure OpenAI env vars found for prefix %s", env_prefix)
        raise Exception("Couldn't find either OpenAI or Azure OpenAI env vars")

    return (client, model)
