from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider
from os import getenv
import os
import logging

logger = logging.getLogger("raft_cli")


def create_client(env_prefix):
    logger.debug("create_client called with env_prefix=%s", env_prefix)

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

        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
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

        client = AzureOpenAI(
            api_version=version,
            azure_endpoint=endpoint,
            azure_ad_token_provider = get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            )
        )
        logger.debug("Created AzureOpenAI client for prefix %s", env_prefix)
    else:
        logger.error("No OpenAI or Azure OpenAI env vars found for prefix %s", env_prefix)
        raise Exception("Couldn't find either OpenAI or Azure OpenAI env vars")

    return (client, model)


def do_test_openai_endpoint(env_prefix):
    assert env_prefix is not None

    (client, model) = create_client(env_prefix)

    response = client.chat.completions.create(
        model=model, 
        messages=[{"role": "user", "content": "Hello"}]
        )
    assert response is not None
    assert len(choices := response.choices) > 0
    assert (choice := choices[0]) is not None
    assert (message := choice.message) is not None
    assert message.content is not None

def do_test_azure_openai_endpoint(endpoint, model, version):
    assert model is not None
    assert endpoint is not None
    assert version is not None

    print(f"endpoint: {endpoint}")
    print(f"model: {model}")

    # Authenticate using the default Azure credential chain
    azure_credential = DefaultAzureCredential()

    client = AzureOpenAI(
        api_version=version,
        azure_endpoint=endpoint,
        azure_ad_token_provider = get_bearer_token_provider(
            azure_credential, "https://cognitiveservices.azure.com/.default"
        )
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello"}]
        )
    assert response is not None
    assert len(choices := response.choices) > 0
    assert (choice := choices[0]) is not None
    assert (message := choice.message) is not None
    assert message.content is not None
