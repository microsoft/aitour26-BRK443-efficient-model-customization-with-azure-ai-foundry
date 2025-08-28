from dotenv import load_dotenv
from dotenv_azd import load_azd_env
from os import getenv
from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider
from openai import AzureOpenAI
from utils import create_client

load_dotenv(".env")
load_azd_env()

def test_embeddings():

    (client, model) = create_client("EMBEDDING")

    response = client.embeddings.create(input = ["Hello"], model=model)
    assert response is not None
    assert len(datas := response.data) > 0
    assert datas[0] is not None
    assert (embedding := datas[0].embedding) is not None
    assert embedding is not None
    assert len(embedding) >= 512
