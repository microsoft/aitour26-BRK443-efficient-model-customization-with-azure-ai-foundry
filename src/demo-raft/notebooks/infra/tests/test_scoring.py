from dotenv import load_dotenv
from dotenv_azd import load_azd_env
from os import getenv
from utils import do_test_openai_endpoint

load_dotenv(".env")
load_azd_env()

def test_judge():
    do_test_openai_endpoint("JUDGE")
