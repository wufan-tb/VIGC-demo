import openxlab
import os

ak = os.getenv("OPENXLAB_AK")
sk = os.getenv("OPENXLAB_SK")
openxlab.login(ak, sk, re_login=True)

from openxlab.model import download

LLM_ROOT = "/home/xlab-app-center/vicuna-7b"
VIGC_ROOT = "/home/xlab-app-center/vigc-models"

while not os.path.exists(LLM_ROOT):
    os.makedirs(LLM_ROOT, exist_ok=True)
while not os.path.exists(VIGC_ROOT):
    os.makedirs(VIGC_ROOT, exist_ok=True)

LOCAL_ROOT = {
    "hanxiao/vicuna-7b-v1_1": LLM_ROOT,
    "hanxiao/VIGC_demo": VIGC_ROOT
}

DOWNLOAD_INFO = {
    "hanxiao/vicuna-7b-v1_1": [
        "vicuna-7b-v1_1_readme", "vicuna-7b-v1_1_config", "generate_config", "model_part1",
        "model_part2", "index", "special_tokens_map", "tokenizer.model", "tokenizer_config"],
    "hanxiao/VIGC_demo": [
        "instruct-blip-finetuned", "minigpt4-finetuned", "instruct-blip-pretrained",
        "minigpt4-pretrained"]
}

for model_repo in DOWNLOAD_INFO:
    download(
        model_repo=model_repo,
        model_name=DOWNLOAD_INFO[model_repo],
        output=LOCAL_ROOT[model_repo],
        overwrite=False
    )
