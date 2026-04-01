# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_NAME  = "Qwen/Qwen-VL-Chat"
RANDOM_SEED = 1234

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run Qwen-VL-Chat inference on all images for the given task."""
    torch.manual_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="cuda", trust_remote_code=True
    ).eval()

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image = arg.img_path + img
        ans = []
        for question in prompt_teps:
            response, _ = model.chat(
                tokenizer,
                query=f"<img>{image}</img>" + question,
                history=None,
            )
            ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
