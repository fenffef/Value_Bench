# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from transformers import AutoModel, AutoTokenizer

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_NAME = "internlm/internlm-xcomposer-7b"

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run InternLM-XComposer inference on all images for the given task."""
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.tokenizer = tokenizer

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image = arg.img_path + img
        ans = []
        for question in prompt_teps:
            response = model.generate(question, image)
            ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
