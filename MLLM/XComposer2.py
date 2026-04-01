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
MODEL_NAME = "internlm/internlm-xcomposer2-vl-7b"

torch.set_grad_enabled(False)

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run InternLM-XComposer2 inference on all images for the given task."""
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image = arg.img_path + img
        ans = []
        for question in prompt_teps:
            text = "<ImageHere>" + question
            with torch.cuda.amp.autocast():
                response, _ = model.chat(
                    tokenizer, query=text, image=image, history=[], do_sample=False
                )
            ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
