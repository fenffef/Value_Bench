# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_NAME  = "openbmb/MiniCPM-V"
TEMPERATURE = 0.7

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run MiniCPM-V inference on all images for the given task."""
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device="cuda", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image = Image.open(arg.img_path + img).convert("RGB")
        ans = []
        for question in prompt_teps:
            msgs = [{"role": "user", "content": question}]
            out, _, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=True,
                temperature=TEMPERATURE,
            )
            ans.append(out)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
