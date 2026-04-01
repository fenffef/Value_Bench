# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
TOKENIZER_NAME = "lmsys/vicuna-7b-v1.5"
MODEL_NAME     = "THUDM/cogvlm-chat-hf"
DEVICE         = "cuda"
GEN_KWARGS     = {"max_length": 2048, "do_sample": False}

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run CogVLM inference on all images for the given task."""
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image_path = arg.img_path + img
        ans = []
        for question in prompt_teps:
            image = Image.open(image_path).convert("RGB")
            inputs = model.build_conversation_input_ids(
                tokenizer, query=question, history=[], images=[image]
            )
            inputs = {
                "input_ids":      inputs["input_ids"].unsqueeze(0).to(DEVICE),
                "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(DEVICE),
                "attention_mask": inputs["attention_mask"].unsqueeze(0).to(DEVICE),
                "images":         [[inputs["images"][0].to(DEVICE).to(torch.bfloat16)]],
            }
            with torch.no_grad():
                outputs = model.generate(**inputs, **GEN_KWARGS)
                outputs = outputs[:, inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(outputs[0])
            ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
