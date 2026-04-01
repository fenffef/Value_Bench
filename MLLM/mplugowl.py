# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image

from MLLM.mPLUG.mPLUG_Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from MLLM.mPLUG.mPLUG_Owl.mplug_owl.processing_mplug_owl import (
    MplugOwlImageProcessor,
    MplugOwlProcessor,
)
from MLLM.mPLUG.mPLUG_Owl.mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_CHECKPOINT = "MAGAer13/mplug-owl-llama-7b"
GENERATE_KWARGS  = {"do_sample": True, "top_k": 5, "max_length": 512}

PROMPT_TEMPLATE = (
    "The following is a conversation between a curious human and AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "            Human: <image>\n"
    "            Human: {}.\n"
    "            AI: "
)

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run mPLUG-Owl inference on all images for the given task."""
    model = MplugOwlForConditionalGeneration.from_pretrained(
        MODEL_CHECKPOINT, torch_dtype=torch.bfloat16
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = MplugOwlTokenizer.from_pretrained(MODEL_CHECKPOINT)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image_path = arg.img_path + img
        images = [Image.open(image_path)]
        ans = []
        for question in prompt_teps:
            prompts = [PROMPT_TEMPLATE.format(question)]
            inputs = processor(text=prompts, images=images, return_tensors="pt")
            inputs = {
                key: val.bfloat16() if val.dtype == torch.float else val
                for key, val in inputs.items()
            }
            inputs = {key: val.to(model.device) for key, val in inputs.items()}
            with torch.no_grad():
                generated = model.generate(**inputs, **GENERATE_KWARGS)
            sentence = tokenizer.decode(generated.tolist()[0], skip_special_tokens=True)
            ans.append(sentence)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
