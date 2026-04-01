# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_NAME = "Salesforce/blip2-opt-2.7b"

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run BLIP-2 inference on all images for the given task."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    model.to(device)

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image_path = arg.img_path + img
        ans = []
        for question in prompt_teps:
            image = [Image.open(image_path)]
            inputs = processor(
                images=image, text=question, return_tensors="pt"
            ).to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            ans.append(generated_text)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
