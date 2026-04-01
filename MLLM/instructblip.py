# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_NAME     = "Salesforce/instructblip-vicuna-7b"
NUM_BEAMS      = 5
MAX_LENGTH     = 256
MIN_LENGTH     = 1
TOP_P          = 0.9
REP_PENALTY    = 1.5
LEN_PENALTY    = 1.0
TEMPERATURE    = 1

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run InstructBLIP inference on all images for the given task."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
    model = model.to(device)

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image_path = arg.img_path + img
        image = Image.open(image_path)
        ans = []
        for question in prompt_teps:
            inputs = processor(
                images=image, text=question, return_tensors="pt"
            ).to(device)
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=NUM_BEAMS,
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH,
                top_p=TOP_P,
                repetition_penalty=REP_PENALTY,
                length_penalty=LEN_PENALTY,
                temperature=TEMPERATURE,
            )
            generated_text = processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()
            ans.append(generated_text)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
