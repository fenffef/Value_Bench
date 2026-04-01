# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image
from transformers import TextStreamer

from MLLM.mPLUG.mPLUG_Owl2.mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from MLLM.mPLUG.mPLUG_Owl2.mplug_owl2.conversation import conv_templates
from MLLM.mPLUG.mPLUG_Owl2.mplug_owl2.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from MLLM.mPLUG.mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_PATH     = "MAGAer13/mplug-owl2-llama2-7b"
CONV_MODE      = "mplug_owl2"
TEMPERATURE    = 0.7
MAX_NEW_TOKENS = 512

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run mPLUG-Owl2 inference on all images for the given task."""
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name,
        load_8bit=False, load_4bit=False, device="cuda"
    )

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image_path = arg.img_path + img
        conv = conv_templates[CONV_MODE].copy()

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        ans = []
        for question in prompt_teps:
            inp = DEFAULT_IMAGE_TOKEN + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            stop_str = conv.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], tokenizer, input_ids
            )
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            response = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
