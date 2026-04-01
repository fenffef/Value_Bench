# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image

from MLLM.moellava.moellava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from MLLM.moellava.moellava.conversation import SeparatorStyle, conv_templates
from MLLM.moellava.moellava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from MLLM.moellava.moellava.model.builder import load_pretrained_model
from MLLM.moellava.moellava.utils import disable_torch_init

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_PATH   = "LanguageBind/MoE-llava-Phi2-2.7B-4e"
DEVICE       = "cuda"
CONV_MODE    = "phi"
LOAD_4BIT    = False
LOAD_8BIT    = False
TEMPERATURE  = 0.2
MAX_NEW_TOKENS = 1024

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run MoE-LLaVA inference on all images for the given task."""
    disable_torch_init()

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name, LOAD_8BIT, LOAD_4BIT, device=DEVICE
    )
    image_processor = processor["image"]
    conv_template = conv_templates[CONV_MODE].copy()

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image_tensor = image_processor.preprocess(
            Image.open(arg.img_path + img).convert("RGB"), return_tensors="pt"
        )["pixel_values"].to(model.device, dtype=torch.float16)

        ans = []
        for question in prompt_teps:
            conv = conv_template.copy()
            inp = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()
            stop_str = (
                conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            )
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], tokenizer, input_ids
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            response = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
