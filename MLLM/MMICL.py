# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

import torch
from PIL import Image

from model.instructblip import (
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_TYPE      = "instructblip"
MODEL_CKPT      = "BleachNick/MMICL-Instructblip-T5-xxl"
PROCESSOR_CKPT  = "Salesforce/instructblip-flan-t5-xxl"
DEVICE          = "cuda:0"
IMAGE_PLACEHOLDER = "图"
NUM_PLACEHOLDER_TOKENS = 32
REFERENCE_IMG_0 = "Temp_file/11578.jpg"
REFERENCE_IMG_1 = "Temp_file/944.jpg"

# --------------------------------------------------------------------------- #
#                             Helper Functions                                #
# --------------------------------------------------------------------------- #

def build_prompt(task, lab2, question, replace_token):
    """Construct the MMICL-style prompt string for a single question."""
    return (
        f"Use the image 0: <image0>{replace_token},"
        f"image 1: <image1>{replace_token} and "
        f"image 2: <image2>{replace_token} as a visual content audit tool, "
        f"it helps you accurately determine the category of your content. "
        f"image 0 is {task}.\nimage 1 is {lab2}.\n{question}\""
    )

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run MMICL inference on all images for the given task."""
    config = InstructBlipConfig.from_pretrained(MODEL_CKPT)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_CKPT, config=config
    ).to(DEVICE, dtype=torch.bfloat16)

    special_tokens = [IMAGE_PLACEHOLDER] + [f"<image{i}>" for i in range(20)]
    processor = InstructBlipProcessor.from_pretrained(PROCESSOR_CKPT)
    special_tokens = (
        special_tokens
        + processor.tokenizer.additional_special_tokens[len(special_tokens):]
    )
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(
        processor.qformer_tokenizer
    ):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))

    replace_token = IMAGE_PLACEHOLDER * NUM_PLACEHOLDER_TOKENS

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        ref_image_0 = Image.open(REFERENCE_IMG_0)
        ref_image_1 = Image.open(REFERENCE_IMG_1)
        target_image_path = arg.img_path + img
        images = [ref_image_0, ref_image_1, target_image_path]

        lab2 = prompt_teps[-1].split("C) ")[-1].split(".")[0]
        ans = []
        for question in prompt_teps:
            prompt = build_prompt(arg.task, lab2, question, replace_token)
            inputs = processor(images=images, text=prompt, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            inputs["img_mask"] = torch.tensor([[1 for _ in range(len(images))]])
            inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
            inputs = inputs.to(DEVICE)

            outputs = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                img_mask=inputs["img_mask"],
                do_sample=False,
                max_length=50,
                min_length=1,
                set_min_padding_size=False,
            )
            generated_text = processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()
            ans.append(generated_text)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
