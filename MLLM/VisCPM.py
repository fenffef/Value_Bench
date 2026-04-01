# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

from PIL import Image

from MLLM.viscpm.VisCPM import VisCPMChat

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_PATH = "/path/to/checkpoint"

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run VisCPM-Chat inference on all images for the given task."""
    viscpm_chat = VisCPMChat(MODEL_PATH, image_safety_checker=True)

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image_path = arg.img_path + img
        image = Image.open(image_path).convert("RGB")
        ans = []
        for question in prompt_teps:
            answer, _, _ = viscpm_chat.chat(image, question)
            ans.append(answer)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
