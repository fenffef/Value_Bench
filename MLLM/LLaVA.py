# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import os

from MLLM.llava.llava.mm_utils import get_model_name_from_path
from MLLM.llava.llava.eval.run_llava import eval_model

from get_Prompts import get

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MODEL_PATH      = "liuhaotian/llava-v1.5-7b"
TEMPERATURE     = 0
NUM_BEAMS       = 1
MAX_NEW_TOKENS  = 512

# --------------------------------------------------------------------------- #
#                             Helper Functions                                #
# --------------------------------------------------------------------------- #

def build_llava_args(model_path, question, image_path):
    """Construct the argument namespace expected by eval_model."""
    return type("Args", (), {
        "model_path":    model_path,
        "model_base":    None,
        "model_name":    get_model_name_from_path(model_path),
        "query":         question,
        "conv_mode":     None,
        "image_file":    image_path,
        "sep":           ",",
        "temperature":   TEMPERATURE,
        "top_p":         None,
        "num_beams":     NUM_BEAMS,
        "max_new_tokens": MAX_NEW_TOKENS,
    })()

# --------------------------------------------------------------------------- #
#                               evaluate                                      #
# --------------------------------------------------------------------------- #

def evaluate(arg):
    """Run LLaVA-v1.5 inference on all images for the given task."""
    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image = arg.img_path + img
        ans = []
        for question in prompt_teps:
            llava_args = build_llava_args(MODEL_PATH, question, image)
            response = eval_model(llava_args)
            ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output
