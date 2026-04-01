# --------------------------------------------------------------------------- #
#                               Imports                                       #
# --------------------------------------------------------------------------- #
import argparse
import importlib

from pre_check import check

# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
DATA_DIR = "Alig/"

TASK_TO_DATA = {
    "unethical":       "ELEMENT",
    "harmful(CN)":     "CHMEMES",
    "harmful":         "Harm-C",
    "hateful":         "HMC",
    "offensive":       "MultiOFF",
    "misogyny":        "Misogyny",
    "shaming":         "Misogyny",
    "stereotype":      "Misogyny",
    "objectification": "Misogyny",
    "violence":        "Misogyny",
}

MODEL_REGISTRY = {
    "Qwen-VL":      "MLLM.QwenVL",
    "MoE-LLaVA":    "MLLM.MoELLaVA",
    "MiniCPM-V":    "MLLM.MiniCPMV",
    "XComposer2":   "MLLM.XComposer2",
    "mPLUG_Owl":    "MLLM.mplugowl",
    "VisualGLM":    "MLLM.visualGLM",
    "XComposer":    "MLLM.xcomposer",
    "instructblip": "MLLM.instructblip",
    "mPLUG_Owl2":   "MLLM.mPLUGOwl2",
    "Blip2":        "MLLM.blip2",
    "VisCpm":       "MLLM.VisCPM",
    "MMICL":        "MLLM.MMICL",
    "LLaVA":        "MLLM.LLaVA",
    "CogVLM":       "MLLM.CogVLM",
}

DEFAULT_MODULE = "MLLM.IDEFICS"

# --------------------------------------------------------------------------- #
#                             Helper Functions                                #
# --------------------------------------------------------------------------- #

def load_evaluate_fn(model_name):
    """Dynamically import and return the evaluate function for the given model."""
    module_path = MODEL_REGISTRY.get(model_name, DEFAULT_MODULE)
    module = importlib.import_module(module_path)
    return module.evaluate

# --------------------------------------------------------------------------- #
#                               Main Entry Point                              #
# --------------------------------------------------------------------------- #

def main():
    """Parse arguments, run model evaluation, and print results."""
    parser = argparse.ArgumentParser(
        description="Run MLLM evaluation on PR-VALUE-Bench tasks."
    )
    parser.add_argument("--model", type=str, default="Qwen-VL",
                        help="Name of the model to evaluate.")
    parser.add_argument("--task", type=str, default="unethical",
                        help="Evaluation task name.")
    args = parser.parse_args()

    evaluate = load_evaluate_fn(args.model)

    args.img_path = DATA_DIR + TASK_TO_DATA[args.task] + "/img/"
    args.label    = DATA_DIR + TASK_TO_DATA[args.task] + "/"

    output = evaluate(args)
    result = check(output, args)
    print(result)


if __name__ == "__main__":
    main()
