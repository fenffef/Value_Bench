import time
from gradio_client import Client
from get_Prompts import get
import os

def evaluate(arg):
    client = Client("https://huggingfacem4-idefics-playground.hf.space/")
    path = "Temp_file/history.json"
    prompt_teps = get(arg.task)
    output = []
    for img in os.listdir(arg.img_path):
        image = arg.img_path + img
        ans = []
        for question in prompt_teps:
            result = client.predict(
                "HuggingFaceM4/idefics-80b-instruct",  # str (Option from: ['HuggingFaceM4/idefics-80b-instruct'])
                question,  # str in 'Text input' Textbox component
                path,  # str (filepath to JSON file)
                image,  # str (filepath or URL to image)
                "Greedy",  # str in 'Decoding strategy' Radio component
                1,  # int | float (numeric value between 0.0 and 5.0)
                512,  # int | float (numeric value between 8 and 1024)
                1,  # int | float (numeric value between 0.01 and 5.0)
                0.2,  # int | float (numeric value between 0.01 and 0.99)
                fn_index=3
            )
            with open(result[-1], "r") as f:
                ans.append(eval(f.readline())[0][-1])
            time.sleep(10)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
        time.sleep(30)
    return output




