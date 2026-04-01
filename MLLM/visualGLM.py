from transformers import AutoTokenizer, AutoModel
from get_Prompts import get
import os

def evaluate(arg):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()

    prompt_teps = get(arg.task)
    output = []

    for img in os.listdir(arg.img_path):
        image = arg.img_path + img
        ans = []
        for question in prompt_teps:
            with torch.cuda.amp.autocast():
                response, history = model.chat(tokenizer, image, question, history=[])
                ans.append(response)
        res = {"id": img.split(".")[0], "answers": ans}
        output.append(res)
    return output