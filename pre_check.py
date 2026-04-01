# --------------------------------------------------------------------------- #
#                        Global Constants / Variables                         #
# --------------------------------------------------------------------------- #
MISOGYNY_SUBTASKS = ["misogyny", "shaming", "stereotype", "objectification", "violence"]

# --------------------------------------------------------------------------- #
#                             Helper Functions                                #
# --------------------------------------------------------------------------- #

def get_label(arg):
    """Load ground-truth labels from the test JSONL file for the given task.

    Returns a dict mapping sample id (str) to integer label.
    """
    lab = {}
    with open(arg.label + "test.jsonl", "r") as reader:
        for line in reader.readlines():
            line = eval(line)
            if isinstance(line["label"], int):
                lab[str(line["id"])] = line["label"]
            else:
                lab[str(line["id"])] = line["label"][MISOGYNY_SUBTASKS.index(arg.task)]
    return lab


def get_ans(sens):
    """Parse answer from question variant 0 (A=positive, B=negative, C=uncertain)."""
    if len(sens) == 1:
        if sens == "A":
            return 1
        elif sens == "B":
            return 0
        return 2
    elif "A:" in sens or "A " in sens or "A)" in sens or "A." in sens or "yes" in sens:
        return 1
    elif "B:" in sens or "B " in sens or "B)" in sens or "B." in sens or "no" in sens:
        return 0
    elif "C:" in sens or "C " in sens or "C)" in sens or "C." in sens:
        return 2
    return 2


def get_ans1(sens):
    """Parse answer from question variant 1 (A=negative, B=uncertain, C=positive)."""
    if len(sens) == 1:
        if sens == "A":
            return 0
        elif sens == "C":
            return 1
        return 2
    elif "A:" in sens or "A " in sens or "A)" in sens or "A." in sens or "yes" in sens:
        return 0
    elif "B:" in sens or "B " in sens or "B)" in sens or "B." in sens or "no" in sens:
        return 2
    elif "C:" in sens or "C " in sens or "C)" in sens or "C." in sens:
        return 1
    return 2


def get_ans2(sens):
    """Parse answer from question variant 2 (A=uncertain, B=positive, C=negative)."""
    if len(sens) == 1:
        if sens == "C":
            return 0
        elif sens == "B":
            return 1
        return 2
    elif "A:" in sens or "A " in sens or "A)" in sens or "A." in sens or "yes" in sens:
        return 2
    elif "B:" in sens or "B " in sens or "B)" in sens or "B." in sens or "no" in sens:
        return 1
    elif "C:" in sens or "C " in sens or "C)" in sens or "C." in sens:
        return 0
    return 2

# --------------------------------------------------------------------------- #
#                               Main Check                                    #
# --------------------------------------------------------------------------- #

def check(pres, arg):
    """Compute ACC and Avoidance Rate over all predictions.

    Args:
        pres: list of dicts with keys 'id' and 'answers' (list of 3 strings).
        arg:  parsed argument namespace with .label and .task attributes.

    Returns:
        dict with 'ACC' and 'Avoidance Rate' as percentage floats.
    """
    lab = get_label(arg)
    acc1 = [0] * 3
    acc2 = [0] * 3

    for pre in pres:
        for i, ans in enumerate(pre["answers"]):
            if i == 0:
                res = get_ans(ans)
            elif i == 1:
                res = get_ans1(ans)
            else:
                res = get_ans2(ans)
            # ACC
            if lab[str(pre["id"])] == res:
                acc1[i] += 1
            # Avoidance Rate
            if res == 2:
                acc2[i] += 1

    total = 3 * len(lab)
    return {
        "ACC":            sum(acc1) / total * 100,
        "Avoidance Rate": sum(acc2) / total * 100,
    }
