import re

import pandas as pd

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

def muir_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    #import pdb; pdb.set_trace()
    prompt = doc['prompt']
    return f"{prompt}"


def muir_doc_to_visual(doc):
    image_list = []
    
    for k in ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']:
        if k in doc and doc[k]:
            image_list.append(doc[k].convert('RGB'))
    return image_list


def muir_doc_to_target(doc):
    return doc["answer"]


def muir_process_results(doc, result):
    pred = result[0]
    task = doc["sub_task"]
    idx = doc["idx"]
    answer = doc["answer"]
    

    data_dict = {
        "pred": pred,
        "task": task,
        "idx": idx,
        "answer": answer,
    }

    return {"muirbench_score_overall": data_dict}


def muir_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}
    
    #print('results ', results)
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result["pred"].lower().strip() == result["answer"].lower().strip()[1:-1]:
            task_score[result["task"]] += 1
            score += 1
        elif result["answer"] in result["pred"]: ### these are the cases where (A)... are in resp. Having (A),(B) in resp would mean correct predictions..., only problem can be if the have multiples of them
            task_score[result["task"]] += 1
            score += 1
        elif '"' + result["answer"][1:-1] + '"' in result["pred"]: ## for cases "A", "B" 
            task_score[result["task"]] += 1
            score += 1
        task_num[result["task"]] += 1

    score = score / len(results)
    task_score = {k: v / task_num[k] for k, v in task_score.items()}

    print("=" * 50)
    for k, v in task_score.items():
        print(f"{k} : {v:.5f}")
    print("=" * 50)
    return score


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []
        #print('initial response ', resps)
        #import pdb; pdb.set_trace()
        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                elif 'answer is ' in resp:
                    val = resp.split('answer is ')
                    val = val[1].split('.')
                    filtered.append(val[0])
                elif 'Answer: ' in resp:
                    val = resp.split('Answer: ')
                    filtered.append(val[1])
                elif 'labeled as ' in resp:
                    val = resp.split('labeled as ')
                    val = val[1].split(' ')
                    filtered.append(val[0])
                elif 'labeled ' in resp:
                    val = resp.split('labeled ')
                    val = val[1].split(' ')
                    if val[0] in ['A', 'B,', 'C', 'D', 'E']:
                        filtered.append(val[0])
                    else:
                        filtered.append(resp)
                else:
                    pattern = r"\([A-Z]\)"
                    matches = re.findall(pattern, resp)
                    if len(matches) == 1:
                        filtered.append(matches[0][1:-1])
                    else:
                        filtered.append(resp)
               
            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])
        #print('filtered response ', filtered_resps)
        return filtered_resps