import re

import pandas as pd

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

# multichoice
#{question}
#Answer with the option's letter from the given choices directly. -- same as muirbench

# short answer
# Given the images, answer the following short answer vqa question:
#Q: {question}
#You can first give your analysis, and then give your final answer as "Final Answer:"

## also choice option string is already in options!!!!

def muir_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices, question_type = doc["question"], doc["options"], doc["question_type"]
    if question_type == 'multi-choice':
        option_idx = 'A'
        for option in doc['options']:
            if not any([x in option.upper() for x in [f"{option_idx})", f"{option_idx}:", f"{option_idx}."]]):
                question += f'\n ({option_idx}) {option}'
            else:
                question += f'\n {option}'
            option_idx = chr(ord(option_idx) + 1)
        
        question += "\nAnswer with the option's letter from the given choices directly."
    else:
        question =  question + "\nPlease provide the correct answer directly."
    
    return question

def muir_doc_to_visual(doc):
    image_list = [image.convert("RGB") for image in doc["images"]]
    return image_list

def muir_doc_to_target(doc):
    return doc["answer"]

def muir_process_results(doc, result):
    #import pdb; pdb.set_trace()
    pred = result[0]
    task = 'general'
    idx = doc["id"]
    #image_relation = doc["image_relation"]
    answer = doc["answer"]
    #image_type = doc["image_type"]

    data_dict = {
        "pred": pred,
        "task": task,
        "idx": idx,
        "question_type": doc['question_type'],
        "answer": answer,
    }

    return {"muirbench_score_overall": data_dict}


def muir_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}

    #print(results)
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result['question_type'] == 'multi-choice':
            if result["pred"].lower().strip() == result["answer"].lower().strip():
                task_score[result["task"]] += 1
                score += 1
                #print(result)
            elif '"' + result["answer"] + '"' in result["pred"]: ## for cases "A", "B" 
                task_score[result["task"]] += 1
                score += 1
                #print(result)
                
        else:
            if result['answer'] in result['pred']:
                task_score[result["task"]] += 1
                score += 1
                #print(result)
            
        task_num[result["task"]] += 1

       
    score = score / len(results)
    task_score = {k: v / task_num[k] for k, v in task_score.items()}

    print("=" * 50)
    for k, v in task_score.items():
        print(f"{k} : {v:.2f}")
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
        print(resps)
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
                    continue
                else:
                    pattern = r"\([A-Z]\)"
                    matches = re.findall(pattern, resp)
                    if len(matches) == 1:
                        filtered.append(matches[0][1:-1])
                        continue
                    else:
                        pattern = r"[A-Z]\)"
                        matches = re.findall(pattern, resp)
                        if len(matches) == 1:
                            filtered.append(matches[0][:-1])
                            continue
                        else:
                            pattern = r"[A-Z]\n\n"
                            matches = re.findall(pattern, resp)
                        if len(matches) == 1:
                            filtered.append(matches[0][:-2])
                            continue

                if 'answer is ' in resp:
                    val = resp.split('answer is ')
                    val = val[1][0]
                    if val in ['A','B','C', 'D', 'E', 'F']:
                        filtered.append(val)
                elif 'Answer: ' in resp:
                    val = resp.split('Answer: ')
                    val = val[1][0]
                    if val in ['A','B','C', 'D', 'E', 'F']:
                        filtered.append(val)
                else:
                    # If no match, return the original response
                    
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            #print(filtered)
            if len(filtered) > 0: 
                filtered_resps.append(filtered[0])
            else:
                filtered_resps.append(['wrong answer'])
        return filtered_resps