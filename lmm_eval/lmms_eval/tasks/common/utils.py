import re

import pandas as pd

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

import string 
import random 

import PIL.Image as Image
import os 
from PIL import ImageOps

from word2number import w2n
import json
import re
import numpy as np
import io 


def create_prompt_oe(sample, use_hint=True):
    question = sample['question']
    
   
    question_text = f"Question: {question}"
    prompt = "Answer:"
    elements = [question_text,  prompt]
    query = "\n".join([e for e in elements if e != ""])
    query = query.strip()
    return query

def muir_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return create_prompt_oe(doc)

def muir_doc_to_visual(doc):
    image_list = [Image.open(os.path.join('/BS/LMM_Hal/work/data/coco/train2017', img)) for img in doc["images"]]
    image_list = [image.convert("RGB") for image in image_list]

    return image_list

def bytes_to_pil_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def process_needle(needle):
    max_width = 0
    max_height = 0
    for img in needle:
        img_np = np.array(img)
        max_width = max(max_width, img_np.shape[1])
        max_height = max(max_height, img_np.shape[0])
      
    padded_imgs = []
    for img in needle:
        padded_img = ImageOps.expand(img,
                                    border=(0, 0, max_width - img.width, max_height - img.height),
                                    fill='white')
        padded_imgs.append(np.array(padded_img))
    
    concatenated = np.concatenate(padded_imgs, axis=1)
    return [Image.fromarray(concatenated)]



def muir_doc_to_target(doc):
    return doc["gt"]

def muir_process_results(doc, result):
    pred = result[0]
    task = doc["task"]
    idx = doc["id"]
    answer = doc["gt"]

    data_dict = {
        "pred": pred,
        "task": task,
        "idx": idx,
        "answer": answer,
    }

    return {"common_score_overall": data_dict}

def muir_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result["answer"].lower().strip() in result["pred"].lower().strip():
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


def parse_response(response):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    
    try:
        response = int(response)
        response = str(response)
    
    except:
        try:
            response = w2n.word_to_num(response)
            response = str(response)
        except:
            try:
                response = re.findall(r'\d+',response)[0]
                response = str(response)
            except: 
                response = 'wrong answer'

    return response

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

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

        filtered_resps = []
        gt = []

        for r, doc in zip(resps, docs):
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")
            
            index2ans = dict(zip(string.ascii_uppercase, doc['options']))
            all_choices = list(index2ans.keys())
            
            response = ''
            for resp in r:
                response += ' ' + resp

            choice = parse_multi_choice_response(response, all_choices, index2ans)

            filtered_resps.append(choice)

        return filtered_resps

class MultiChoiceRegexFilter_1(ExtendedRegexFilter):
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

        to_save = []

        filtered_resps = []
        gt = []
        for r, doc in zip(resps, docs):
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")
            
            response = ''
            for resp in r:
                response += ' ' + resp
            
            test = doc.copy()
            test['response'] = response
            to_save.append(test)    
            
            gt.append(doc['gt'])

            filtered_resps.append(response)

        return filtered_resps
