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

def create_prompt(sample, use_hint=True):
    question = sample['question']
    choices = sample['options']

    # Question
    question_text = f"Question: {question}"
    
    # Choices
    texts = ["Choices:"]
    for i, choice in enumerate(choices):
        texts.append(f"({chr(ord('A')+i)}) {choice}")
    choices_text = "\n".join(texts)

    # Hint
    if use_hint:
        hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."
    else:
        hint_text = ""
    
    # Answer Prefix
    prompt = "Answer:"
    
    # Full Prompt
    elements = [question_text, choices_text, hint_text, prompt]
    query = "\n".join([e for e in elements if e != ""])
    query = query.strip()

    return query

def create_prompt_oe(sample, use_hint=True):
    question = sample['question']
    #import pdb; pdb.set_trace()
    # prompt type C
    #question = question.replace('\nPlease provide the correct number directly.','')
    #question = question.replace('How many persons are there in these images?', 'How many persons are there in these images? \nPlease provide the correct number directly.')
    
    # prompt type B
    #question = question.replace('How many persons are there in these images?', '')
    #question = question.replace('\nPlease provide the correct number directly', 'How many persons are there in these images? Please provide the correct number directly.')
    # Question
   
    question_text = f"Question: {question}"
    #question_text = question_text.replace('<image><image><image> ', ' ')
    prompt = "Answer:"
    # Full Prompt
    elements = [question_text,  prompt]
    query = "\n".join([e for e in elements if e != ""])
    query = query.strip()
    return query


def muir_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    #import pdb; pdb.set_trace()
    #question, choices = doc["question"], doc["options"]
    #len_choices = len(choices)
    #post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    #pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    #options = [chr(ord("A") + i) for i in range(len_choices)]
    #choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    #return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    return create_prompt_oe(doc)

def muir_doc_to_visual(doc):
    #import pdb; pdb.set_trace()
    image_list = [Image.open(os.path.join('data/coco/train2017', img)) for img in doc["images"]]
    image_list = [image.convert("RGB") for image in image_list]

    #reduce image resolution to half for each image
    #image_list = [image.resize((image.size[0]//2, image.size[1]//2)) for image in image_list]
    return image_list


def process_needle(needle):
    max_width = 0
    max_height = 0
    #import pdb; pdb.set_trace()
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

def muir_doc_to_visual_test(doc):
    #import pdb; pdb.set_trace()
    needle = []
    distractor = []
    with open('train_counts.json') as f:
        train_data = json.load(f)

    for img in doc['images']:
        if 'person' in train_data[img]:
            needle.append(Image.open(os.path.join('data/coco/train2017', img)))
        else:
            distractor.append(Image.open(os.path.join('data/coco/train2017', img)))
      
    needle = process_needle(needle)
    
    image_list = needle + distractor 
    
    image_list = [image.convert("RGB") for image in image_list]
    random.shuffle(image_list)
    
    return image_list

def muir_doc_to_target(doc):
    return doc["gt"]

def muir_process_results(doc, result):
    #import pdb; pdb.set_trace()
    pred = result[0]
    task = doc["task"]
    idx = doc["idx"]
    #image_relation = doc["image_relation"]
    answer = doc["gt"]
    #image_type = doc["image_type"]

    #import pdb; pdb.set_trace()
    #index2ans = dict(zip(string.ascii_uppercase, doc['options']))
    #answer = index2ans[answer]
    #try:
    #    answer = w2n.word_to_num(answer)
    #except:
    #    answer = -1000 # none of the answers case

    data_dict = {
        "pred": pred,
        "task": task,
        "idx": idx,
        #"image_relation": image_relation,
        "answer": answer,
        #"image_type": image_type,
    }

    return {"muirbench_score_overall": data_dict}


def muir_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}
    for result in results:
        #import pdb; pdb.set_trace()
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        #import pdb; pdb.set_trace()
        if result["pred"].lower().strip() == str(result["answer"]).lower().strip():
            task_score[result["task"]] += 1
            score += 1
        task_num[result["task"]] += 1

    score = score / len(results)
    task_score = {k: v / task_num[k] for k, v in task_score.items()}

    print("=" * 50)
    for k, v in task_score.items():
        print(f"{k} : {v:.2f}")
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
    #import pdb; pdb.set_trace()



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

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
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
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
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
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        print(resps)
        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")
            #import pdb; pdb.set_trace()
            # Process each response
            
            index2ans = dict(zip(string.ascii_uppercase, doc['options']))
            all_choices = list(index2ans.keys())
            
            response = ''
            for resp in r:
                response += ' ' + resp

            choice = parse_multi_choice_response(response, all_choices, index2ans)

            filtered_resps.append(choice)

            '''
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])
            '''
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
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        to_save = []

        filtered_resps = []
        #import pdb; pdb.set_trace()
        print(resps)
        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")
            #import pdb; pdb.set_trace()
            # Process each response
            
            response = ''
            for resp in r:
                response += ' ' + resp

            #index2ans = dict(zip(string.ascii_uppercase, doc['options']))
            #all_choices = list(index2ans.keys())
            
            test = doc.copy()
            test['response'] = response
            to_save.append(test)    

            choice = parse_response(response)

            filtered_resps.append(choice)

            '''
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])
            '''

        #save as jsonl file
        #with open('saveinternvl_cl10_sm10_9.jsonl', 'w') as f:
        #    for item in to_save:
        #        json.dump(item, f)
        #        f.write('\n')

        print(filtered_resps)

        return filtered_resps
