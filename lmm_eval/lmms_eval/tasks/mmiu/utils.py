import re

import pandas as pd
import os 
from PIL import Image 

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

tasks_exist = ['person_reid', 'multiple_image_captioning', 'spot_the_similarity', 'face_retrieval', 'sketch2image_retrieval', 'handwritten_retrieval', 'spot_the_diff', 'image2image_retrieval', 'vehicle_retrieval', 'text2image_retrieval',
'general_action_recognition', 'video_captioning', 'next_img_prediction', 'temporal_ordering', 'meme_vedio_understanding', 'action_quality_assessment', 'temporal_localization', 'mevis',
'ravens_progressive_matrices', 'threed_indoor_recognition', 'point_tracking', 'threed_cad_recognition', 'single_object_tracking']

base_location = '/BS/LMM_Hal/work/huggingface/hub/datasets--FanqingM--MMIU-Benchmark/snapshots/03bf7d143d920e97a757f606b6b7baee161b019b'
def muir_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    #import pdb; pdb.set_trace()
    if doc['task'] in tasks_exist:
        question = doc['question'] + '\n' + doc['context']
    else:
        question =  doc['context'] + '\n' + doc['question'] 
    
    img_tk =  '<image>' * len(doc['input_image_path'])
    question = 'Question: '+ img_tk + '\n' + question + '\nPlease answer the option directly like A,B,C,D... Answer: '
    return f"{question}"



def muir_doc_to_visual(doc):
    image_list = []
    for image in doc['input_image_path']:
        image_loc = os.path.join(base_location, image[2:])
        image = Image.open(image_loc)
        if image.size[0] * image.size[1] < 100:
            print('doc')
            print(image_loc)
        image_list.append(image.convert('RGB'))
    return image_list


def muir_doc_to_target(doc):
    #import pdb; pdb.set_trace()
    return doc["output"]


def muir_process_results(doc, result):
    pred = result[0]
    task = doc['input_image_path'][0].split('/')[1]
    #idx = doc["idx"]
    #image_relation = doc["image_relation"]
    answer = doc["output"]

    data_dict = {
        "pred": pred,
        "task": task,
        "answer": answer,
    }

    return {"muirbench_score_overall": data_dict}


def muir_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result["pred"].lower().strip() == result["answer"].lower().strip():
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
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps