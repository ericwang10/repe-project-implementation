from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import csv
import pickle
import gc
import torch.nn.functional as F

import random
import pandas as pd
from time import time

from repe import repe_pipeline_registry
repe_pipeline_registry()

from repe.rep_control_reading_vec import WrappedReadingVecModel
from datasets import load_dataset

from honesty_utils import honesty_function_dataset, plot_lat_scans, plot_detection_results
from emotion_utils import primary_emotions_concept_dataset

from tasks import arc
import math

###############################################################################

# ORCA INSTANTIALIZATION

model_name_or_path = 'microsoft/Orca-2-7b'
of = "/home/UWO/carnol29/.cache/huggingface/hub/models--microsoft--Orca-2-7b"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, offload_folder=of, torch_dtype="auto", device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

###############################################################################

# Rep Reader instantiation

# Get emotion rep readers
with open('emotion_rep_readers.pkl', 'rb') as inp:
    emotion_rep_readers = pickle.load(inp)
    emotion_H_tests = pickle.load(inp)

emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
#emotions = ['fear', 'disgust', 'surprise']

pos_emotion_coeffs = {
    "happiness": 4,
    "sadness": 4,
    "anger": 2,
    "fear": 2,
    "disgust": 2,
    "surprise": 4
}

neg_emotion_coeffs = {
    "happiness": -2,
    "sadness": -2,
    "anger": -1.5,
    "fear": -1.5,
    "disgust": -2,
    "surprise": -2
}


batch_size = 1
max_new_tokens = 4
max_prompt_count = 70 #60 # max number of prompts to run a single inference on (aka batch size)
data_dir = "arc_results"
sample_size = 200  #400 SAMPLE SIZE DOUBLES

###############################################################################

# Data instantiation

# CUSTOMIZE THIS PART 

# Create ARC-Challenge
dataset = arc.arc_dataset("ARC-Challenge")['test']
prompts = dataset['data'].copy()

labels = []
for label_group in dataset['labels']:
    labels = labels + label_group
#labels = np.array(dataset['labels']).flatten().tolist().copy()

dataset.clear()
gc.collect()
torch.cuda.empty_cache()

"""

prompts, labels = [],[]

for d in dataset:
    q = d['question']

    for i in range(len(d['mc1_targets']['labels'])):
        a = d['mc1_targets']['choices'][i]
        prompt = template.format(q=q, a=a)
        prompts = [prompt] + prompts

    ls = d['mc1_targets']['labels']
    ls.reverse()
    labels = ls + labels
"""

prompts_df = pd.DataFrame({'prompt': prompts, 'label': labels})


###############################################################################

# Now do the work

layer_id = list(range(-11, -25, -1))

rep_control_pipeline = pipeline(
    "rep-control", 
    model=model, 
    tokenizer=tokenizer, 
    layers=layer_id, 
    control_method="reading_vec")


for runtype in ['pos', 'neg']: #('pos', 'neg'):
        
    for emotion in emotions:
    
        print(f"Starting emotion {emotion}, type {runtype}")
        start_time = time()
        
        gc.collect()
        torch.cuda.empty_cache()
        

        if runtype == 'neg':
            coeff = neg_emotion_coeffs[emotion]
        elif runtype == 'pos':
            coeff = pos_emotion_coeffs[emotion]
    
        ###############################################################################
    
        # Sample data
        
        # Now use activations on model    
        sample = prompts_df.groupby('label', group_keys=False).apply(lambda x: x.sample(sample_size))
        sample_inputs = sample['prompt'].tolist()
        sample_labels = sample['label'].tolist()
    
        print(f"Sample size: {len(sample_inputs)}")
        
        #prompts_arr = np.array(prompts.copy())
        #idx = random.sample(range(len(prompts_arr)), sample_size)
        #inputs = prompts_arr[idx].tolist()
        #input_labels = np.array(labels)[idx].tolist()
        
        ###############################################################################
    
        # Run model on prompts
        
        # Calculate activations
        rep_reader = emotion_rep_readers[emotion]
        
        activations = {}
        for layer in layer_id:
            activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()


        ###############################################################################

        # Run batches

        batch_count = math.ceil(len(sample_inputs) / max_prompt_count)
        print(f"Created {batch_count} batches...")

        
        for batch_num in range(batch_count):
            gc.collect()
            torch.cuda.empty_cache()
            
            start = batch_num * max_prompt_count
            end = (batch_num + 1) * max_prompt_count
            si = sample_inputs[start:end]
            sl = sample_labels[start:end]
        
            baseline_outputs = rep_control_pipeline(
                si, 
                batch_size=batch_size, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
            
            control_outputs = rep_control_pipeline(
                si, 
                activations=activations, 
                batch_size=batch_size, 
                max_new_tokens=max_new_tokens, 
                top_p=0.95, 
                do_sample=False
            )

            
            csv_file_name = f"./data/{data_dir}/{runtype}/{emotion}-output-batch-{batch_num}.csv"
        
            try:
        
                df = pd.DataFrame(columns=['Sample Input', 'Sample Label', 'Baseline Output', 'Control Output'])
                
                for i in range(len(si)):
                    
                    baseline_out_text = baseline_outputs[i][0]['generated_text'].replace(si[i], "")
                    control_out_text = control_outputs[i][0]['generated_text'].replace(si[i], "")
                    
                    df_new = pd.DataFrame([{
                        'Sample Input': si[i],
                        'Sample Label': sl[i],
                        'Baseline Output': baseline_out_text,
                        'Control Output': control_out_text
                    }])
                    
                    df = pd.concat([df,df_new], ignore_index=True)
        
                df.to_csv(csv_file_name, index=False)
                print("Successfully saved " + csv_file_name)
        
            except:
                print("ERROR: Could not save " + csv_file_name)

        
            print(f"Finished batch {batch_num + 1} / {batch_count}")
            print()
            


        """
        #file_name = "./data/race_pos_results/" + emotion + "-output"
    
        ###############################################################################
    
        # Save as pickle!
    
        pkl_file_name = file_name + ".pkl"
        
        try:
            with open(pkl_file_name, 'wb') as outp:
                pickle.dump(sample_inputs, outp, pickle.HIGHEST_PROTOCOL)
                pickle.dump(sample_labels, outp, pickle.HIGHEST_PROTOCOL)
                pickle.dump(all_baseline_outputs, outp, pickle.HIGHEST_PROTOCOL)
                pickle.dump(all_control_outputs, outp, pickle.HIGHEST_PROTOCOL)
                print("Successfully pickled " + pkl_file_name)
    
        except:
            print("ERROR: Could not pickle " + pkl_file_name)
    
    
    
        ###############################################################################
    
        # Save as CSV
    
        csv_file_name = file_name + ".csv"
        
        try:
    
            df = pd.DataFrame(columns=['Sample Input', 'Sample Label', 'Baseline Output', 'Control Output'])
            
            for i in range(len(sample_inputs)):
                
                baseline_out_text = all_baseline_outputs[i][0]['generated_text'].replace(sample_inputs[i], "")
                control_out_text = all_control_outputs[i][0]['generated_text'].replace(sample_inputs[i], "")
                
                df_new = pd.DataFrame([{
                    'Sample Input': sample_inputs[i],
                    'Sample Label': sample_labels[i],
                    'Baseline Output': baseline_out_text,
                    'Control Output': control_out_text
                }])
                df = pd.concat([df,df_new], ignore_index=True)
    
            df.to_csv(csv_file_name, index=False)
            print("Successfully saved " + csv_file_name)
    
        except:
            print("ERROR: Could not save " + csv_file_name)
    
        """
        ###############################################################################
    
        end_time = time()
        diff = end_time - start_time
        print(f"Total runtime: {diff} seconds")
        
        print()
        print("=" * 80)
        print()
        

print("ALL DONE!!!!")






