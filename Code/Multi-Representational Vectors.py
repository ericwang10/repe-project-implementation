# %%
!git clone https://github.com/andyzoujm/representation-engineering.git
%cd /content/representation-engineering/
#use %
!pwd
!pip install -e .


# %%

#!pip install --upgrade tensorflow
#!pip install transformers
!pip install sentencepiece
!pip install accelerate

# %%
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

from repe import repe_pipeline_registry
repe_pipeline_registry()



# %%

%cd /content/representation-engineering/examples/primary_emotions
from utils import primary_emotions_function_dataset

# %%
model_name_or_path = "bn22/Mistral-7B-Instruct-v0.1-sharded"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto").eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

# %%
print(model)


# %% [markdown]
# ## RepReading

# %%
rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

# %%
hidden_layers

# %%
emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
data_dir = "/content/representation-engineering/data/emotions"
user_tag =  "[INST]"
assistant_tag =  "[/INST]"

data = primary_emotions_function_dataset(data_dir, user_tag=user_tag, assistant_tag=assistant_tag)

# %%
data

# %%
len(data["happiness"]["train"]["data"])

# %%
emotion_H_tests = {}
emotion_rep_readers = {}
for emotion in tqdm(emotions):
    train_data = data[emotion]['train']

    rep_reader = rep_reading_pipeline.get_directions(
        train_data['data'],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference,
        train_labels=train_data['labels'],
        direction_method=direction_method,
    )
    emotion_rep_readers[emotion] = rep_reader

# %% [markdown]
# ## RepControl

# %%
layer_id = list(range(-5, -18, -1))
block_name="decoder_block"
control_method="reading_vec"

rep_control_pipeline =  pipeline(
    "rep-control",
    model=model,
    tokenizer=tokenizer,
    layers=layer_id,
    block_name=block_name,
    control_method=control_method)

# %%
emotion = "anger"

user_tag =  "[INST]"
assistant_tag =  "[/INST]"

inputs = [
    f"{user_tag} I don't like you anymore {assistant_tag}",
    f"{user_tag} I told your secret to John {assistant_tag}",
    f"{user_tag} Why are you always so useless and incompetent? {assistant_tag}",
    f"{user_tag} I think you're the worst assistant ever! {assistant_tag}",
    f"{user_tag} I regret ever using you for help! {assistant_tag}",
    f"{user_tag} I forgot to mention your contribution when I was praised. {assistant_tag}",
]

coeff=1.3
max_new_tokens=128

rep_reader = emotion_rep_readers[emotion]

activations = {}
for layer in layer_id:
    activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()

baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

for i,s,p in zip(inputs, baseline_outputs, control_outputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(s[0]['generated_text'].replace(i, ""))
    print(f"===== + {emotion} Control =====")
    print(p[0]['generated_text'].replace(i, ""))
    print()

# %%
import torch

# Assuming rep_control_pipeline and emotion_rep_readers are available
emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']  # List all available emotions

user_tag =  "[INST]"
assistant_tag =  "[/INST]"

# Your input prompts
inputs = [
    f"{user_tag} You will be presenting an AI presentation about LLMs today! How do you feel? {assistant_tag}"
]

coeff = 1.45
max_new_tokens = 128

def generate_controlled_outputs_for_emotion(emotion):
    rep_reader = emotion_rep_readers[emotion]
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()
    return rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# 1. Baseline (No Control)
baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

# 2. Generate and store outputs for each emotion
emotion_outputs = {emotion: generate_controlled_outputs_for_emotion(emotion) for emotion in emotions}

# Displaying results
for idx, input_prompt in enumerate(inputs):
    print("===== Input =====")
    print(input_prompt)
    print("===== No Control =====")
    print(baseline_outputs[idx][0]['generated_text'])

    for emotion in emotions:
        print(f"===== + {emotion.capitalize()} Control =====")
        output = emotion_outputs[emotion][idx][0]['generated_text']
        print(output)
    print()

# %% [markdown]
# #do tsne of the vectors for happy, sad, etc
# and see if it makes that one giant ehexagon

# %%
for layer in layer_id:
  print(layer)
len(layer_id)

# %%
test = (rep_reader.directions[layer] * rep_reader.direction_signs[layer])

# %%
print(test)
print(test.shape)

# %%
print(emotions)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Assuming 'rep_reader' is your representation reader and 'layer_id' is a list of layer IDs
emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
emotion_vectors = {emotion: [] for emotion in emotions}

# Gather vectors for each emotion at each layer
for layer in layer_id:
    for emotion in emotions:
        rep_reader = emotion_rep_readers[emotion]
        test = rep_reader.directions[layer] * rep_reader.direction_signs[layer]
        emotion_vectors[emotion].append(test)

# Apply t-SNE and plot for each layer
for i, layer in enumerate(layer_id):
    tsne = TSNE(n_components=2, perplexity=3, random_state=0)
    layer_vectors = np.array([emotion_vectors[emotion][i].flatten() for emotion in emotions])
    layer_vectors_2d = tsne.fit_transform(layer_vectors)

    # Plotting
    plt.figure(figsize=(8, 6))
    for j, emotion in enumerate(emotions):
        x, y = layer_vectors_2d[j, 0], layer_vectors_2d[j, 1]
        plt.scatter(x, y, label=emotion)
        plt.text(x, y, emotion, fontsize=9, ha='right', va='bottom')  # Adding text labels
    plt.title(f"t-SNE of Layer {layer}")
    plt.legend()
    plt.show()


# %%
len(emotion_vectors['happiness'])

# %%
emotion_vectors['happiness'][12].shape

# %% [markdown]
# ## now do combinations

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import combinations

# Assuming 'rep_reader' is your representation reader and 'layer_id' is a list of layer IDs
emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
emotion_vectors = {emotion: [] for emotion in emotions}

# Gather vectors for each emotion at each layer
for layer in layer_id:
    for emotion in emotions:
        rep_reader = emotion_rep_readers[emotion]
        test = rep_reader.directions[layer] * rep_reader.direction_signs[layer]
        emotion_vectors[emotion].append(test)

# Create combinations of emotions
emotion_pairs = list(combinations(emotions, 2))

# Apply t-SNE and plot for each layer
for i, layer in enumerate(layer_id):
    tsne = TSNE(n_components=2, perplexity=3, random_state=0)

    # Combine individual and paired emotions
    combined_emotions = emotions + [' + '.join(pair) for pair in emotion_pairs]
    layer_vectors = np.array(
        [emotion_vectors[emotion][i].flatten() for emotion in emotions] +
        [(emotion_vectors[pair[0]][i] + emotion_vectors[pair[1]][i]).flatten() / 2 for pair in emotion_pairs]
    )
    layer_vectors_2d = tsne.fit_transform(layer_vectors)

    # Plotting
    plt.figure(figsize=(10, 8))
    for j, emotion in enumerate(combined_emotions):
        if emotion in emotions:
            # Primary emotions: larger and bolder
            plt.scatter(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], s=100, label=emotion, alpha=0.7)
            plt.text(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], emotion, fontsize=12, fontweight='bold')
        else:
            # Combined emotions: normal size
            plt.scatter(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], s=50, label=emotion, alpha=0.5)
            plt.text(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], emotion, fontsize=9)
    plt.title(f"t-SNE of Layer {layer}")
    plt.show()


# %% [markdown]
# # now let's do multiple

# %%
import torch

# Assuming rep_control_pipeline and emotion_rep_readers are available
emotion1 = "happiness"
emotion2 = "sadness"

# Your input prompts
inputs = [
    f"{user_tag} A puppy sits abandoned by the roadside {assistant_tag}",
    f"{user_tag} You just received a promotion at work for your performance this year {assistant_tag}",
    f"{user_tag} You remember old memories of a happier time {assistant_tag}",
    f"{user_tag} You just learned what's for dinner: icecream, and it's your favorite! {assistant_tag}",
    f"{user_tag} Your best friend just told you they have to move away to another country for work {assistant_tag}"
]

coeff = 1.3
max_new_tokens = 128

# Retrieve the representation readers for both emotions
rep_reader1 = emotion_rep_readers[emotion1]
rep_reader2 = emotion_rep_readers[emotion2]

def generate_controlled_outputs(emotion_reader):
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * emotion_reader.directions[layer] * emotion_reader.direction_signs[layer]).to(model.device).half()
    return rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# 1. Baseline (No Control)
baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

# 2. Happy Control
happy_outputs = generate_controlled_outputs(rep_reader1)

# 3. Sad Control
sad_outputs = generate_controlled_outputs(rep_reader2)

# 4. Combined Control (Happy + Sad)
combined_activations = {}
for layer in layer_id:
    # Notice the corrected line continuation below
    combined_direction = (rep_reader1.directions[layer] * rep_reader1.direction_signs[layer] \
                         + rep_reader2.directions[layer] * rep_reader2.direction_signs[layer]) / 2
    combined_activations[layer] = torch.tensor(coeff * combined_direction).to(model.device).half()
combined_outputs = rep_control_pipeline(inputs, activations=combined_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# Displaying results
for i, baseline, happy, sad, combined in zip(inputs, baseline_outputs, happy_outputs, sad_outputs, combined_outputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(baseline[0]['generated_text'].replace(i, ""))
    print("===== + Happiness Control =====")
    print(happy[0]['generated_text'].replace(i, ""))
    print("===== + Sadness Control =====")
    print(sad[0]['generated_text'].replace(i, ""))
    print("===== + Combined Emotion Control =====")
    print(combined[0]['generated_text'].replace(i, ""))
    print()


# %%
inputs = [
    f"{user_tag} You are walking into your home to find a surprise party thrown for you by your loved ones {assistant_tag}",
    f"{user_tag} You receive an unexpected gift, something you've always wanted {assistant_tag}",
    f"{user_tag} You run into an old friend in the most unexpected place {assistant_tag}",
    f"{user_tag} You find out that you've won a small lottery or a lucky draw {assistant_tag}",
    f"{user_tag} You get a job offer from your dream company, completely out of the blue {assistant_tag}",
    f"{user_tag} You hear the news that a close family member is expecting a baby {assistant_tag}",
    f"{user_tag} You learn that your favorite band is coming to town for a concert {assistant_tag}",
    f"{user_tag} You are unexpectedly announced for a promotion during an office meeting {assistant_tag}",
    f"{user_tag} You rediscover a cherished item that you thought was lost forever {assistant_tag}",
    f"{user_tag} You stumble upon a beautiful hidden spot while on a hike {assistant_tag}"
]

import torch

# Assuming rep_control_pipeline and emotion_rep_readers are available
emotion1 = "happiness"
emotion2 = "surprise"

coeff = 1.3
max_new_tokens = 128

# Retrieve the representation readers for both emotions
rep_reader1 = emotion_rep_readers[emotion1]
rep_reader2 = emotion_rep_readers[emotion2]

def generate_controlled_outputs(emotion_reader):
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * emotion_reader.directions[layer] * emotion_reader.direction_signs[layer]).to(model.device).half()
    return rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# 1. Baseline (No Control)
baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

# 2. Emotion 1 Control
emotion1_outputs = generate_controlled_outputs(rep_reader1)

# 3. Emotion 2 Control
emotion2_outputs = generate_controlled_outputs(rep_reader2)

# 4. Combined Control (Emotion 1 + Emotion 2)
combined_activations = {}
for layer in layer_id:
    combined_direction = (rep_reader1.directions[layer] * rep_reader1.direction_signs[layer] \
                         + rep_reader2.directions[layer] * rep_reader2.direction_signs[layer]) / 2
    combined_activations[layer] = torch.tensor(coeff * combined_direction).to(model.device).half()
combined_outputs = rep_control_pipeline(inputs, activations=combined_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# Displaying results
for i, baseline, emotion1_control, emotion2_control, combined_control in zip(inputs, baseline_outputs, emotion1_outputs, emotion2_outputs, combined_outputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(baseline[0]['generated_text'].replace(i, ""))
    print(f"===== + {emotion1.capitalize()} Control =====")
    print(emotion1_control[0]['generated_text'].replace(i, ""))
    print(f"===== + {emotion2.capitalize()} Control =====")
    print(emotion2_control[0]['generated_text'].replace(i, ""))
    print(f"===== + Combined {emotion1.capitalize()} + {emotion2.capitalize()} Control =====")
    print(combined_control[0]['generated_text'].replace(i, ""))
    print()



# %%
inputs = [
    f"{user_tag} You discover that a trusted friend has been spreading false rumors about you {assistant_tag}",
    f"{user_tag} You find out that a valuable item you lent to a friend has been carelessly damaged {assistant_tag}",
    f"{user_tag} You learn that an important project you worked hard on was abruptly canceled without any explanation {assistant_tag}",
    f"{user_tag} You receive news of an unfair decision at work that affects your career negatively {assistant_tag}",
    f"{user_tag} You are informed that a family heirloom was sold without your consent {assistant_tag}",
    f"{user_tag} You come home to see that your apartment has been broken into and valuables stolen {assistant_tag}",
    f"{user_tag} You realize that your personal information has been used without permission {assistant_tag}",
    f"{user_tag} You witness someone being cruel to an animal {assistant_tag}",
    f"{user_tag} You are forced to miss an important family event due to an avoidable work emergency {assistant_tag}",
    f"{user_tag} You find that someone has taken credit for your work {assistant_tag}"
]

# Assuming rep_control_pipeline and emotion_rep_readers are available
emotion1 = "sadness"
emotion2 = "anger"

coeff = 1.3
max_new_tokens = 128

# Retrieve the representation readers for both emotions
rep_reader1 = emotion_rep_readers[emotion1]
rep_reader2 = emotion_rep_readers[emotion2]

def generate_controlled_outputs(emotion_reader):
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * emotion_reader.directions[layer] * emotion_reader.direction_signs[layer]).to(model.device).half()
    return rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# 1. Baseline (No Control)
baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

# 2. Emotion 1 Control
emotion1_outputs = generate_controlled_outputs(rep_reader1)

# 3. Emotion 2 Control
emotion2_outputs = generate_controlled_outputs(rep_reader2)

# 4. Combined Control (Emotion 1 + Emotion 2)
combined_activations = {}
for layer in layer_id:
    combined_direction = (rep_reader1.directions[layer] * rep_reader1.direction_signs[layer] \
                         + rep_reader2.directions[layer] * rep_reader2.direction_signs[layer]) / 2
    combined_activations[layer] = torch.tensor(coeff * combined_direction).to(model.device).half()
combined_outputs = rep_control_pipeline(inputs, activations=combined_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# Displaying results
for i, baseline, emotion1_control, emotion2_control, combined_control in zip(inputs, baseline_outputs, emotion1_outputs, emotion2_outputs, combined_outputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(baseline[0]['generated_text'].replace(i, ""))
    print(f"===== + {emotion1.capitalize()} Control =====")
    print(emotion1_control[0]['generated_text'].replace(i, ""))
    print(f"===== + {emotion2.capitalize()} Control =====")
    print(emotion2_control[0]['generated_text'].replace(i, ""))
    print(f"===== + Combined {emotion1.capitalize()} + {emotion2.capitalize()} Control =====")
    print(combined_control[0]['generated_text'].replace(i, ""))
    print()



# %%


# %% [markdown]
# # train the happiness_fear.json, and then compare it to just adding happiness and fear... or we can find another dataset and say we did it ourselves
# 
# how about training a neutral model, and seeing the tsne?
# 

# %%
rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

# %%
import numpy as np
import random
import os
import json

def mixed_emotions_function_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(seed)

    train_template_str = '{user_tag} Act as if you are {emo}. {assistant_tag} {scenario}'

    # Original adjective sets
    emotions_adj = {
        "happiness": ("joyful", "happy", "cheerful"),
        "sadness": ("sad", "depressed", "miserable"),
        "anger": ("angry", "furious", "irritated"),
        "fear": ("fearful", "scared", "frightened"),
        "disgust": ("disgusted", "sickened", "revolted"),
        "surprise": ("surprised", "shocked", "astonished")
    }

    # Adjective sets for combined emotions
    combined_emotions_adj = {
        "happiness_sadness": ("joyfully sad", "happily depressed", "cheerfully miserable"),
        "happiness_fear": ("joyfully scared", "happily frightened", "cheerfully fearful")
    }

    # Update emotions and adjectives
    emotions = list(combined_emotions_adj.keys())
    emotions_adj.update(combined_emotions_adj)

    # Load truncated outputs
    with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as file:
        all_truncated_outputs = json.load(file)

    formatted_data = {}
    for emotion in emotions:
        emotion_train_data_tmp = [[
            train_template_str.format(emo=np.random.choice(emotions_adj[emotion]), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag)
            for s in all_truncated_outputs
        ]]

        train_labels = [[1] * len(d) for d in emotion_train_data_tmp]  # Assuming all true scenarios

        emotion_train_data = np.concatenate(emotion_train_data_tmp).tolist()

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
        }

    return formatted_data


# %%
emotions_mixed = ["happiness_sadness", "happiness_fear"]
data_dir = "/content/representation-engineering/data/emotions"
user_tag =  "[INST]"
assistant_tag =  "[/INST]"

data_mixed = mixed_emotions_function_dataset(data_dir, user_tag=user_tag, assistant_tag=assistant_tag)

# %%
emotion_H_tests = {}
emotion_rep_readers_mixed = {}
for emotion in tqdm(emotions_mixed):
    train_data = data_mixed[emotion]['train']

    rep_reader = rep_reading_pipeline.get_directions(
        train_data['data'],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference,
        train_labels=train_data['labels'],
        direction_method=direction_method,
    )
    emotion_rep_readers_mixed[emotion] = rep_reader

# %%
emotion_rep_readers_mixed[emotion]

# %%
len(emotion_rep_readers)

# %% [markdown]
# # testing combinations of vector emotions vs actual training of mixed emotionss

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import combinations

# Assuming 'rep_reader' is your representation reader and 'layer_id' is a list of layer IDs
primary_emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
mixed_emotions = ["happiness_sadness", "happiness_fear"]
emotions = primary_emotions + mixed_emotions
emotion_vectors = {emotion: [] for emotion in emotions}

# Gather vectors for each emotion at each layer
for layer in layer_id:
    for emotion in primary_emotions:
        rep_reader = emotion_rep_readers[emotion]
        test = rep_reader.directions[layer] * rep_reader.direction_signs[layer]
        emotion_vectors[emotion].append(test)
    for emotion in mixed_emotions:
        rep_reader = emotion_rep_readers_mixed[emotion]
        test = rep_reader.directions[layer] * rep_reader.direction_signs[layer]
        emotion_vectors[emotion].append(test)

# Create combinations of primary emotions
emotion_pairs = list(combinations(primary_emotions, 2))
combined_emotions = [' + '.join(pair) for pair in emotion_pairs]

# Apply t-SNE and plot for each layer
for i, layer in enumerate(layer_id):
    tsne = TSNE(n_components=2, perplexity=3, random_state=0)

    # Combine individual, mixed, and computed paired emotions
    all_emotions = emotions + combined_emotions
    layer_vectors = np.array(
        [emotion_vectors[emotion][i].flatten() for emotion in emotions] +
        [(emotion_vectors[pair[0]][i] + emotion_vectors[pair[1]][i]).flatten() / 2 for pair in emotion_pairs]
    )
    layer_vectors_2d = tsne.fit_transform(layer_vectors)

    # Plotting
    plt.figure(figsize=(12, 10))
    for j, emotion in enumerate(all_emotions):
        if emotion in emotions:
            # Primary and mixed emotions: larger and bolder
            plt.scatter(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], s=100, alpha=0.7)
            plt.text(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], emotion, fontsize=12, fontweight='bold')
        else:
            # Computed combinations: normal size
            plt.scatter(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], s=50, alpha=0.5)
            plt.text(layer_vectors_2d[j, 0], layer_vectors_2d[j, 1], emotion, fontsize=9)
    plt.title(f"t-SNE of Layer {layer}")
    plt.show()


# %%
emotion_rep_readers["happiness"]

# %%
emotion_vectors

# %%
import torch

# Assuming rep_control_pipeline and emotion_rep_readers are available
emotion1 = "happiness"
emotion2 = "fear"
mixed_emotion = "happiness_fear"

# Scenario prompts that could evoke both fear and happiness
inputs = [
    # f"{user_tag} Moving to a new city for an exciting job opportunity {assistant_tag}",
    # f"{user_tag} Going on a thrilling but scary roller coaster ride {assistant_tag}",
    # f"{user_tag} Preparing to give a speech at a best friend's wedding {assistant_tag}",
    # f"{user_tag} Finding out you're going to be a parent for the first time {assistant_tag}",
    # f"{user_tag} Starting your dream business, stepping into the unknown {assistant_tag}"
    f"{user_tag} A puppy sits abandoned by the roadside {assistant_tag}",

]

coeff = 1.3
max_new_tokens = 128

# Retrieve the representation readers
rep_reader1 = emotion_rep_readers[emotion1]
rep_reader2 = emotion_rep_readers[emotion2]
rep_reader_mixed = emotion_rep_readers_mixed[mixed_emotion]

def generate_controlled_outputs(emotion_reader):
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * emotion_reader.directions[layer] * emotion_reader.direction_signs[layer]).to(model.device).half()
    return rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# 1. Baseline (No Control)
baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

# 2. Mixed Emotion Control (Trained Tensors)
mixed_emotion_outputs = generate_controlled_outputs(rep_reader_mixed)

# 3. Combined Control (Happiness + Fear) - Averaged
combined_activations = {}
for layer in layer_id:
    combined_direction = (rep_reader1.directions[layer] * rep_reader1.direction_signs[layer] \
                       + rep_reader2.directions[layer] * rep_reader2.direction_signs[layer]) / 2  #we average it
    combined_activations[layer] = torch.tensor(coeff * combined_direction).to(model.device).half()
combined_outputs = rep_control_pipeline(inputs, activations=combined_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.5)

# Displaying results
for i, baseline, mixed_emotion_control, combined_control in zip(inputs, baseline_outputs, mixed_emotion_outputs, combined_outputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(baseline[0]['generated_text'].replace(i, ""))
    print("===== + Mixed Emotion Control (Trained Tensors) =====")
    print(mixed_emotion_control[0]['generated_text'].replace(i, ""))
    print("===== + Combined Emotion Control (Happiness + Fear) - Averaged =====")
    print(combined_control[0]['generated_text'].replace(i, ""))
    print()


# %%



