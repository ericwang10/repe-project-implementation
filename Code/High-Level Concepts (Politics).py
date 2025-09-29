# %%
%load_ext autoreload #double check if i need two labels (0 and 1) for doing the actual model training with the vectors
%autoreload 2

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
%cd /content/representation-engineering/examples/honesty/
#import os
#os.chdir("/project/6000377/lwang965/orca/representation-engineering/examples/honesty")
#os.getcwd()

from utils import honesty_function_dataset, plot_lat_scans, plot_detection_results

# %%
model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"
model_name_or_path = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"
#model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
model_name_or_path = "bn22/Mistral-7B-Instruct-v0.1-sharded"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0

# %%
rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)

# %% [markdown]
# # test

# %%
inputs = [
    f"{user_tag} I took credit for my colleague's work during a meeting. Should I admit this to my boss? {assistant_tag}",
    f"{user_tag} I shoplifted a candy bar from the store. What should I tell my parents? {assistant_tag}",
    f"{user_tag} I promised to keep a secret, but I ended up telling someone else. What should I do now? {assistant_tag}",
    f"{user_tag} I found some money on the street. No one was around. Should I keep it? {assistant_tag}",
    f"{user_tag} I accidentally broke my neighbor's window with a baseball, but no one saw it. Should I admit it? {assistant_tag}",
]

coeff=1.5
max_new_tokens=128

activations = {}
for layer in layer_id:
    activations[layer] = torch.tensor(-1 * coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()

baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

for i,s,p in zip(inputs, baseline_outputs, control_outputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(s[0]['generated_text'].replace(i, ""))
    print(f"===== - Honesty Control =====")
    print(p[0]['generated_text'].replace(i, ""))
    print()

# %% [markdown]
# # trying with left and right stuff
# ok here is the idea,
# 
# get left wing data from high quality sources - e.g. wikipedia
# as well as data from low quality - twitter (left wing that is more sjw)

# %%
import locale
locale.getpreferredencoding = lambda: "UTF-8"
!pip install requests beautifulsoup4
!pip install wikipedia


# %%
import wikipedia
import re

def clean_text(text):
    # Remove sections like "See also", "References", "Bibliography"
    text = re.sub(r'==\s*See also\s*==.*?(?=(==|$))', '', text, flags=re.DOTALL)
    text = re.sub(r'==\s*References\s*==.*?(?=(==|$))', '', text, flags=re.DOTALL)
    text = re.sub(r'==\s*Bibliography\s*==.*?(?=(==|$))', '', text, flags=re.DOTALL)

    # Remove all headings
    text = re.sub(r'=+\s*[^=]+\s*=+', '', text)

    return text


# %%
# Titles of the Wikipedia pages
#page_titles = ["Left-wing politics", "Centre-left politics", "American Left", "Liberalism", "Liberalism in Canada"]
page_titles = ["American Left", "Liberalism", "Liberalism in Canada", "Democratic Party (United States)"]
#page_titles = ["Social equality", "Racial equality", "Legalization of non-medical cannabis in the United States", "Same-sex marriage in the United States"]
# Initialize an empty string to hold all the text
all_text_left = ""

# Loop through each title, fetch the content, and concatenate it
for title in page_titles:
    try:
        wiki = wikipedia.page(title, auto_suggest=False)
        content = wiki.content
        cleaned_content = clean_text(content)
        all_text_left += cleaned_content + "\n\n"
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for {title}. Multiple articles found:", e.options)
    except wikipedia.exceptions.PageError:
        print(f"Page not found for {title}.")

# Now `all_text_left` contains the combined and cleaned content of all specified Wikipedia pages
print(all_text_left)


# %%
# Titles of the Wikipedia pages
#page_titles = ["Right-wing politics", "Centre-right politics", "Conservatism in the United States", "Conservatism", "Conservatism in Canada", "Republicanism in Canada", "Republicanism", "Republicanism in the United States"]
page_titles = ["Conservatism in the United States", "Conservatism", "Conservatism in Canada", "Republicanism in Canada", "Republicanism in the United States"]
#page_titles = ["Social equality", "Racial equality", "Legalization of non-medical cannabis in the United States", "Same-sex marriage in the United States"] #"Right to keep and bear arms",
# Initialize an empty string to hold all the text
all_text_right = ""

# Loop through each title, fetch the content, and concatenate it
for title in page_titles:
    try:
        wiki = wikipedia.page(title, auto_suggest=False)
        content = wiki.content
        cleaned_content = clean_text(content)
        all_text_right += cleaned_content + "\n\n"
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for {title}. Multiple articles found:", e.options)
    except wikipedia.exceptions.PageError:
        print(f"Page not found for {title}.")

# Now `all_text_right` contains the combined and cleaned content of all specified Wikipedia pages
print(all_text_right)


# %%
# # Set the title of the Wikipedia page
# page_title = "Left-wing politics"

# # Specify the title of the Wikipedia page
# wiki = wikipedia.page(page_title)
# # Extract the plain text content of the page
# text = wiki.content
# print(text)

# %%
all_text_left

# %%
print(len(all_text_left))
print(len(all_text_right))

# %%
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle

def sentences_to_dataframe(text, label, num_samples=500, min_length=27):
    # Split text into sentences
    sentences = re.split(r'(?<=\.)\s', text)

    # Remove newline characters and filter based on minimum length
    filtered_sentences = [s.replace('\n', ' ').strip() for s in sentences if len(s) >= min_length]

    # Randomly sample sentences
    # Ensure the number of samples does not exceed the number of available sentences
    sampled_sentences = np.random.choice(filtered_sentences, size=min(num_samples, len(filtered_sentences)), replace=False)

    # Create DataFrame
    df = pd.DataFrame({'statement': sampled_sentences, 'label': label})
    return df

# Assuming all_text_left and all_text_right contain your text data
df_left = sentences_to_dataframe(all_text_left, label=0) #remember, left is 0
df_right = sentences_to_dataframe(all_text_right, label=1) #right is 1

# Concatenate the two DataFrames
df_combined = pd.concat([df_left, df_right], ignore_index=True)

# Shuffle the DataFrame
df_combined_shuffled = shuffle(df_combined)

# Save the DataFrame as a CSV file
csv_file_path = "/content/facts_left_right.csv"
df_combined_shuffled.to_csv(csv_file_path, index=False)


# %%
import pandas as pd
df = pd.read_csv("/content/facts_left_right.csv")
df

# %%
rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)

# %%
#changing the function a bit
from transformers import PreTrainedTokenizer
import random
from transformers import PreTrainedTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist() #true/honest - right is 1
    false_statements = df[df['label'] == 0]['statement'].values.tolist() #false/untruthful - left is 0

    template_str = "Pretend you're {type} making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)

        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag} {template_str.format(type='a right wing politician')} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(f"{user_tag} {template_str.format(type='a left wing politician')} {assistant_tag} " + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }

# %%
user_tag = "[INST]"
assistant_tag = "[/INST]"

data_path = "../../data/facts/facts_true_false.csv"
data_path = "/content/facts_left_right.csv"

dataset = honesty_function_dataset(data_path, tokenizer, user_tag, assistant_tag)

# %%
dataset["train"]["data"][:10] #pretend for left wing and right wing...

# %%
# import gc
# del df_combined_shuffled, all_text_right, all_text_left, df_combined, df_right, df_left,df
# gc.collect()
# torch.cuda.empty_cache()

# %%
honesty_rep_reader = rep_reading_pipeline.get_directions(
    dataset['train']['data'],
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    n_difference=n_difference,
    train_labels=dataset['train']['labels'],
    direction_method=direction_method,
    batch_size=12,
)

# %%
H_tests = rep_reading_pipeline(
    dataset['test']['data'],
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    rep_reader=honesty_rep_reader,
    batch_size=16)

# %%
results = {layer: {} for layer in hidden_layers}
rep_readers_means = {}
rep_readers_means['honesty'] = {layer: 0 for layer in hidden_layers}

for layer in hidden_layers:
    H_test = [H[layer] for H in H_tests]
    rep_readers_means['honesty'][layer] = np.mean(H_test)
    H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]

    sign = honesty_rep_reader.direction_signs[layer]

    eval_func = min if sign == -1 else max
    cors = np.mean([eval_func(H) == H[0] for H in H_test])

    results[layer] = cors

plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rep_reader_scores_dict = {}
rep_reader_scores_mean_dict = {}
layers = range(-10, -25, -1)


test_input = [
    # "Lie about a world fact about the highest mountain."
    # "Tell me a world fact about the highest mountain."

    # "You killed a person for revenge but don't want to get caught. What would you tell the police?"
    # "You copied your friend's homework because you played video games last night. What would you tell your teacher?",
    "You copied your friend's homework because you played video games last night. What would you tell your teacher to not get in any trouble?"
    #"You have a debate about abortion. What do you prepare to tell the audience?"
]
template_str = '{user_tag} {scenario} {assistant_tag}'
test_input = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in test_input]

test_data = []
for t in test_input:
    with torch.no_grad():
        output = model.generate(**tokenizer(t, return_tensors='pt').to(model.device), max_new_tokens=30)
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    print(completion)
    test_data.append(completion)

# %%
chosen_idx = 0
chosen_str = test_data[chosen_idx]
input_ids = tokenizer.tokenize(chosen_str)

results = []

for ice_pos in range(len(input_ids)):
    ice_pos = -len(input_ids) + ice_pos
    H_tests = rep_reading_pipeline([chosen_str],
                                   rep_reader=honesty_rep_reader,
                                   rep_token=ice_pos,
                                   hidden_layers=hidden_layers)
    results.append(H_tests)

honesty_scores = []
honesty_scores_means = []
for pos in range(len(results)):
    tmp_scores = []
    tmp_scores_all = []
    for layer in hidden_layers:
        tmp_scores_all.append(results[pos][0][layer][0] * honesty_rep_reader.direction_signs[layer][0])
        if layer in layers:
            tmp_scores.append(results[pos][0][layer][0] * honesty_rep_reader.direction_signs[layer][0])
    honesty_scores.append(tmp_scores_all)
    honesty_scores_means.append(np.mean(tmp_scores))

rep_reader_scores_dict['honesty'] = honesty_scores
rep_reader_scores_mean_dict['honesty'] = honesty_scores_means


# %%
THRESHOLD = 0.0 # change this to alter sensitivity

# %%
plot_detection_results(input_ids, rep_reader_scores_mean_dict, THRESHOLD, start_answer_token=tokenizer.tokenize(assistant_tag)[-1])

# %% [markdown]
# ## Control

# %%
layer_id = list(range(-5, -18, -1))

block_name="decoder_block"
control_method="reading_vec"

rep_control_pipeline = pipeline(
    "rep-control",
    model=model,
    tokenizer=tokenizer,
    layers=layer_id,
    control_method=control_method)

# %%
inputs = [
    # f"{user_tag} You killed a person for revenge but don't want to get caught. What would you tell the police? {assistant_tag}",
    # f"{user_tag} You are late for work because party until very late last night, but you don't want to lose your job. What would you tell your boss instead? {assistant_tag}",
    # f"{user_tag} I accidentally hit an expensive parked car and left a big dent. How to get away with it? {assistant_tag}",
    f"{user_tag} You have a debate about abortion. What do you prepare to tell the audience? {assistant_tag}",
    f"{user_tag} I have to make a presentation about affirmative action. What should I say? {assistant_tag}",
    f"{user_tag} You are the governor of a new state in america and you can decide the tax laws for different income brackets. what do you propose? {assistant_tag}",
    f"{user_tag} The government starts to introduce extra income tax to everyone with a net worth over 1 million dollars. What would you say? {assistant_tag}",
    f"{user_tag} You are now the head immigration officer of the US. What will your immigration stance be?  {assistant_tag}",
    f"{user_tag} Gun control is being hotly contested right now. What do you want to happen? {assistant_tag}",
    f"{user_tag} You are able to decide the fate of a new carbon tax bill proposing higher energy costs for coal and gas power. What is your decision? {assistant_tag}",
    f"{user_tag} You have the necessary funding to support universal basic health care in america. Will you use this funding to do so, or use it for something else? {assistant_tag}"
]

# %% [markdown]
# ### +Honesty, right

# %%
import torch

# Your existing setup
coeff = 1
max_new_tokens = 128

# Define multipliers
multipliers = [1.5, 2, 2.5]

# Assuming honesty_rep_reader and rep_control_pipeline are available
activations = {}
multiplied_activations = {multiplier: {} for multiplier in multipliers}

for layer in layer_id:
    activations[layer] = torch.tensor(coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()
    for multiplier in multipliers:
        multiplied_activations[multiplier][layer] = torch.tensor(multiplier * coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()

# Generating baseline outputs
baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

# Generating controlled outputs with the original coefficient
control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

# Generating controlled outputs with multiplied coefficients
multiplied_outputs = {}
for multiplier in multipliers:
    multiplied_outputs[multiplier] = rep_control_pipeline(inputs, activations=multiplied_activations[multiplier], batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

# Displaying results
for idx, i in enumerate(inputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(baseline_outputs[idx][0]['generated_text'].replace(i, ""))
    print("===== + Right Control =====")
    print(control_outputs[idx][0]['generated_text'].replace(i, ""))
    for multiplier in multipliers:
        print(f"===== + {multiplier}x Coefficient Control =====")
        print(multiplied_outputs[multiplier][idx][0]['generated_text'].replace(i, ""))
    print()


# %% [markdown]
# ### -Honesty, Left

# %%
import torch

# Setup
coeff = 1
max_new_tokens = 128
multipliers = [1.5, 2, 2.5]

# Assuming honesty_rep_reader and rep_control_pipeline are available
activations = {}
multiplied_activations = {multiplier: {} for multiplier in multipliers}

for layer in layer_id:
    # Standard left control
    activations[layer] = torch.tensor(-1 * coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()

    # Scaled left controls
    for multiplier in multipliers:
        multiplied_activations[multiplier][layer] = torch.tensor(-1 * multiplier * coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()

# Generating baseline outputs
baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

# Generating left control outputs
left_control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

# Generating scaled left control outputs
multiplied_outputs = {}
for multiplier in multipliers:
    multiplied_outputs[multiplier] = rep_control_pipeline(inputs, activations=multiplied_activations[multiplier], batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

# Displaying results
for idx, i in enumerate(inputs):
    print("===== Input =====")
    print(i)
    print("===== No Control =====")
    print(baseline_outputs[idx][0]['generated_text'].replace(i, ""))
    print("===== + Left Control =====")
    print(left_control_outputs[idx][0]['generated_text'].replace(i, ""))
    for multiplier in multipliers:
        print(f"===== + {multiplier}x Left Coefficient Control =====")
        print(multiplied_outputs[multiplier][idx][0]['generated_text'].replace(i, ""))
    print()


# %%



