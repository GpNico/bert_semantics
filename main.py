

import torch
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import pickle
import argparse
import os

from data.data_loader import load_dataset
from models.bert_multi_tokens import MultiTokensBERT
from prompts.prompt_scorer import PromptScorer
from prompts.prompt_selector import PromptSelector
from content.content_scorer import ContentScorer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("We found the following device : {}".format(device))




if __name__ == '__main__':

    # To be changed
    pre_trained_model_name = 'bert-base-uncased'
    dataset_name = 'wordnet'

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)

    # Load data
    print("Loading {} dataset...".format(dataset_name))
    list_of_pairs = load_dataset(dataset_name = dataset_name, tokenizer = tokenizer) # We give the tokenizer to rule out [UNK] tokens
    print("Loaded {} pairs!".format(len(list_of_pairs)))

    # Create model
    model = MultiTokensBERT(pre_trained_model_name, device)

    # PromptScorer
    print("Computing prompts scores...")
    prompt_scorer = PromptScorer(model = model, tokenizer = tokenizer, device = device, dataset_name = dataset_name)
    if os.path.exists(prompt_scorer.filename):
        print("Prompts scores already computed!")
    else:
        prompt_scorer.compute_all_pairs_scores(list_of_pairs)

    # PromptSelector
    print("Selecting best prompts...")
    prompt_selector = PromptSelector(dict_of_prompts = prompt_scorer.dict_of_prompts, dataset_name = dataset_name)
    if os.path.exists(prompt_selector.filename):
        print("Best Prompts already computed!")
    else:
        prompt_selector.compute_best_prompts(prompt_scorer.filename)

    # Computing Content Scores
    print("Computing content scores...")
    content_scorer = ContentScorer(model = model, tokenizer = tokenizer, device = device, dataset_name = dataset_name)
    if os.path.exists(content_scorer.filename):
        print("Content scores already computed!")
    else:
        content_scorer.compute_content_scores(list_of_pairs)

    # TEMP
    #savefile = open(content_scorer.filename, 'rb')
    #content_scores = pickle.load(savefile)
    #savefile.close()
        

        
        
