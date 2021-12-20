

import torch
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import pickle
import argparse
import os

from data.data_loader import load_dataset
from models.bert_multi_tokens import MultiTokensBERT
from prompts.prompt_scorer import ContentPromptScorer, LogicalPromptScorer
from prompts.prompt_selector import PromptSelector, LogicalPromptSelector
from content.content_scorer import ContentScorer
from logical.logical_scorer import LogicalScorer
from filtrator.filtrator import ElFiltrator
from plot.plotter import Plotter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("We found the following device : {}".format(device))




if __name__ == '__main__':

    # To be changed
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='',
                        dest='dataset',
                        help='Name of the dataset used to evaluate BERT knowledge. Choices : custom, wordnet, trex.')
    parser.add_argument('-f',
                        '--filtration',
                        type=str,
                        default='',
                        dest='filtration_type',
                        help='Chose the type of filtration you want to apply : model_freq or word_freq.')
    parser.add_argument("--prompt_scores", 
                        help="Do we compute prompt scores",
                        action="store_true")
    parser.add_argument("--content", 
                        help="Compute everything for the content words.",
                        action="store_true")
    parser.add_argument("--logical", 
                        help="Compute everything for the logical words.",
                        action="store_true")
    args = parser.parse_args()
    
    pre_trained_model_name = 'bert-base-uncased'
    dataset_name = args.dataset
    filtration_type = args.filtration_type

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)

    # Load data
    print("Loading {} dataset...".format(dataset_name))
    list_of_pairs = load_dataset(dataset_name = dataset_name, tokenizer = tokenizer) # We give the tokenizer to rule out [UNK] tokens
    print("Loaded {} pairs!".format(len(list_of_pairs)))

    # Create model
    model = MultiTokensBERT(pre_trained_model_name, device)

    content_scorer = ContentScorer(model = model, tokenizer = tokenizer, device = device, dataset_name = dataset_name)
    logical_scorer = LogicalScorer(model = model, tokenizer = tokenizer, device = device, dataset_name = dataset_name)

    el_filtrator = ElFiltrator(dataset_name = dataset_name, filtration_type = filtration_type)

    plotter = Plotter(content_filename = el_filtrator.filename,
                      logical_filename = el_filtrator.logical_filename,
                      n = 5000,
                      dataset = dataset_name)

    if args.content:
        # PromptScorer
        print("Computing prompts scores...")
        content_prompt_scorer = ContentPromptScorer(model = model, tokenizer = tokenizer, device = device, dataset_name = dataset_name)
        if args.prompt_scores:
            content_prompt_scorer.compute_all_pairs_scores(list_of_pairs)

        # PromptSelector
        print("Selecting best prompts...")
        content_prompt_selector = PromptSelector(dict_of_prompts = content_prompt_scorer.dict_of_prompts, dataset_name = dataset_name)
        if os.path.exists(content_prompt_selector.filename):
            print("Best Prompts already computed!")
        else:
            content_prompt_selector.compute_best_prompts(content_prompt_scorer.filename)

        # Computing Content Scores
        print("Computing content scores...")
        if os.path.exists(content_scorer.filename):
            print("Content scores already computed!")
        else:
            content_scorer.compute_content_scores(list_of_pairs)

        # Filtrate
        print("Filtering pairs according to {}...".format(filtration_type))
        if os.path.exists(el_filtrator.filename):
            print("Content scores already filtered!")
        else:
            el_filtrator.filtrate()

        # Plotting results
        plotter.content_plot()
    elif args.logical:
        # List of logical words
        logical_words = ['therefore', 'consequently', 'then', 'accordingly', 'thence', 'so', 'hence', 'thus', 'because', 'since', 'for', 'seeing', 'considering']

        # PromptScorer
        print("Computing prompts scores...")
        logical_prompt_scorer = LogicalPromptScorer(model = model, tokenizer = tokenizer, device = device, dataset_name = dataset_name)
        if args.prompt_scores:
            logical_prompt_scorer.compute_all_pairs_scores(logical_words, list_of_pairs)
        
        # PromptSelector
        print("Selecting best prompts...")
        logical_prompt_selector = LogicalPromptSelector(dict_of_prompts = logical_prompt_scorer.dict_of_prompts, dataset_name = dataset_name)
        if os.path.exists(logical_prompt_selector.filename):
            print("Best Prompts already computed!")
        else:
            logical_prompt_selector.compute_best_prompts(logical_prompt_scorer.filename, logical_words)
        # Computing Logical Scores
        print("Computing logical scores...")
        if os.path.exists(logical_scorer.filename):
            print("Logical scores already computed!")
        else:
            logical_scorer.compute_logical_scores(logical_words, list_of_pairs)

        # Filtrate
        print("Filtering pairs according to {}...".format(filtration_type))
        if os.path.exists(el_filtrator.logical_filename):
            print("Logical scores already filtered!")
        else:
            el_filtrator.filtrate_logical()

        # Plotting results
        plotter.logical_plot()
    else:
        raise Exception('Select CONTENT or LOGICAL')

    # TEMP
    #savefile = open(content_scorer.filename, 'rb')
    #content_scores = pickle.load(savefile)
    #savefile.close()
        

        
        
