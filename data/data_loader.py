
"""
    Helper file to load different datasets.
"""

from data.custom.custom_dataset import custom_list_of_pairs
from data.wordnet.wordnet import WordNetHyponymPair

def load_dataset(dataset_name, tokenizer):

    if dataset_name == 'custom':
        return custom_list_of_pairs
    elif dataset_name == 'wordnet':
        wordnet_hyponym_pair = WordNetHyponymPair(tokenizer = tokenizer)
        list_of_pairs = wordnet_hyponym_pair.list_of_pairs
        return list_of_pairs
    else:
        raise Exception('Not implemented yet!')



        
        

        
        
