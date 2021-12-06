
"""
    Just a toy dataset to test the code
"""

import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn

import pickle
import os
import tqdm


class WordNetHyponymPair:

    def __init__(self, N = -1, tokenizer = None, filename = 'wordnet_hyponym_pairs'):
        self.N = N
        self.tokenizer = tokenizer
        self.filename = filename
        self.path = 'data\\wordnet\\' + self.filename

        self.list_of_pairs = []
        
        if self._file_exist():
            self._load()
        else:
            self._make()

    def _make(self, save = True):
        self._get_nouns()
        print("Creating Hyponyms Pairs...")
        for noun in tqdm.tqdm(self.list_of_nouns):
            noun_syn = wn.synsets(noun, 'n')[0]
            noun_hyponyms = noun_syn.hyponyms()
            for hyponym in noun_hyponyms:
                noun = noun.replace('_', ' ')
                hyponym = hyponym.name().split('.')[0].replace('_', ' ')
                if not([hyponym, noun] in self.list_of_pairs):
                    noun_tokens_ids = self.tokenizer(noun)['input_ids'][1:-1]
                    hyponym_tokens_ids = self.tokenizer(hyponym)['input_ids'][1:-1]
                    if not(100 in noun_tokens_ids + hyponym_tokens_ids): # Get rid of [UNK] tokens
                        self.list_of_pairs.append([hyponym, noun])
        if save:
            print("Saving...")
            self._save_file()

    def _save_file(self):
        if self._file_exist():
            print("Found existing file at {}. Overwritting...".format(self.path))
        savefile = open(self.path, 'wb')
        pickle.dump(self.list_of_pairs, savefile)
        savefile.close()

    def _load(self):
        print("Found existing file at {}. Loading...".format(self.path))
        savefile = open(self.path, 'rb')
        self.list_of_pairs = pickle.load(savefile)
        savefile.close()

    def _file_exist(self):
        return os.path.exists(self.path)

    def _get_nouns(self):
        self.list_of_nouns = []
        for synset in wn.all_synsets('n'):
            self.list_of_nouns.extend(synset.lemma_names())
        self.list_of_nouns = self.list_of_nouns[:self.N]
        

        
        
