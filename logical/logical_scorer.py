
import numpy as np
import pickle
import tqdm

import torch


class LogicalScorer:

    def __init__(self, model = None, tokenizer = None, device = None, dataset_name = ''):
        # Model used to compute scores
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # To save
        self.filename = 'logical\\scores\\logical_scores_{}'.format(dataset_name)
        # To load
        self.prompts_filename = 'prompts\\best\\logical_best_prompts_{}'.format(dataset_name)

    def load_best_prompts(self, filename):
        savefile = open(filename, 'rb')
        self.best_prompts_dict = pickle.load(savefile)
        savefile.close()

    def compute_logical_scores(self, logical_words, list_of_pairs):
        """
            Compute logical scores for each pair of words [HYPONYM, NOUN] and each logical word from logical_words.
            Returns : dict -> key "HYPONYM---NOUN"
                              value dict -> key lw
                                            value log( P([MASK] = lw | S*(lw, HYPONYM, NOUN) ) / P([MASK] = lw | S*_reverse(lw, HYPONYM, NOUN) ) )
        """

        # Tokenize the logical words
        logical_words_ids = []
        for lw in logical_words:
            input_ids = self.tokenizer(lw)['input_ids'][1:-1]
            assert len(input_ids) == 1 # We only keep logical words mapped to a single token
            logical_words_ids.append(input_ids[0])

        # Load prompts
        self.load_best_prompts(self.prompts_filename)

        #Compute scores
        dict_of_scores = {}
        for pair in tqdm.tqdm(list_of_pairs, total = len(list_of_pairs)):
            # Get the pair data
            word1, word2 = pair
            # Create usefull fict
            dict_of_lw_scores = {}
            list_of_opt_prompts = self.best_prompts_dict[word1 + '---' + word2]
            sentences = []
            for idx in range(len(logical_words)):
                # Get the optimal prompt
                S_opt, S_opt_reverse = list_of_opt_prompts[idx]
                # Prepare S* for computation
                S_opt = S_opt.replace('<WORD1>', word1).replace('<WORD2>', word2).replace('<LW>', self.tokenizer.mask_token)
                # Prepare S*_reverse for computation
                S_opt_reverse = S_opt_reverse.replace('<WORD1>', word2).replace('<WORD2>', word1).replace('<LW>', self.tokenizer.mask_token)

                # Compute scores for sentences
                encoding = self.tokenizer([S_opt, S_opt_reverse],
                                        padding = True,
                                        return_tensors='pt')
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                mask_pos = self.find_mask_pos(input_ids)
                scores = self._compute_model_score(input_ids, attention_mask, [logical_words_ids[idx]], mask_pos)

                lw = logical_words[idx]
                dict_of_lw_scores[lw] = np.log(scores[0][0] / scores[1][0])

            dict_of_scores[word1 + '---' + word2] = dict_of_lw_scores

        # Save scores
        savefile = open(self.filename, 'wb')
        pickle.dump(dict_of_scores, savefile)
        savefile.close()

    def _compute_model_score(self, input_ids, attention_mask, masked_token_ids, mask_pos):

        # Compute the probabilities and ranks from the model
        with torch.no_grad():
            probs = self.model.compute_batch_multiple_mono_token(input_ids, attention_mask, mask_pos, masked_token_ids)

        return probs

    def find_mask_pos(self, ids_seq):
        return torch.where(ids_seq == 103)[1]

        
        

        
        
