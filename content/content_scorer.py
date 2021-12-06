
import numpy as np
import pickle
import tqdm

from prompts.prompt_material import TRANSFORMATIONS


class ContentScorer:

    def __init__(self, model = None, tokenizer = None, device = None, dataset_name = ''):
        # Model used to compute scores
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load transformations names
        self.transformations_names = TRANSFORMATIONS

        # To save
        self.filename = 'content\\scores\\content_scores_{}'.format(dataset_name)
        # To load
        self.prompts_filename = 'prompts\\best\\best_prompts_{}'.format(dataset_name)

    def load_best_prompts(self, filename):
        savefile = open(filename, 'rb')
        self.best_prompts_dict = pickle.load(savefile)
        savefile.close()

    def compute_content_scores(self, list_of_pairs):
        """
            Compute content scores for each pair of words [HYPONYM, NOUN].
            Returns : dict -> key "HYPONYM---NOUN"
                              value dict -> key transf =/= vanilla
                                            value log( P([MASK] = NOUN | S*(HYPONYM, [MASK]) ) / P([MASK] = NOUN | S*_phi(HYPONYM, [MASK]) ) )
        """
        # Load prompts
        self.load_best_prompts(self.prompts_filename)

        #Compute scores
        dict_of_scores = {}
        for pair in tqdm.tqdm(list_of_pairs, total = len(list_of_pairs)):
            # Get the pair data
            word1, word2 = pair
            masked_token_ids_1 = self.tokenizer(word1)['input_ids'][1:-1]
            masked_token_ids_2 = self.tokenizer(word2)['input_ids'][1:-1]
            N_masks_1 = len(masked_token_ids_1)
            N_masks_2 = len(masked_token_ids_2)
            # Create usefull fict
            dict_of_transf_scores = {}
            dict_of_opt_prompts = self.best_prompts_dict[word1 + '---' + word2]
            for transf in self.transformations_names:
                if transf == 'vanilla':
                    continue
                # Get the optimal prompt
                S_opt, S_opt_phi = dict_of_opt_prompts[transf]
                # Prepare S* for computation
                S_opt = S_opt.replace('<WORD1>', word1).replace('<WORD2>', N_masks_2*self.tokenizer.mask_token)
                # Prepare S*_phi for computation
                S_opt_phi, word_masked = self.phi(S_opt_phi, transf, word1, word2, N_masks_1, N_masks_2)
                # Compute scores
                score = self._compute_one_score(S_opt, masked_token_ids_2)
                if word_masked == 1:
                    score_phi = self._compute_one_score(S_opt_phi, masked_token_ids_1)
                elif word_masked == 2:
                    score_phi = self._compute_one_score(S_opt_phi, masked_token_ids_2)
                final_score = np.log(score / score_phi)
                # Save it
                dict_of_transf_scores[transf] = final_score
            dict_of_scores[word1 + '---' + word2] = dict_of_transf_scores

        # Save scores
        savefile = open(self.filename, 'wb')
        pickle.dump(dict_of_scores, savefile)
        savefile.close()

    def _compute_one_score(self, sentence, masked_token_ids):

        # Compute input_ids and attention_mask of the sentence
        encoding = self.tokenizer(sentence,
                     max_length=64, 
                     padding='max_length',
                     return_tensors='pt'
                     )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # The model needs the masks_to_predict_pos
        masks_to_predict_pos = self.find_all_masks_pos(input_ids)

        # Compute the probabilities and ranks from the model
        probs_n_ranks = self.model.compute_greedy(input_ids, attention_mask, masks_to_predict_pos, masked_token_ids)

        # Compute scores
        score = probs_n_ranks[:,0].prod()

        return score


    def find_all_masks_pos(self, ids_seq):
        pos = []
        for k in range(ids_seq.shape[1]):
            if ids_seq[0][k] == 103:
                pos.append(k)
        return pos

    def phi(self, vanilla_sentence, transf, word1, word2, N_masks_1, N_masks_2):
        if transf == 'opposite':
            sentence = vanilla_sentence.replace('<WORD1>', N_masks_2*self.tokenizer.mask_token).replace('<WORD2>', word1)
            word_masked = 2
        elif transf == 'reverse':
            sentence = vanilla_sentence.replace('<WORD1>', word2).replace('<WORD2>', N_masks_1*self.tokenizer.mask_token)
            word_masked = 1

        return sentence, word_masked
        
        

        
        
