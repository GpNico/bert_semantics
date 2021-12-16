
import numpy as np
import pickle
import tqdm
import os

import torch

from prompts.prompt_material import DETS_LIST, CONTENT_STRUCTS_PREFIX_LIST, CONTENT_STRUCTS_MIDDLE_LIST, CONTENT_STRUCTS_SUFFIX_LIST, TRANSFORMATIONS, LOGICAL_PREFIXES_LIST, LOGICAL_STRUCTS_LW_LIST

#######################################
#                                     #
#                CONTENT              #
#                                     #
#######################################


class ContentPromptScorer:

    def __init__(self, model = None, tokenizer = None, device = None, dataset_name = ''):
        # Model used to compute scores
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load prompts materials
        self.dets_list = DETS_LIST
        self.structs_dict = {'prefix': CONTENT_STRUCTS_PREFIX_LIST,
                             'middle': CONTENT_STRUCTS_MIDDLE_LIST,
                             'suffix': CONTENT_STRUCTS_SUFFIX_LIST}

        # Load transformations names
        self.transformations_names = TRANSFORMATIONS

        # Define template
        self.vanilla_template = '<PREFIX> <DET1> <WORD1> <MIDDLE> <DET2> <WORD2> <SUFFIX>.'
        self.key_template = '<det1>-<det2>-<prefix>-<middle>-<suffix>'

        # Compute keys
        self._compute_keys()

        # Where to save data
        self.filename = 'prompts\\scores\\content_prompts_scores_{}'.format(dataset_name)

        # Compute Prompts
        self.create_prompts()
        
    def _compute_keys(self):
        """
            Compute all the possible keys in the form idx_{det1}-idx_{det2}-idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
        """
        N_dets = len(self.dets_list)
        N_prefix = len(self.structs_dict['prefix'])
        N_middle = len(self.structs_dict['middle'])
        N_suffix = len(self.structs_dict['suffix'])

        list_of_keys = []
        for idx_det1 in range(N_dets):
            for idx_det2 in range(N_dets):
                for idx_prefix in range(N_prefix):
                    for idx_middle in range(N_middle):
                        for idx_suffix in range(N_suffix):
                            key = self.key_template.replace('<det1>', str(idx_det1)).replace('<det2>', str(idx_det2))
                            key = key.replace('<prefix>', str(idx_prefix)).replace('<middle>', str(idx_middle)).replace('<suffix>', str(idx_suffix))
                            list_of_keys.append(key)
        self.list_of_keys = list_of_keys

    def _from_key_to_words(self, key):
        """
            Expect a key of the form idx_{det1}-idx_{det2}-idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
        """
        list_of_idx = [int(idx) for idx in key.split('-')]
        det1 = self.dets_list[list_of_idx[0]]
        det2 = self.dets_list[list_of_idx[1]]
        prefix = self.structs_dict['prefix'][list_of_idx[2]]
        middle = self.structs_dict['middle'][list_of_idx[3]]
        suffix = self.structs_dict['suffix'][list_of_idx[4]]

        return [det1, det2, prefix, middle, suffix]

    def _create_prompt(self, dets, structs):
        
        det1, det2 = dets
        prefix, middle, suffix = structs

        sentence = self.vanilla_template.replace('<DET1>', det1).replace('<DET2>', det2)
        sentence = sentence.replace('<PREFIX>', prefix).replace('<MIDDLE>', middle).replace('<SUFFIX>', suffix)
        
        return sentence

    def create_prompts(self):
        """
            Returns : keys idx_{det1}-idx_{det2}-idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
                      value sentence
        """

        dict_of_prompts = {}
        for key in self.list_of_keys:
            words_from_keys = self._from_key_to_words(key)
            dets, structs = words_from_keys[0:2], words_from_keys[2:5]
            sentence = self._create_prompt(dets, structs)
            dict_of_prompts[key] = sentence

        self.dict_of_prompts = dict_of_prompts


    def compute_all_pairs_scores(self, list_of_words):
        """
            expect words = list of pairs [HYPONYM, NOUN]
            returns : dict -> key "HYPONYM---NOUN"
                              value dict -> key transf
                                            value dict -> keys idx_{det1}-idx_{det2}-idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
                                                          value [score_mask1, score_mask2]
        """

        # Compute Prompts Scores
        if os.path.exists(self.filename): # Previous save
            savefile = open(self.filename, 'rb')
            all_pairs_scores_dict = pickle.load(savefile)
            savefile.close()
        else:
            all_pairs_scores_dict = {}
        num_treated = 0
        for words in tqdm.tqdm(list_of_words, total = len(list_of_words)):
            word1, word2 = words
            key = word1 + '---' + word2
            if key in all_pairs_scores_dict.keys(): #If we have already computed this key go to the next
                continue
            scores_dict = self.batch_compute_one_pair_scores(words)
            all_pairs_scores_dict[key] = scores_dict
            num_treated += 1
            if num_treated % 20000 == 0: #Save from time to time
                savefile = open(self.filename, 'wb')
                pickle.dump(all_pairs_scores_dict, savefile)
                savefile.close()
            
        self.all_pairs_scores_dict = all_pairs_scores_dict

        # Save scores
        savefile = open(self.filename, 'wb')
        pickle.dump(all_pairs_scores_dict, savefile)
        savefile.close()


    def compute_one_pair_scores(self, words):
        """
            expect words = [HYPONYM, NOUN]
            returns : dict -> key transf
                              value dict -> keys idx_{det1}-idx_{det2}-idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
                                            value [score_mask1, score_mask2]
        """
        # Tokenize the words to know the number of masks to add
        word1, word2 = words
        masked_token_ids_1 = self.tokenizer(word1)['input_ids'][1:-1]
        masked_token_ids_2 = self.tokenizer(word2)['input_ids'][1:-1]

        N_masks_1 = len(masked_token_ids_1)
        N_masks_2 = len(masked_token_ids_2)

        # Construct sentences
        scores_dict = {}
        for transf in self.transformations_names:
            transf_score_dict = {}
            for key in self.list_of_keys:
                vanilla_sentence = self.dict_of_prompts[key]
                sentence, mask1_rank, mask2_rank = self.phi(vanilla_sentence, transf, N_masks_1, N_masks_2)
                # Compute input_ids and attention_mask of the sentence
                encoding = self.tokenizer(sentence,
                            return_tensors='pt'
                            )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                # The model needs the masks_to_predict_pos
                masks_to_predict_pos = self.find_masks_pos(input_ids)
                score_mask1 = self._compute_model_score(input_ids, attention_mask, masked_token_ids_1, masks_to_predict_pos[mask1_rank - 1])
                score_mask2 = self._compute_model_score(input_ids, attention_mask, masked_token_ids_2, masks_to_predict_pos[mask2_rank - 1])
                transf_score_dict[key] = [score_mask1, score_mask2]
            scores_dict[transf] = transf_score_dict

        return scores_dict

    def _compute_model_score(self, input_ids, attention_mask, masked_token_ids, masks_to_predict_pos):

        # Compute the probabilities and ranks from the model
        with torch.no_grad():
            probs_n_ranks = self.model.compute_greedy(input_ids, attention_mask, masks_to_predict_pos, masked_token_ids)

        # Compute scores
        score = probs_n_ranks[:,0].prod()

        return score


    def batch_compute_one_pair_scores(self, words):
        """
            expect words = [HYPONYM, NOUN]
            returns : dict -> key transf
                              value dict -> keys idx_{det1}-idx_{det2}-idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
                                            value [score_mask1, score_mask2]
        """
        # Tokenize the words to know the number of masks to add
        word1, word2 = words
        masked_token_ids_1 = self.tokenizer(word1, return_tensors='pt')['input_ids'][:,1:-1].repeat(len(self.list_of_keys),1).to(self.device)
        masked_token_ids_2 = self.tokenizer(word2, return_tensors='pt')['input_ids'][:,1:-1].repeat(len(self.list_of_keys),1).to(self.device)

        N_masks_1 = masked_token_ids_1.shape[1]
        N_masks_2 = masked_token_ids_2.shape[1]

        # Construct sentences
        scores_dict = {}
        for transf in self.transformations_names:
            transf_score_dict = {}
            sentences = []
            mask1_ranks, mask2_ranks = [], []
            for key in self.list_of_keys:
                vanilla_sentence = self.dict_of_prompts[key]
                sentence, mask1_rank, mask2_rank = self.phi(vanilla_sentence, transf, N_masks_1, N_masks_2)
                sentences.append(sentence)
                mask1_ranks.append(mask1_rank)
                mask2_ranks.append(mask2_rank)
            # Compute input_ids and attention_mask of the sentence
            encoding = self.tokenizer(sentences,
                            padding = True,
                            return_tensors='pt'
                            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            # The model needs the masks_to_predict_pos
            masks_to_predict_pos = self.batch_find_masks_pos(input_ids) # We suppose this is ok
            scores_mask1 = self._batch_compute_model_score(input_ids, attention_mask, masked_token_ids_1, self.helper(masks_to_predict_pos, mask1_ranks).to(self.device))
            scores_mask2 = self._batch_compute_model_score(input_ids, attention_mask, masked_token_ids_2, self.helper(masks_to_predict_pos, mask2_ranks).to(self.device))
            for idx in range(len(self.list_of_keys)):
                key = self.list_of_keys[idx]
                transf_score_dict[key] = [scores_mask1[idx].item(), scores_mask2[idx].item()]
            scores_dict[transf] = transf_score_dict

        return scores_dict

    def _batch_compute_model_score(self, input_ids, attention_mask, masked_token_ids, masks_to_predict_pos):

        # Compute the probabilities and ranks from the model
        with torch.no_grad():
            probs = self.model.batch_compute_greedy(input_ids, attention_mask, masks_to_predict_pos, masked_token_ids)

        # Compute scores
        scores = probs.prod(dim=1) # shape [batch_size = len(self.list_of_keys)]

        return scores


    def batch_find_masks_pos(self, ids_seq):
        masks_pos = torch.where(ids_seq == 103)[1]
        pos_clusters = []
        cluster = []
        for k in range(masks_pos.shape[0]):
            cluster.append(masks_pos[k])
            if (k < len(masks_pos) -1) and (masks_pos[k] + 1 != masks_pos[k + 1]): #The next mask pos does not follow the previous one
                pos_clusters.append(torch.LongTensor(cluster))
                cluster = []
        pos_clusters.append(torch.LongTensor(cluster))
        return pos_clusters

    def helper(self, list_of_tensors, mask_rank):
        batch_size = len(self.list_of_keys)
        mask_pos = []
        for k in range(batch_size):
            mask_pos.append(list_of_tensors[2*k:2*k+2][mask_rank[k] - 1])
        return torch.cat(mask_pos)


    def find_masks_pos(self, ids_seq):
        """
            Compute all mask_token positions in the sequence, then divide it into clusters (following sequence) and returns the mask_rank^th cluster.
        """
        def find_all_masks_pos(ids_seq):
            pos = []
            for k in range(ids_seq.shape[1]):
                if ids_seq[0][k] == 103:
                    pos.append(k)
            return pos
        
        all_masks_pos = find_all_masks_pos(ids_seq)
        
        pos_clusters = []
        cluster = []
        for k in range(len(all_masks_pos)):
            cluster.append(all_masks_pos[k])
            if (k < len(all_masks_pos) -1) and (all_masks_pos[k] + 1 != all_masks_pos[k + 1]): #The next mask pos does not follow the previous one
                pos_clusters.append(cluster)
                cluster = []
        pos_clusters.append(cluster)
        
        return pos_clusters

    def phi(self, vanilla_sentence, transf, N_masks_1, N_masks_2):
        """
            Take a sentence s and returns phi(s) and the rank of mask1 (cf. google doc.)
            The template vanilla is something like : "MASK1 is MASK2" thus MASK1 is rank 1 and MASK2 is rank 2
            Whereas for the transformation opposite : "MASK2 is MASK1" thus MASK1 is rank 2 and MASK2 is rank 1
        """

        if transf == 'vanilla':
            sentence = vanilla_sentence.replace('<WORD1>', N_masks_1*self.tokenizer.mask_token).replace('<WORD2>', N_masks_2*self.tokenizer.mask_token)
            mask1_rank, mask2_rank = 1, 2
        elif transf == 'opposite':
            sentence = vanilla_sentence.replace('<WORD1>', N_masks_2*self.tokenizer.mask_token).replace('<WORD2>', N_masks_1*self.tokenizer.mask_token)
            mask1_rank, mask2_rank = 2, 1
        elif transf == 'reverse':
            sentence = vanilla_sentence.replace('<WORD1>', N_masks_2*self.tokenizer.mask_token).replace('<WORD2>', N_masks_1*self.tokenizer.mask_token)
            mask1_rank, mask2_rank = 2, 1

        return sentence, mask1_rank, mask2_rank
        
        

#######################################
#                                     #
#                LOGICAL              #
#                                     #
#######################################


class LogicalPromptScorer:

    def __init__(self, model = None, tokenizer = None, device = None, dataset_name = ''):
        # Model used to compute scores
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load prompts materials
        self.dets_list = DETS_LIST
        self.structs_dict = {'prefixes': LOGICAL_PREFIXES_LIST,
                             'struct_lw': LOGICAL_STRUCTS_LW_LIST}

        # Define template
        self.vanilla_template = '<PREFIX1> <DET1> <WORD1> <STRUCT_LW> <LW> <PREFIX2> <DET2> <WORD2>.'
        self.key_template = '<det1>-<det2>-<prefixes>-<struct_lw>'

        # Compute keys
        self._compute_keys()

        # Where to save data
        self.filename = 'prompts\\scores\\logical_prompts_scores_{}'.format(dataset_name)

        # Compute Prompts
        self.create_prompts()
        
    def _compute_keys(self):
        """
            Compute all the possible keys in the form idx_{det1}-idx_{det2}-idx_{prefixes}-idx_{struct_lw}
        """
        N_dets = len(self.dets_list)
        N_prefixes = len(self.structs_dict['prefixes'])
        N_struct_lw = len(self.structs_dict['struct_lw'])

        list_of_keys = []
        for idx_det1 in range(N_dets):
            for idx_det2 in range(N_dets):
                for idx_prefixes in range(N_prefixes):
                        for idx_struct_lw in range(N_struct_lw):
                            key = self.key_template.replace('<det1>', str(idx_det1)).replace('<det2>', str(idx_det2))
                            key = key.replace('<prefixes>', str(idx_prefixes)).replace('<struct_lw>', str(idx_struct_lw))
                            list_of_keys.append(key)
        self.list_of_keys = list_of_keys

    def _from_key_to_words(self, key):
        """
            Expect a key of the form idx_{det1}-idx_{det2}-idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
        """
        list_of_idx = [int(idx) for idx in key.split('-')]
        det1 = self.dets_list[list_of_idx[0]]
        det2 = self.dets_list[list_of_idx[1]]
        prefixes = self.structs_dict['prefixes'][list_of_idx[2]]
        struct_lw = self.structs_dict['struct_lw'][list_of_idx[3]]

        return [det1, det2, prefixes, struct_lw]

    def _create_prompt(self, dets, prefixes, struct_lw):
        
        det1, det2 = dets
        prefix1, prefix2 = prefixes
        # Sentence in the right order "This is a seagull, therefore it is a bird."
        sentence = self.vanilla_template.replace('<DET1>', det1).replace('<DET2>', det2)
        sentence = sentence.replace('<PREFIX1>', prefix1).replace('<PREFIX2>', prefix2).replace('<STRUCT_LW>', struct_lw)
        # Sentence in the reverse order "It is a bird, therefore this is a seagull."
        sentence_reverse = self.vanilla_template.replace('<DET1>', det2).replace('<DET2>', det1)
        sentence_reverse = sentence_reverse.replace('<PREFIX1>', prefix2).replace('<PREFIX2>', prefix1).replace('<STRUCT_LW>', struct_lw)
        
        return sentence, sentence_reverse

    def create_prompts(self):
        """
            Returns : keys idx_{det1}-idx_{det2}-idx_{prefixes}-idx_{struct_lw}
                      value [sentence, sentence_reverse]
        """

        dict_of_prompts = {}
        for key in self.list_of_keys:
            words_from_keys = self._from_key_to_words(key)
            dets, prefixes, struct_lw = words_from_keys[0:2], words_from_keys[2], words_from_keys[3]
            sentence, sentence_reverse = self._create_prompt(dets, prefixes, struct_lw)
            dict_of_prompts[key] = [sentence, sentence_reverse]

        self.dict_of_prompts = dict_of_prompts


    def compute_all_pairs_scores(self, logical_words, list_of_words):
        """
            expect words = list of pairs [HYPONYM, NOUN]
            returns : dict -> key "HYPONYM---NOUN"
                              value dict -> keys idx_{det1}-idx_{det2}-idx_{prefixes}-idx_{struct_lw}
                                            value [[score_lw for lw in logical_words], [score_reverse_lw for lw in logical_words]]
        """
        # Tokenize the logical words
        logical_words_ids = []
        for lw in logical_words:
            input_ids = self.tokenizer(lw)['input_ids'][1:-1]
            assert len(input_ids) == 1 # We only keep logical words mapped to a single token
            logical_words_ids.append(input_ids[0])

        # Compute Prompts Scores
        if os.path.exists(self.filename): # Previous save
            savefile = open(self.filename, 'rb')
            all_pairs_scores_dict = pickle.load(savefile)
            savefile.close()
        else:
            all_pairs_scores_dict = {}
        num_treated = 0
        for words in tqdm.tqdm(list_of_words, total = len(list_of_words)):
            word1, word2 = words
            key = word1 + '---' + word2
            if key in all_pairs_scores_dict.keys(): # If we have already computed this key go to the next
                continue
            scores_dict = self.batch_compute_one_pair_scores(logical_words_ids, words)
            all_pairs_scores_dict[key] = scores_dict
            num_treated += 1
            if num_treated % 20000 == 0: # Save from time to time
                savefile = open(self.filename, 'wb')
                pickle.dump(all_pairs_scores_dict, savefile)
                savefile.close()
            
        self.all_pairs_scores_dict = all_pairs_scores_dict

        # Save scores
        savefile = open(self.filename, 'wb')
        pickle.dump(all_pairs_scores_dict, savefile)
        savefile.close()


    def compute_one_pair_scores(self, logical_words_ids, words):
        """
            expect words = [HYPONYM, NOUN]
            returns : dict -> keys idx_{det1}-idx_{det2}-idx_{prefixes}-idx_{struct_lw}
                              value [[score_lw for lw in logical_words], [score_reverse_lw for lw in logical_words]]
        """

        word1, word2 = words

        # Construct sentences
        scores_dict = {}
        for key in self.list_of_keys:
            sentence, sentence_reverse = self.dict_of_prompts[key]
            sentence = sentence.replace('<WORD1>', word1).replace('<WORD2>', word2).replace('<LW>', self.tokenizer.mask_token)
            sentence_reverse = sentence_reverse.replace('<WORD1>', word2).replace('<WORD2>', word1).replace('<LW>', self.tokenizer.mask_token)
            # Compute scores for sentence
            encoding = self.tokenizer(sentence,
                        return_tensors='pt'
                        )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            mask_pos = self.find_mask_pos(input_ids)
            scores = self._compute_model_score(input_ids, attention_mask, logical_words_ids, mask_pos)
            
            # Compute scores for sentence_reverse
            encoding_reverse = self.tokenizer(sentence_reverse,
                        return_tensors='pt'
                        )
            input_ids_reverse = encoding_reverse['input_ids'].to(self.device)
            attention_mask_reverse = encoding_reverse['attention_mask'].to(self.device)
            mask_pos_reverse = self.find_mask_pos(input_ids_reverse)
            scores_reverse = self._compute_model_score(input_ids_reverse, attention_mask_reverse, logical_words_ids, mask_pos_reverse)
            
            scores_dict[key] = [scores, scores_reverse]

        return scores_dict

    def batch_compute_one_pair_scores(self, logical_words_ids, words):
        """
            expect words = [HYPONYM, NOUN]
            returns : dict -> keys idx_{det1}-idx_{det2}-idx_{prefixes}-idx_{struct_lw}
                              value [[score_lw for lw in logical_words], [score_reverse_lw for lw in logical_words]]
        """

        word1, word2 = words

        # Construct sentences
        scores_dict = {}
        sentences = []
        for key in self.list_of_keys:
            sentence, sentence_reverse = self.dict_of_prompts[key]
            sentence = sentence.replace('<WORD1>', word1).replace('<WORD2>', word2).replace('<LW>', self.tokenizer.mask_token)
            sentence_reverse = sentence_reverse.replace('<WORD1>', word2).replace('<WORD2>', word1).replace('<LW>', self.tokenizer.mask_token)
            sentences.append(sentence)
            sentences.append(sentence_reverse)
        # Compute scores for sentence
        encoding = self.tokenizer(sentences,
                                  padding = True,
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        mask_pos = self.find_mask_pos(input_ids)
        scores = self._batch_compute_model_score(input_ids, attention_mask, logical_words_ids, mask_pos)

        for k in range(len(self.list_of_keys)):
            key = self.list_of_keys[k]
            scores_dict[key] = [scores[2*k], scores[2*k + 1]]

        return scores_dict

    def _compute_model_score(self, input_ids, attention_mask, masked_token_ids, mask_pos):

        # Compute the probabilities and ranks from the model
        with torch.no_grad():
            probs_n_ranks = self.model.compute_multiple_mono_token(input_ids, attention_mask, mask_pos, masked_token_ids)

        # Compute scores
        scores = probs_n_ranks[:,0] # drop rank

        return scores

    def _batch_compute_model_score(self, input_ids, attention_mask, masked_token_ids, mask_pos):

        # Compute the probabilities and ranks from the model
        with torch.no_grad():
            probs = self.model.compute_batch_multiple_mono_token(input_ids, attention_mask, mask_pos, masked_token_ids)

        return probs


    def find_mask_pos(self, ids_seq):
        return torch.where(ids_seq == 103)[1]
    