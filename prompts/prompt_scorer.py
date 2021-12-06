
import numpy as np
import pickle
import tqdm

from prompts.prompt_material import DETS_LIST, STRUCTS_PREFIX_LIST, STRUCTS_MIDDLE_LIST, STRUCTS_SUFFIX_LIST, TRANSFORMATIONS


class PromptScorer:

    def __init__(self, model = None, tokenizer = None, device = None, dataset_name = ''):
        # Model used to compute scores
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load prompts materials
        self.dets_list = DETS_LIST
        self.structs_dict = {'prefix': STRUCTS_PREFIX_LIST,
                             'middle': STRUCTS_MIDDLE_LIST,
                             'suffix': STRUCTS_SUFFIX_LIST}

        # Load transformations names
        self.transformations_names = TRANSFORMATIONS

        # Define template
        self.vanilla_template = '<PREFIX> <DET1> <WORD1> <MIDDLE> <DET2> <WORD2> <SUFFIX>.'
        self.key_template = '<det1>-<det2>-<prefix>-<middle>-<suffix>'

        # Compute keys
        self._compute_keys()

        # Where to save data
        self.filename = 'prompts\\scores\\prompts_scores_{}'.format(dataset_name)

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
        all_pairs_scores_dict = {}
        for words in tqdm.tqdm(list_of_words, total = len(list_of_words)):
            word1, word2 = words
            key = word1 + '---' + word2
            scores_dict = self.compute_one_pair_scores(words)
            all_pairs_scores_dict[key] = scores_dict

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
                score_mask1 = self._compute_model_score(sentence, masked_token_ids_1, mask1_rank)
                score_mask2 = self._compute_model_score(sentence, masked_token_ids_2, mask2_rank)
                transf_score_dict[key] = [score_mask1, score_mask2]
            scores_dict[transf] = transf_score_dict

        return scores_dict

    def _compute_model_score(self, sentence, masked_token_ids, mask_rank):

        # Compute input_ids and attention_mask of the sentence
        encoding = self.tokenizer(sentence,
                     max_length=64, 
                     padding='max_length',
                     return_tensors='pt'
                     )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # The model needs the masks_to_predict_pos
        masks_to_predict_pos = self.find_masks_pos(input_ids, mask_rank)

        # Compute the probabilities and ranks from the model
        probs_n_ranks = self.model.compute_greedy(input_ids, attention_mask, masks_to_predict_pos, masked_token_ids)

        # Compute scores
        score = probs_n_ranks[:,0].prod()

        return score


    def find_masks_pos(self, ids_seq, mask_rank):
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
        
        return pos_clusters[mask_rank - 1]

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
        
        
