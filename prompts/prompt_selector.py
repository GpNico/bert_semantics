
import numpy as np
import pickle
import tqdm

from prompts.prompt_material import DETS_LIST, CONTENT_STRUCTS_PREFIX_LIST, CONTENT_STRUCTS_MIDDLE_LIST, CONTENT_STRUCTS_SUFFIX_LIST, TRANSFORMATIONS, LOGICAL_PREFIXES_LIST, LOGICAL_STRUCTS_LW_LIST

#######################################
#                                     #
#               CONTENT               #
#                                     #
#######################################


class PromptSelector:

    def __init__(self, dict_of_prompts, dataset_name = ''):
        # Load prompts materials
        self.dets_list = DETS_LIST
        self.structs_dict = {'prefix': CONTENT_STRUCTS_PREFIX_LIST,
                             'middle': CONTENT_STRUCTS_MIDDLE_LIST,
                             'suffix': CONTENT_STRUCTS_SUFFIX_LIST}

        # Load transformations names
        self.transformations_names = TRANSFORMATIONS

        # Get the dict of prompts from PromptScorer
        self.dict_of_prompts = dict_of_prompts

        # Compute Keys
        self._compute_struct_keys()
        self._compute_det_keys()

        # To save
        self.filename = 'prompts\\best\\content_best_prompts_{}'.format(dataset_name)

    def load_scores(self, filename):
        savefile = open(filename, 'rb')
        self.all_pairs_scores_dict = pickle.load(savefile)
        savefile.close()

    def _compute_struct_keys(self):
        """
            Compute all the possible keys in the form idx_{struct_prefix}-idx_{struct_middle}-idx_{struct_suffix}
        """
        N_prefix = len(self.structs_dict['prefix'])
        N_middle = len(self.structs_dict['middle'])
        N_suffix = len(self.structs_dict['suffix'])

        list_of_keys = []
        for idx_prefix in range(N_prefix):
            for idx_middle in range(N_middle):
                for idx_suffix in range(N_suffix):
                    key = '<prefix>-<middle>-<suffix>'
                    key = key.replace('<prefix>', str(idx_prefix)).replace('<middle>', str(idx_middle)).replace('<suffix>', str(idx_suffix))
                    list_of_keys.append(key)
        self.struct_keys = list_of_keys

    def _compute_det_keys(self):
        """
            Compute all the possible keys in the form idx_{det1}-idx_{det2}
        """
        N_dets = len(self.dets_list)

        list_of_keys = []
        for idx_det1 in range(N_dets):
            for idx_det2 in range(N_dets):
                key = '<det1>-<det2>'
                key = key.replace('<det1>', str(idx_det1)).replace('<det2>', str(idx_det2))
                list_of_keys.append(key)
        self.det_keys = list_of_keys



    def compute_best_prompts(self, filename):
        """
            returns : dict -> key "HYPONYM---NOUN"
                              value dict -> key transf =/= vanilla
                                            value [S*, S*_phi]
            Ex : S* = 'an <WORD1> is a <WORD2>'
                 S*_phi = 'a <WORD1> is an <WORD2>'
        """
        self.load_scores(filename)

        list_of_pairs = list(self.all_pairs_scores_dict.keys())

        best_prompts = {}
        for pair in tqdm.tqdm(list_of_pairs, total = len(list_of_pairs)):
            pair_best_prompts = {}
            scores_dict = self.all_pairs_scores_dict[pair]
            for transf in self.transformations_names:
                if transf == 'vanilla':
                    continue
                best_struct_key, vanilla_best_det_key, transf_best_det_key = self._compute_best_prompt_transf(scores_dict, transf)
                vanilla_best_key = vanilla_best_det_key + '-' + best_struct_key
                transf_best_key = transf_best_det_key + '-' + best_struct_key

                S_opt = self.dict_of_prompts[vanilla_best_key]
                S_opt_phi = self.dict_of_prompts[transf_best_key]
                
                pair_best_prompts[transf] = [S_opt, S_opt_phi]
            best_prompts[pair] = pair_best_prompts

            # Save prompts
            savefile = open(self.filename, 'wb')
            pickle.dump(best_prompts, savefile)
            savefile.close()



    def _compute_best_prompt_transf(self, scores_dict, transf):
        """
            Compute the best prompt for each transformations =/= vanilla according to :
            STRUCT* = argmax_{STRUCT}(
                                    max_{DET1, DET2} P(MASK1=eagle|S(STRUCT)(DET1, DET2)) x P(MASK2=bird|S(STRUCT)(DET1, DET2))
                                            x 
                                    max_{DET1, DET2} P(MASK1=eagle|S_phi(STRUCT))(DET1, DET2)) x P(MASK2=bird|S_phi(STRUCT)(DET1, DET2))
                                    )


            Returns : [best_struct_key, vanilla_best_det_key, transf_best_det_key]
        """

        # Get prompt scores
        vanilla_prompt_scores = scores_dict['vanilla']
        transf_prompt_scores = scores_dict[transf]

        # Compute the dict -> key struct_keys
        #                     value dict -> key det_keys
        #                                   value scores_dict[transf][det_key + '-' + struct_key][0]*scores_dict[transf][det_key + '-' + struct_key][1]
        structured_vanilla_scores_dict = {}
        structured_transf_scores_dict = {}
        for struct_key in self.struct_keys:
            temp_vanilla_dict = {}
            temp_transf_dict = {}
            for det_key in self.det_keys:
                temp_vanilla_dict[det_key] = vanilla_prompt_scores[det_key + '-' + struct_key][0]*vanilla_prompt_scores[det_key + '-' + struct_key][1]
                temp_transf_dict[det_key] = transf_prompt_scores[det_key + '-' + struct_key][0]*transf_prompt_scores[det_key + '-' + struct_key][1]
            structured_vanilla_scores_dict[struct_key] = temp_vanilla_dict
            structured_transf_scores_dict[struct_key] = temp_transf_dict

        # Compute argmax on det
        vanilla_det_argmax_dict = {}
        transf_det_argmax_dict = {}
        for struct_key in self.struct_keys:
            # Compute for each stuct_key max_{DET1, DET2} P(MASK1=eagle|S(STRUCT)(DET1, DET2)) x P(MASK2=bird|S(STRUCT)(DET1, DET2))
            vanilla_det_scores = np.array(list(structured_vanilla_scores_dict[struct_key].values()))
            vanilla_best_det_idx = np.argmax(vanilla_det_scores)
            vanilla_best_det_scores = vanilla_det_scores[vanilla_best_det_idx]
            vanilla_best_det = list(structured_vanilla_scores_dict[struct_key].keys())[vanilla_best_det_idx]
            vanilla_det_argmax_dict[struct_key] = [vanilla_best_det, vanilla_best_det_scores]

            # Compute for each stuct_key max_{DET1, DET2} P(MASK1=eagle|S_phi(STRUCT))(DET1, DET2)) x P(MASK2=bird|S_phi(STRUCT)(DET1, DET2))
            transf_det_scores = np.array(list(structured_transf_scores_dict[struct_key].values()))
            transf_best_det_idx = np.argmax(transf_det_scores)
            transf_best_det_scores = transf_det_scores[transf_best_det_idx]
            transf_best_det = list(structured_transf_scores_dict[struct_key].keys())[transf_best_det_idx]
            transf_det_argmax_dict[struct_key] = [transf_best_det, transf_best_det_scores]

        # Finally compute argmax on struct
        struct_scores = []
        for struct_key in self.struct_keys:
            struct_score = vanilla_det_argmax_dict[struct_key][1]*transf_det_argmax_dict[struct_key][1]
            struct_scores.append(struct_score)
        struct_scores = np.array(struct_scores)
        best_struct_idx = np.argmax(struct_scores)
        best_struct = self.struct_keys[best_struct_idx] # Best Struct
        vanilla_best_det_final = vanilla_det_argmax_dict[best_struct][0] # Best Det for vanilla
        transf_best_det_final = transf_det_argmax_dict[best_struct][0] # Best Det for transf

        return best_struct, vanilla_best_det_final, transf_best_det_final
        

        
        
#######################################
#                                     #
#                LOGICAL              #
#                                     #
#######################################

class LogicalPromptSelector:

    def __init__(self, dict_of_prompts, dataset_name = ''):
        # Load prompts materials
        self.dets_list = DETS_LIST
        self.structs_dict = {'prefixes': LOGICAL_PREFIXES_LIST,
                             'struct_lw': LOGICAL_STRUCTS_LW_LIST}


        # Get the dict of prompts from PromptScorer
        self.dict_of_prompts = dict_of_prompts

        # To save
        self.filename = 'prompts\\best\\logical_best_prompts_{}'.format(dataset_name)

    def load_scores(self, filename):
        savefile = open(filename, 'rb')
        self.all_pairs_scores_dict = pickle.load(savefile)
        savefile.close()

    def compute_best_prompts(self, filename, logical_words):
        """
            returns : dict -> key "HYPONYM---NOUN"
                              value dict -> [[S*(lw), S*_reverse(lw)] for lw in logical_words]
        """
        self.load_scores(filename)

        list_of_pairs = list(self.all_pairs_scores_dict.keys())

        best_prompts = {}
        for pair in tqdm.tqdm(list_of_pairs, total = len(list_of_pairs)):
            pair_best_prompts = []
            scores_dict = self.all_pairs_scores_dict[pair]

            scores = np.array(list(scores_dict.values()))
            keys = list(scores_dict.keys())

            for idx in range(len(logical_words)):
                lw_scores = scores[:,:, idx]

                best_idx  = self._compute_best_prompt_lw(lw_scores)
                best_key = keys[best_idx]

                S_opt = self.dict_of_prompts[best_key]
                
                pair_best_prompts.append(S_opt)
            best_prompts[pair] = pair_best_prompts

            # Save prompts
            savefile = open(self.filename, 'wb')
            pickle.dump(best_prompts, savefile)
            savefile.close()



    def _compute_best_prompt_lw(self, lw_scores):
        """
            Compute the best prompt for each transformations =/= vanilla according to :
            PREFIXES*, STRUCT_LW*, DET1*, DET2* = argmax_{PREFIXES, STRUCT_LW, DET1, DET2}(
                                     P(MASK=therefore|S(PREFIXES, STRUCT_LW, DET1, DET2)) 
                                            x 
                                     P(MASK=therefore|S_reverse(PREFIXES, STRUCT_LW, DET1, DET2)
                                    )


            Returns : best_key
        """
        # By the way lw_scores has been calculated we just need to compute :
        final_scores = lw_scores.prod(axis = 1)

        # Best scores
        best_idx = np.argmax(final_scores)

        return best_idx