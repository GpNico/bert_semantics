
import numpy as np
import pickle
import tqdm

from wordfreq import word_frequency


class ElFiltrator:

    def __init__(self, dataset_name = '', filtration_type = ''):
        self.prop = 0.5
        self.filtration_type = filtration_type
        # to save 
        if filtration_type == '':
            self.filename = 'content\\scores\\content_scores_{}'.format(dataset_name)
            self.logical_filename = 'logical\\scores\\logical_scores_{}'.format(dataset_name)
        else:
            self.filename = 'content\\scores\\content_scores_{}_{}'.format(dataset_name, filtration_type)
            self.logical_filename = 'logical\\scores\\logical_scores_{}_{}'.format(dataset_name, filtration_type)
            
        # To load
        self.prompts_filename = 'prompts\\best\\content_best_prompts_{}'.format(dataset_name)
        self.keys_filename = 'prompts\\best\\content_best_keys_{}'.format(dataset_name)
        self.prompts_scores_filename = 'prompts\\scores\\content_prompts_scores_{}'.format(dataset_name)
        self.content_scores_filename = 'content\\scores\\content_scores_{}'.format(dataset_name)
        self.logical_scores_filename = 'logical\\scores\\logical_scores_{}'.format(dataset_name)

    def load_content_scores(self, filename):
        savefile = open(filename, 'rb')
        self.content_scores_dict = pickle.load(savefile)
        savefile.close()

    def load_logical_scores(self, filename):
        savefile = open(filename, 'rb')
        self.logical_scores_dict = pickle.load(savefile)
        savefile.close()

    def load_best_prompts(self, filename):
        savefile = open(filename, 'rb')
        self.best_prompts_dict = pickle.load(savefile)
        savefile.close()

    def load_best_keys(self, filename):
        savefile = open(filename, 'rb')
        self.best_keys_dict = pickle.load(savefile)
        savefile.close()

    def load_prompts_scores(self, filename):
        savefile = open(filename, 'rb')
        self.prompts_scores_dict = pickle.load(savefile)
        savefile.close()


    def construct_filtered_scores(self, filtered_keys):
        """
            Arg:
                filtered_keys : list of keys like 'judo---sport'
            Returns:
                dict -> key from filtered_keys
                        value content_scores_dict[key]
        """
        filtered_content_scores = {}
        for key in filtered_keys:
            filtered_content_scores[key] = self.content_scores_dict[key]

        return filtered_content_scores

    def filtrate(self):
        """
            pass
        """

        self.load_content_scores(self.content_scores_filename)

        if self.filtration_type == '':
            return
        elif self.filtration_type == 'model_freq':
            filtered_keys = self.model_freq_filtration()
        elif self.filtration_type == 'word_freq':
            filtered_keys = self.word_freq_filtration()
        else:
            raise Exception('This filtration type is not implemented. Please try model_freq or word_freq.')

        filtered_content_scores = self.construct_filtered_scores(filtered_keys)

        # Save scores
        savefile = open(self.filename, 'wb')
        pickle.dump(filtered_content_scores, savefile)
        savefile.close()

    def filtrate_logical(self):
        """
            We load the filtrate content scores and just output the logical scores for those pairs.
        """
        if self.filtration_type == '':
            return
        # Load filtered content scores
        savefile = open(self.filename, 'rb')
        filtered_content_scores = pickle.load(savefile)
        savefile.close()
        # Content filtered keys
        content_filtered_keys = list(filtered_content_scores.keys())
        # Load logical scores
        self.load_logical_scores(self.logical_scores_filename)
        # Go
        filtered_logical_scores = {}
        for pair_key in tqdm.tqdm(content_filtered_keys, total = len(content_filtered_keys)):
            filtered_logical_scores[pair_key] = self.logical_scores_dict[pair_key]

        print("{} pairs kept!".format(len(content_filtered_keys)))
        # Save scores
        savefile = open(self.logical_filename, 'wb')
        pickle.dump(filtered_logical_scores, savefile)
        savefile.close()


    def model_freq_filtration(self):
        """
            Filtrate according to the score given by :
                min_{transf} ( min( P[MASK2 = word2 | S*(transf)(MASK1, MASK2)], P[MASK1 = word1 | S*(transf)(MASK1, MASK2)] ) )
            where S*(transf) is the optimal vanilla template for the transformation transf
        """
        self.load_best_keys(self.keys_filename)
        self.load_prompts_scores(self.prompts_scores_filename)

        filtration_scores = {} # dict -> key 'judo---sport' ; value filtration score

        for pair_key in tqdm.tqdm(self.best_keys_dict.keys(), total = len(list(self.best_keys_dict.keys()))):
            transf_scores = []
            transf_best_keys = self.best_keys_dict[pair_key]
            for transf in transf_best_keys.keys():
                vanilla_key = transf_best_keys[transf][0]
                transf_scores.append( min(self.prompts_scores_dict[pair_key]['vanilla'][vanilla_key]) )
            filtration_scores[pair_key] = min(transf_scores)

        values = np.array(list(filtration_scores.values()))
        keys = list(filtration_scores.keys())

        sorted_idx = np.argsort(values)

        filtered_keys = []
        for key_idx in sorted_idx[int(len(keys)*self.prop):]:
            filtered_keys.append(keys[key_idx])

        print("{} pairs kept!".format(len(filtered_keys)))
        return filtered_keys


    def word_freq_filtration(self):

        self.load_best_keys(self.keys_filename)
        list_of_pair_keys = list(self.best_keys_dict.keys())

        filtration_scores = {}

        for pair_key in tqdm.tqdm(list_of_pair_keys, total = len(list_of_pair_keys)):
            word1, word2 = pair_key.split('---')
            filtration_scores[pair_key] = min(self.compute_expression_freq(word1), self.compute_expression_freq(word2))

        values = np.array(list(filtration_scores.values()))
        keys = list(filtration_scores.keys())

        sorted_idx = np.argsort(values)

        filtered_keys = []
        for key_idx in sorted_idx[int(len(keys)*self.prop):]:
            filtered_keys.append(keys[key_idx])

        print("{} pairs kept!".format(len(filtered_keys)))

        return filtered_keys


    def compute_expression_freq(self, expression):
        list_of_words = expression.split(' ') #The expression can be composed of multiple words
        freq_list = []
        for word in list_of_words:
            freq_list.append(word_frequency(word, 'en'))
        return np.array(freq_list).mean()


   
        

        
        
