import numpy as np
import matplotlib.pyplot as plt
import pickle



class Plotter:

    def __init__(self, content_filename, logical_filename, n, dataset):
        self.content_filename = content_filename
        self.logical_filename = logical_filename
        self.n = n
        self.dataset = dataset

    def content_plot(self):
        #filename = 'content\\scores\\content_scores_wordnet'
        savefile = open(self.content_filename, 'rb')
        dict_of_scores = pickle.load(savefile)
        savefile.close()

        list_of_keys = list(dict_of_scores.keys())
        list_of_values = list(dict_of_scores.values())

        transformations = list(list_of_values[0].keys())
        N_transformations = len(transformations)

        transf_scores = {k: [] for k in transformations}
        for elem in list_of_values:
            for transf in transformations:
                transf_scores[transf].append(elem[transf])

        fig, axs = plt.subplots(1, N_transformations, figsize = (8*N_transformations, 8))
        colors = ['b','g','r','c','m','y']

        for j in range(N_transformations):
            data = np.array(transf_scores[transformations[j]])
            b, bins, patches = axs[j].hist(data, 200, density=False, facecolor=colors[j], alpha = 0.75, label = 'MEAN : {}'.format(np.round(data.mean(), decimals = 2)))
            axs[j].axvline(data.mean(), linestyle='--', color='grey', linewidth=2*0.8)
            axs[j].set_xlabel('scores')
            axs[j].set_ylabel('prop')
            axs[j].set_title('vanilla vs {}'.format(transformations[j]))
            axs[j].legend()

        plt.savefig('plot/figures/content_plot_' + self.dataset + '.png')
        plt.show()

    def logical_plot(self):
        savefile = open(self.logical_filename, 'rb')
        dict_of_scores = pickle.load(savefile)
        savefile.close()

        # Get best and worst pairs (for content metric)
        transf_worst_keys, transf_best_keys = self.compute_n_best_worst()
        transf_worst = self.compute_values_from_keys(transf_worst_keys)
        transf_best = self.compute_values_from_keys(transf_best_keys)
        transformations = list(transf_best.keys())

        # Get logical data 
        list_of_pairs = list(dict_of_scores.keys())
        N_data = 1 + 2*len(transformations)
        
        logical_words = list(dict_of_scores[list_of_pairs[0]].keys())
        N_logical_words = len(logical_words)

        processed_data = self.prepare_data(dict_of_scores)

        fig, axs = plt.subplots(1 , 1, figsize = (21, 7))

        ind = np.arange(N_logical_words)
        # We plot all pairs
        logical_words_values = [np.array(processed_data[lw]).mean() for lw in logical_words]
        axs.bar(ind + 0/(N_data+1), logical_words_values, width = 1/(N_data+1))
        # We plot best pairs
        for k in range(len(transformations)):
            logical_words_values = [np.array(transf_best[transformations[k]][lw]).mean() for lw in logical_words]
            axs.bar(ind + (1 + k)/(N_data+1), logical_words_values, width = 1/(N_data+1))
        # We plot worst pairs
        for k in range(len(transformations)):
            logical_words_values = [np.array(transf_worst[transformations[k]][lw]).mean() for lw in logical_words]
            axs.bar(ind + (1 + len(transformations) + k)/(N_data+1), logical_words_values, width = 1/(N_data+1))

        axs.set_xticks(ind+ 0.5*(N_data - 1)/(N_data+1))
        axs.set_xticklabels(logical_words)
        axs.set_ylabel('Scores')
        axs.set_title('Logical words scores {}'.format(self.dataset))
        axs.legend(labels = ['all'] + ['{} best {}'.format(self.n, transf) for transf in transformations] + ['{} worst {}'.format(self.n, transf) for transf in transformations])
        axs.axhline(0, color='grey', linewidth=0.8)

        axs.axvline(ind[8] - 1/(N_data+1), color='black', linewidth=3, linestyle = '--')

        plt.savefig('plot/figures/logic_plot_' + self.dataset + '_n' + str(self.n) + '.png')
        plt.show()


    def prepare_data(self, dict_of_scores):
        """
        Args:
        dict -> key 'HYPONYM---NOUN'
                value dict -> key lw
                              value score

        Returns:
        dict -> key lw
                value [scores for pair]
        """

        list_of_pairs = list(dict_of_scores.keys())
        logical_words = list(dict_of_scores[list_of_pairs[0]].keys())

        processed_data = {k: [] for k in logical_words}
        for k in range(len(list_of_pairs)):
            for lw in logical_words:
                processed_data[lw].append(dict_of_scores[list_of_pairs[k]][lw])

        return processed_data

    def compute_n_best_worst(self):
        savefile = open(self.content_filename, 'rb')
        dict_of_scores = pickle.load(savefile)
        savefile.close()

        list_of_keys = list(dict_of_scores.keys())
        list_of_values = list(dict_of_scores.values())

        transformations = list(list_of_values[0].keys())
        N_transformations = len(transformations)

        transf_scores = {k: [] for k in transformations}
        for elem in list_of_values:
            for transf in transformations:
                transf_scores[transf].append(elem[transf])

        transf_worst = {k: [] for k in transformations}
        transf_best = {k: [] for k in transformations}
        for transf in transformations:
            data = np.array(transf_scores[transf])
            sort_idx = np.argsort(data)
            n_last_idx = sort_idx[:self.n]
            n_first_idx  = sort_idx[-self.n:]
            for k in range(self.n):
                transf_worst[transf].append(list_of_keys[n_last_idx[k]])
                transf_best[transf].append(list_of_keys[n_first_idx[k]])
        return transf_worst, transf_best

    def compute_values_from_keys(self, keys):
        """
        Returns: dict -> key transf =/= vanilla
                         value dict -> key lw
                                       value [scores for pair]
        """
        savefile = open(self.logical_filename, 'rb')
        dict_of_scores = pickle.load(savefile)
        savefile.close()

        list_of_pairs = list(dict_of_scores.keys())
        logical_words = list(dict_of_scores[list_of_pairs[0]].keys())

        transformations = list(keys.keys())

        transf_scores = {}
        for transf in transformations:
            lw_scores = {k: [] for k in logical_words} # Each list will be size n
            for key in keys[transf]:
                scores = dict_of_scores[key]
                for lw in logical_words:
                    lw_scores[lw].append(scores[lw])
            transf_scores[transf] = lw_scores

        return transf_scores




        

        
        
