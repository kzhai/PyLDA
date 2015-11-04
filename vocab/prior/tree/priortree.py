from glob import glob
#from topicmod.corpora.proto.wordnet_file_pb2 import *
from wordnet_file_pb2 import *
from collections import defaultdict
from copy import copy
from numpy import zeros

import codecs

class Node:
    def __init__(self):
        self._offset = -1

        self._raw_count = 0
        self._hypo_count = 0
        self._hyperparam_name = ''

        self._words_count = []
        self._words = []
        self._children_offsets = []

        self._num_paths = 0
        self._num_words = 0
        self._num_children = 0

        self._transition_scalor = 0.01
        self._transition_prior = []

    def set_prior(self, index, value):
        #print index, value
        self._transition_prior[index] = value

    def normalizePrior(self):
        norm = sum(self._transition_prior)
        for ii in range(0, len(self._transition_prior)):
            self._transition_prior[ii] /= norm * 1.0
            self._transition_prior[ii] *= self._transition_scalor

        #print 'node', self._offset, '\t', self._transition_scalor, sum(self._transition_prior)
        if self._num_children > 0:
            assert abs(sum(self._transition_prior) - self._transition_scalor) < 1e-6, \
                    "%0.32f %0.32f" % (sum(self._transition_prior), self._transition_scalor)

class Path:
    def __init__(self):
        self._nodes = []

class VocabTreePrior():
    def __init__(self):
        self._nodes = defaultdict(Node)
        self._word_paths = defaultdict(list)
        self._hyperparams = defaultdict(list)
        self._root = -1

    def load_hyperparams(self, hyper_file):
        hypers = codecs.open(hyper_file, 'r', 'utf-8');
        for line in hypers:
            line = line.strip()
            words = line.split(' ')
            nm = words[0]
            self._hyperparams[nm] = float(words[1])

    def load_tree(self, wn_files, vocab):
        for ii in glob(wn_files):
            wnfile = open(ii, 'r')
            wn = WordNetFile()
            wn.ParseFromString(wnfile.read())

            new_root = wn.root
            #print new_root, self._root
            assert ((new_root == -1) | (self._root == -1) | (new_root == self._root))
            if new_root >= 0:
                self._root = new_root

            for jj in wn.synsets:
                n = Node()
                n._offset = jj.offset
                n._raw_count = jj.raw_count
                n._hypo_count = jj.hyponym_count
                n._hyperparam_name = jj.hyperparameter

                n._transition_scalor = self._hyperparams[jj.hyperparameter]
                for ww in jj.children_offsets:
                    n._children_offsets.append(ww)
                    n._num_children += 1

                for ww in jj.words:
                    #print "hello", ww.term_str
                    tmp = unicode(ww.term_str).encode('utf-8')
                    term_id = vocab.index(tmp)
                    n._words.append(term_id)
                    n._words_count.append(ww.count)
                    n._num_words += 1

                self._nodes[n._offset] = n

    def depth_first_search(self, depth, node_index, traversed, next_pointers):
        max_depth = depth
        assert node_index >= 0
        traversed.append(node_index)
        self._nodes[node_index]._num_paths += 1
        # words of this node
        #print self._nodes[node_index]._words
        for ii in range(0, self._nodes[node_index]._num_words):
            p = Path()
            p._nodes = copy(traversed)
            p._child_index = next_pointers
            p._final_word = self._nodes[node_index]._words[ii]
            self._word_paths[p._final_word].append(p)

        # children of this node
        for ii in range(0, self._nodes[node_index]._num_children):
            child_index = self._nodes[node_index]._children_offsets[ii]
            next_pointers.append(child_index)
            # self._nodes[node_index].set_prior
            child_depth = self.depth_first_search(depth+1, child_index, traversed, next_pointers)

            next_pointers.pop()
            max_depth = max(child_depth, max_depth)
            #print child_depth, depth, max_depth

        traversed.pop()
        return max_depth

    def set_prior(self):
        for index in self._nodes.keys():
            node = self._nodes[index]

            if node._num_children > 0:
                assert node._num_words == 0
                node._transition_prior = zeros(node._num_children)
                for ii in range(0, node._num_children):
                    child_index = node._children_offsets[ii]
                    node.set_prior(ii, self._nodes[child_index]._hypo_count)

            # leaf nodes might contain multiple words
            # if yes, set the prior according to the words count in this node
            # It is equal to changing a node containing multiple words to a node
            # containing multiple leaf node and each node contains only one word
            if node._num_words > 1:
                assert node._num_children == 0
                node._transition_prior = zeros(node._num_words)
                for ii in range(0, node._num_words):
                    #child_index = node._words[ii]
                    #print ii, len(node._words), node._num_words,  node._words[ii], node._words_count[ii]
                    node.set_prior(ii, node._words_count[ii])

            node.normalizePrior()
            #print index
            #print node._transition_prior
            #print '\n'

    def _initialize(self, wn_files, hyper_file, vocab):
        self.load_hyperparams(hyper_file)
        self.load_tree(wn_files, vocab)

        traversed = []
        nex_pointers = []
        max_depth = self.depth_first_search(0, 0, traversed, nex_pointers)
        print "max_depth:", max_depth
        self._max_depth = max_depth
        self.set_prior()
        print "number of words:", len(self._word_paths)
        #for ii in self._word_paths.keys():
            #for pp in range(0, len(self._word_paths[ii])):
                #print 'word', ii, '\t', self._word_paths[ii][pp]._nodes
    
    '''
    # added by Ke Zhai
    def _initialize(self, wn_files, vocab, default_correlation_prior, positive_correlation_prior, negative_correlation_prior):
        #self.load_hyperparams(hyper_file)

        self._hyperparams['DEFAULT_'] = default_correlation_prior;
        self._hyperparams['NL_'] = default_correlation_prior;
        self._hyperparams['ML_'] = positive_correlation_prior;
        self._hyperparams['CL_'] = negative_correlation_prior;
            
        self.load_tree(wn_files, vocab)

        traversed = []
        nex_pointers = []
        max_depth = self.depth_first_search(0, 0, traversed, nex_pointers)
        print "max_depth:", max_depth
        self._max_depth = max_depth
        self.set_prior()
        print "number of words:", len(self._word_paths)
        #for ii in self._word_paths.keys():
            #for pp in range(0, len(self._word_paths[ii])):
                #print 'word', ii, '\t', self._word_paths[ii][pp]._nodes
    '''


'''
def getVocab(vocab_file):
    word_lookup = []
    vocab = open(vocab_file, 'r')
    for line in vocab:
        line = line.strip()
        words = line.split('\t')
        word_lookup.append(words[1])
    return word_lookup

if __name__ == "__main__":

    vocab_file = '../ldawn/vocab/toy.voc'
    vocab = getVocab(vocab_file)

    wn_files = '../ldawn/wn/toy.wn.*'
    hyper_file = '../ldawn/hyperparameters/wn_hyperparams'
    tree = VocabTreePrior()
    tree._initialize(wn_files, hyper_file, vocab)
'''