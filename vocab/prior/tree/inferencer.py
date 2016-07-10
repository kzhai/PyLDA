"""
UncollapsedVariationalBayes for Vanilla LDA with Tree Prior
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy
import nltk
import sys
import codecs;
import collections;

try:
    import cPickle as pickle
except:
    import pickle

#import inferencer

#from inferencer import compute_dirichlet_expectation
#from inferencer import Inferencer

"""
This is a python implementation of lda with tree prior, based on variational inference, with hyper parameter updating.

References:
[1] Y. Hu, J. Boyd-Graber, and B. Satinoff. Interactive Topic Modeling. Association for Computational Linguistics (ACL), 2011.
[2] Y. Hu, K. Zhai, V. and J. Boyd-Graber. Interactive Topic Modeling. Association for Computational Linguistics (ACL), 2011.
"""

def compute_dirichlet_expectation(dirichlet_parameter):
    if (len(dirichlet_parameter.shape) == 1):
        return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
    return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:, numpy.newaxis]

class Inferencer():
    """
    """
    def __init__(self,
                 hyper_parameter_optimize_interval=1,
                 ):
        '''
        update_hyper_parameter=True,
        alpha_update_decay_factor=0.9,
        alpha_maximum_decay=10,
        alpha_converge_threshold=0.000001,
        alpha_maximum_iteration=100,
        model_likelihood_threshold=0.00001
        '''
        
        #Inferencer.__init__(self, hyper_parameter_optimize_interval);
        self._hyper_parameter_optimize_interval = hyper_parameter_optimize_interval;
        assert(self._hyper_parameter_optimize_interval>0);
        
        #self._local_parameter_iterations = local_parameter_iterations
        #assert(self._local_maximum_iteration>0)

        '''
        self._update_hyper_parameter=update_hyper_parameter;

        self._alpha_update_decay_factor = alpha_update_decay_factor;
        self._alpha_maximum_decay = alpha_maximum_decay;
        self._alpha_converge_threshold = alpha_converge_threshold;
        self._alpha_maximum_iteration = alpha_maximum_iteration;

        self._model_likelihood_threshold = model_likelihood_threshold;
        '''
        
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, vocab, prior_tree, number_of_topics, alpha_alpha):
        # data
        
        self.parse_vocabulary(vocab);
        
        # initialize the total number of topics.
        self._number_of_topics = number_of_topics

        self._counter = 0;
        
        # initialize a K-dimensional vector, valued at 1/K.
        # self._alpha_alpha = numpy.random.random((1, self._number_of_topics)) / self._number_of_topics;
        self._alpha_alpha = numpy.zeros((1, self._number_of_topics)) + alpha_alpha;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        # self._data = data

        # initialize the size of the collection, i.e., total number of documents.
        # self._number_of_documents = len(self._data)
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        # self._number_of_terms = len(self._type_to_index)
        
        self.update_tree_structure(prior_tree);
        
        '''
        # initialize a D-by-K matrix gamma, valued at N_d/K
        # self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha + 1.0 * self._number_of_paths / self._number_of_topics;
        # self._gamma = numpy.tile(self._alpha_alpha + 1.0 * self._number_of_terms / self._number_of_topics, (self._number_of_documents, 1));
        self._gamma = self._alpha_alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((self._number_of_documents, self._number_of_topics));
        
        # initialize a _E_log_beta variable, indexed by node, valued by a K-by-C matrix, where C stands for the number of children of that node
        self._E_log_beta = numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_edges));
        for node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[node_index];
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
        '''
        
    def update_tree_structure(self, prior_tree):
        self._maximum_depth = prior_tree._max_depth
        
        self._edge_to_index = {}
        self._index_to_edge = {}

        self._edges_from_internal_node = collections.defaultdict(list);
        
        self._edge_prior = [];

        for parent_node in prior_tree._nodes.keys():
            node = prior_tree._nodes[parent_node]
            
            # if the node is an internal node, compute the prior scalar for every edge
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                for position_index in xrange(len(node._children_offsets)):
                    child_node = node._children_offsets[position_index];
                    self._edge_to_index[(parent_node, child_node)] = len(self._edge_to_index);
                    self._index_to_edge[len(self._index_to_edge)] = (parent_node, child_node);
                    
                    self._edges_from_internal_node[parent_node].append(self._edge_to_index[(parent_node, child_node)]);
                    
                    self._edge_prior.append(node._transition_prior[position_index]);

        self._number_of_edges = len(self._edge_prior);
        
        assert(len(self._edge_to_index) == self._number_of_edges);
        assert(len(self._index_to_edge) == self._number_of_edges);
                    
        self._edge_prior = numpy.array(self._edge_prior)[numpy.newaxis, :];
        assert(self._edge_prior.shape == (1, self._number_of_edges));
        
        '''
        print "edge index mapping"
        for (parent_node, child_node) in self._edge_to_index:
            print "(", parent_node, child_node, ")", self._edge_to_index[(parent_node, child_node)];
        print "index edge mapping"
        for edge_index in self._index_to_edge:
            print edge_index, self._index_to_edge[edge_index];
        print "edge prior", self._edge_prior
        print "edges from internal node"
        for internal_node in self._edges_from_internal_node:
            print internal_node, self._edges_from_internal_node[internal_node]
        '''
        
        self._edges_along_path = collections.defaultdict(list);
        self._paths_through_edge = collections.defaultdict(list);

        # word list indexed by path node_index 
        self._path_index_to_word_index = [];
        self._word_index_to_path_indices = collections.defaultdict(list);
        
        # set of path indices that passes through a node
        # self._paths_through_internal_node = defaultdict(list)

        # compute the prior for every path, initialize self._path_prior_sum_of_word, self._path_prior_of_word,        
        path_index = 0;
        # iterate over all leaf nodes in the prior_tree
        for word in prior_tree._word_paths.keys():
            
            # iterate over all paths in the path of the current word (leaf node)
            for path in xrange(len(prior_tree._word_paths[word])):
                self._path_index_to_word_index.append(word)
                self._word_index_to_path_indices[word].append(path_index);

                nodes_along_word_path = prior_tree._word_paths[word][path]._nodes
                
                # if current word (leaf node) contains multiple words, add a leaf node_index for each word
                # if nodes_along_word_path[-1] in word_leaf[word].keys():
                    # leaf_index = word_leaf[word][nodes_along_word_path[-1]]
                    # nodes_along_word_path.append(leaf_index)
                
                for position_index in xrange(len(nodes_along_word_path) - 1):
                    parent_node = nodes_along_word_path[position_index];
                    child_node = nodes_along_word_path[position_index + 1];

                    # if parent_node not in self._paths_through_internal_node.keys():
                        # self._paths_through_internal_node[parent_node] = set()
                    # self._paths_through_internal_node[parent_node].add(path_index)
                    # self._paths_through_internal_node[parent_node].append(path_index)
                    
                    self._edges_along_path[path_index].append(self._edge_to_index[(parent_node, child_node)]);
                    self._paths_through_edge[self._edge_to_index[(parent_node, child_node)]].append(path_index);
                    
                    # self._paths_through_edge[self._edge_to_index[(parent_node, child_node)]].append(path_index);

                '''                    
                for edge in zip(nodes_along_word_path[:-1], nodes_along_word_path[1:]):
                    edge_index = self._edge_to_index[edge];
                    
                    self._edges_along_path[path_index].append(edge_index);
                    self._paths_through_edge[edge_index].append(path_index);
                '''

                path_index += 1;
                    
                # self._word_paths[word][path_index] = nodes_along_word_path

        self._number_of_paths = len(self._path_index_to_word_index);
        
        '''
        print "path word mapping"
        for path_index in self._path_index_to_word_index:
            print path_index, self._path_index_to_word_index[path_index];
        print "word path mapping"
        for word_index in self._word_index_to_path_indices:
            print word_index, self._word_index_to_path_indices[word_index];
        print "edges along path"
        for path_index in self._edges_along_path:
            print path_index, self._edges_along_path[path_index];
        #print "paths through internal node"
        #for internal_node in self._paths_through_internal_node:
            #print internal_node, self._paths_through_internal_node[internal_node];
        '''


        '''
        self._edge_prior_at_node = defaultdict(dict)
        self._edge_prior_sum_at_node = {}

        self._path_prior_of_word = defaultdict(dict)
        self._path_prior_sum_of_word = {}

        # this data structure is to handle the case that one node contains multiple words
        leaf_index = len(prior_tree._nodes.keys()) - 1
        word_leaf = defaultdict(dict)

        # compute the scalar for every edge, initialize self._edge_prior_at_node and self._edge_prior_sum_at_node
        for node_index in prior_tree._nodes.keys():
            node = prior_tree._nodes[node_index]
            
            # if the node is an internal node, compute the prior scalar for every edge
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                self._edge_prior_sum_at_node[node_index] = node._transition_scalar
                
                #self._number_of_edges += len(node._children_offsets);
                
                for child_index in xrange(len(node._children_offsets)):
                    child_index = node._children_offsets[child_index]
                    self._edge_prior_at_node[node_index][child_index] = node._transition_prior[child_index]

            # if the node is a leaf node and it contains multiple words.
            # if yes, set the prior according to the words count in this node
            # It is equal to changing a node containing multiple words to a node
            # containing multiple leaf node and each node contains only one word
            if len(node._words) > 1:
                assert len(node._children_offsets) == 0
                assert len(node._words) > 0
                self._edge_prior_sum_at_node[node_index] = node._transition_scalar
                
                # TODO: increase the total number of edge
                for child_index in range(0, len(node._words)):
                    word_index = node._words[child_index]
                    leaf_index += 1
                    word_leaf[word_index][node_index] = leaf_index
                    self._edge_prior_at_node[node_index][leaf_index] = node._transition_prior[child_index]

        # nodes list indexed by word node_index and path node_index
        self._word_paths = defaultdict(dict)
        
        # word list indexed by path node_index 
        self._path_index_to_word_index = [];
        self._word_index_to_path_indices = defaultdict(set);
        
        # set of path indices that passes through a node 
        self._paths_through_internal_node = defaultdict()

        # compute the prior for every path, initialize self._path_prior_sum_of_word, self._path_prior_of_word,        
        path_index = -1
        # iterate over all leaf nodes in the prior_tree
        for word in prior_tree._word_paths.keys():
            self._path_prior_sum_of_word[word] = 0
            
            # iterate over all paths in the path of the current word (leaf node)
            for path in xrange(len(prior_tree._word_paths[word])):
                path_index += 1

                self._path_index_to_word_index.append(word)
                self._word_index_to_path_indices[word].add(path_index);

                nodes_along_word_path = prior_tree._word_paths[word][path]._nodes
                
                # if current word (leaf node) contains multiple words, add a leaf node_index for each word
                if nodes_along_word_path[-1] in word_leaf[word].keys():
                    leaf_index = word_leaf[word][nodes_along_word_path[-1]]
                    nodes_along_word_path.append(leaf_index)

                prob = 1.0
                for node_index in xrange(len(nodes_along_word_path) - 1):
                    parent = nodes_along_word_path[node_index]
                    child = nodes_along_word_path[node_index + 1]
                    prob *= self._edge_prior_at_node[parent][child]
                self._path_prior_of_word[word][path_index] = prob
                self._path_prior_sum_of_word[word] += prob
                    
                for node_index in nodes_along_word_path:
                    if node_index not in self._paths_through_internal_node.keys():
                        self._paths_through_internal_node[node_index] = set()
                    self._paths_through_internal_node[node_index].add(path_index)

                self._word_paths[word][path_index] = nodes_along_word_path

        self._number_of_paths = len(self._path_index_to_word_index);
        '''

        '''
        self._E_log_beta = defaultdict();

        for node_index in prior_tree._nodes.keys():
            node = prior_tree._nodes[node_index]
            
            # if the node is an internal node, compute the prior scalar for every edge
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                
                self._E_log_beta[node_index] = compute_dirichlet_expectation(numpy.random.gamma(100., 1./100., (self._number_of_topics, len(node._children_offsets))));
        '''

    '''
    # TODO: make sure the data is properly stored...
    def test(self, data):
        # train_data = self._data;
        # train_gamma=self._gamma;

        self._data = data;
        self._number_of_documents = len(data);
        self._gamma = self._alpha_alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((self._number_of_documents, self._number_of_topics));
        
        clock_e_step = time.time();
        phi_sufficient_statistics, document_level_log_likelihood = self.e_step();
        test_gamma = self._gamma;
        clock_e_step = time.time() - clock_e_step;
        
        # self._data = train_data;
        # self._gamma = train_gamma;
        # self._number_of_documents = len(train_data);

        return test_gamma
    '''

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """
    def optimize_hyperparameters(self, alpha_sufficient_statistics, hyper_parameter_iteration=100, hyper_parameter_decay_factor=0.9, hyper_parameter_maximum_decay=10, hyper_parameter_converge_threshold=1e-6):
        assert (alpha_sufficient_statistics.shape == (self._number_of_topics, ));
        alpha_update = self._alpha_alpha;
        
        decay = 0;
        for alpha_iteration in xrange(hyper_parameter_iteration):
            alpha_sum = numpy.sum(self._alpha_alpha);
            alpha_gradient = self._number_of_documents * (scipy.special.psi(alpha_sum) - scipy.special.psi(self._alpha_alpha)) + alpha_sufficient_statistics;
            alpha_hessian = -self._number_of_documents * scipy.special.polygamma(1, self._alpha_alpha);
            
            if numpy.any(numpy.isinf(alpha_gradient)) or numpy.any(numpy.isnan(alpha_gradient)):
                print "illegal alpha gradient vector", alpha_gradient

            sum_g_h = numpy.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = self._number_of_documents * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the alpha vector
            while True:
                singular_hessian = False

                step_size = numpy.power(hyper_parameter_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                #print "step size is", step_size
                assert(self._alpha_alpha.shape == step_size.shape);
                
                if numpy.any(self._alpha_alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self._alpha_alpha - step_size;
                
                if singular_hessian:
                    decay += 1;
                    if decay > hyper_parameter_maximum_decay:
                        break;
                else:
                    break;
                
            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - self._alpha_alpha));
            self._alpha_alpha = alpha_update;
            if mean_change <= hyper_parameter_converge_threshold:
                break;

        return

    def parse_vocabulary(self, vocab):
        self._type_to_index = {};
        self._index_to_type = {};
        for word in set(vocab):
            self._index_to_type[len(self._index_to_type)] = word;
            self._type_to_index[word] = len(self._type_to_index);
            
        self._vocab = self._type_to_index.keys();
        
    def parse_data(self):
        raise NotImplementedError;

    """
    """
    def learning(self):
        raise NotImplementedError;
    
    """
    """
    def inference(self):
        raise NotImplementedError;

    def export_beta(self, exp_beta_path, top_display=-1):
        raise NotImplementedError;
            
if __name__ == "__main__":
    raise NotImplementedError;










    '''
    def e_step(self):
        raise NotImplementedError

    def m_step(self, phi_sufficient_statistics):
        assert phi_sufficient_statistics.shape == (self._number_of_topics, self._number_of_paths);
        assert numpy.min(phi_sufficient_statistics) >= 0, phi_sufficient_statistics;
        
        var_beta = numpy.tile(self._edge_prior, (self._number_of_topics, 1))
        assert var_beta.shape == (self._number_of_topics, self._number_of_edges);
        assert numpy.min(var_beta) >= 0;
        # for internal_node_index in self._edges_from_internal_node:
            # edges_indices_list = self._edges_from_internal_node[internal_node_index];
            # var_beta[:, edges_indices_list] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_internal_node[internal_node_index]], axis=1)[:, numpy.newaxis];
        
        for edge_index in self._index_to_edge:
            # print var_beta[:, edge_index].shape, numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis].shape;
            var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
        del edge_index
        assert(var_beta.shape == (self._number_of_topics, self._number_of_edges));
        assert numpy.min(var_beta) >= 0;

        self._E_log_beta = numpy.copy(var_beta);
        for internal_node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[internal_node_index];
            assert numpy.min(self._E_log_beta[:, edge_index_list]) >= 0;
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
        del internal_node_index, edge_index_list;
        assert numpy.min(var_beta) >= 0;

        corpus_level_log_likelihood = 0;
        for internal_node_index in self._edges_from_internal_node:
            edges_indices_list = self._edges_from_internal_node[internal_node_index];
            
            # term 3
            tmp = corpus_level_log_likelihood
            corpus_level_log_likelihood += (scipy.special.gammaln(numpy.sum(self._edge_prior[:, edges_indices_list])) - numpy.sum(scipy.special.gammaln(self._edge_prior[:, edges_indices_list]))) * self._number_of_topics;
            corpus_level_log_likelihood += numpy.sum(numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T));
            
            # term 6
            # corpus_level_log_likelihood += numpy.sum( - scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)))
            # corpus_level_log_likelihood += numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]));
            corpus_level_log_likelihood += numpy.sum(-scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)) + numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]), axis=1));
            corpus_level_log_likelihood += numpy.sum(-(var_beta[:, edges_indices_list] - 1) * compute_dirichlet_expectation(var_beta[:, edges_indices_list]));
                    
        assert numpy.min(var_beta) >= 0;

        # compute the sufficient statistics for alpha and update
        if self._update_hyper_parameter:
            # alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
            alpha_sufficient_statistics = compute_dirichlet_expectation(self._gamma);
            alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
            assert alpha_sufficient_statistics.shape == (1, self._number_of_topics);
            self.update_alpha(alpha_sufficient_statistics)
            print "update document topic dirichlet hyperparameter alpha to:", self._alpha_alpha
        
        # print numpy.sum(numpy.exp(self._E_log_beta), axis=1);
        
        return corpus_level_log_likelihood

    def learning(self):
        self._counter += 1;
        
        clock_e_step = time.time();        
        phi_sufficient_statistics, document_level_log_likelihood = self.e_step();
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();        
        corpus_level_log_likelihood = self.m_step(phi_sufficient_statistics);
        clock_m_step = time.time() - clock_m_step;
                
        # compute the log-likelihood of alpha terms
        # alpha_sum = numpy.sum(self._alpha_alpha, axis=1);
        # likelihood_alpha = -numpy.sum(scipy.special.gammaln(self._alpha_alpha), axis=1);
        # likelihood_alpha += scipy.special.gammaln(alpha_sum);
        # likelihood_alpha *= self._number_of_documents;
        
        # likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
        # likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));

        # new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
        
        new_likelihood = document_level_log_likelihood + corpus_level_log_likelihood;
        
        print "e_step and m_step of iteration %d finished in %d and %d seconds respectively with log likelihood %g" % (self._counter, clock_e_step, clock_m_step, new_likelihood)
        # print "document likelihood: ", document_level_log_likelihood
        # print "corpus likelihood: ", corpus_level_log_likelihood
        # if abs((new_likelihood - old_likelihood) / old_likelihood) < self._model_likelihood_threshold:
            # print "model likelihood converged..."
            # break
        # old_likelihood = new_likelihood;
        
        return new_likelihood
    '''

    '''
    def dump_tree(self, output_tree_file):
        output_stream = codecs.open(output_tree_file, 'w', 'utf-8');

        pickle.dump(self._maximum_depth, output_stream);

        pickle.dump(self._edge_to_index, output_stream);
        pickle.dump(self._index_to_edge, output_stream);

        # pickle.dump(self._number_of_edges, output_stream);

        pickle.dump(self._edges_from_internal_node, output_stream);

        pickle.dump(self._edge_prior, output_stream);
        
        output_stream.flush();

        pickle.dump(self._path_index_to_word_index, output_stream);
        pickle.dump(self._word_index_to_path_indices, output_stream);

        # pickle.dump(self._number_of_paths, output_stream);

        pickle.dump(self._edges_along_path, output_stream);
        pickle.dump(self._paths_through_edge, output_stream);
        
        output_stream.flush();
        output_stream.close();

    def dump_parameters(self, output_parameter_file):
        # this pops up error
        output_stream = codecs.open(output_parameter_file, 'w', 'utf-8');
        
        pickle.dump(self._counter, output_stream);
        
        pickle.dump(self._type_to_index, output_stream);
        pickle.dump(self._index_to_type, output_stream);

        # print >> sys.stderr, "dump", self._index_to_type[0], self._index_to_type[1], self._index_to_type[2], self._index_to_type[3]

        pickle.dump(self._number_of_topics, output_stream);
        pickle.dump(self._number_of_edges, output_stream);
        pickle.dump(self._number_of_paths, output_stream);
        
        pickle.dump(self._alpha_alpha, output_stream);
        
        # pickle.dump(self._hybrid_mode, output_stream);
        
        output_stream.flush();
        output_stream.close();
        
    def load_params(self, input_parameter_file):
        input_stream = codecs.open(input_parameter_file, 'r');

        self._counter = pickle.load(input_stream);
        
        self._type_to_index = pickle.load(input_stream);
        self._index_to_type = pickle.load(input_stream);

        # tmp = unicode(ww.term_str).encode('utf-8')
        # print >> sys.stderr, len(self._type_to_index), self._type_to_index
        # print >> sys.stderr, "load", self._index_to_type[0];
        # print >> sys.stderr, "load", unicode(self._index_to_type[0]).encode('utf-8')

        self._number_of_topics = pickle.load(input_stream);
        self._number_of_edges = pickle.load(input_stream);
        self._number_of_paths = pickle.load(input_stream);
        
        self._alpha_alpha = pickle.load(input_stream);
        
        # self._hybrid_mode = pickle.load(input_stream);
        
        input_stream.close();
        
        print >> sys.stderr, "successfully load params from %s..." % input_parameter_file
        
    def load_tree(self, input_tree_file):
        input_stream = codecs.open(input_tree_file, 'r');

        self._maximum_depth = pickle.load(input_stream);

        self._edge_to_index = pickle.load(input_stream);
        self._index_to_edge = pickle.load(input_stream);
        
        # self._number_of_edges = pickle.load(input_stream);

        self._edges_from_internal_node = pickle.load(input_stream);

        self._edge_prior = pickle.load(input_stream);
        
        self._path_index_to_word_index = pickle.load(input_stream);
        self._word_index_to_path_indices = pickle.load(input_stream);
        
        # self._number_of_paths = pickle.load(input_stream);

        self._edges_along_path = pickle.load(input_stream);
        self._paths_through_edge = pickle.load(input_stream);
        
        input_stream.close();
        
        print >> sys.stderr, "successfully load tree from %s..." % input_tree_file

    def dump_E_log_beta(self, output_beta_file):
        numpy.savetxt(output_beta_file, self._E_log_beta);
        
    def load_E_log_beta(self, input_beta_file):
        self._E_log_beta = numpy.loadtxt(input_beta_file);
        print >> sys.stderr, "successfully load E-log-beta from %s..." % input_beta_file
        
    def dump_gamma(self, output_gamma_file):
        output_stream = codecs.open(output_gamma_file, 'w');
        pickle.dump(self._gamma, output_stream);
        output_stream.flush();
        output_stream.close();
        
        # numpy.savetxt(output_gamma_file, self._gamma);
        
    def load_gamma(self, input_gamma_file):
        input_stream = codecs.open(input_gamma_file, 'r');
        self._gamma = pickle.load(input_stream);
        input_stream.close();
        
        print >> sys.stderr, "successfully load gamma from %s..." % input_gamma_file

        # self._gamma = numpy.loadtxt(input_gamma_file);
    '''