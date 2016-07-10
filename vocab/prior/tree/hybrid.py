"""
UncollapsedVariationalBayes for Vanilla LDA with Tree Prior
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy
import nltk;
import sys;
import codecs

from collections import defaultdict;

"""
This is a python implementation of vanilla lda with tree prior, based on variational inference, with hyper parameter updating.

References:
[1] Y. Hu, J. Boyd-Graber, and B. Satinoff. Interactive Topic Modeling. Association for Computational Linguistics (ACL), 2011.
"""

def parse_data(documents_file, vocabulary_file=None):
    '''
    type_to_index = {};
    index_to_type = {};
    vocabulary = [];
    if (vocabulary_file!=None):
        input_file = codecs.open(vocabulary_file, mode='r', encoding='utf-8');
        for line in input_file:
            #line = line.strip().split()[0];
            assert len(line.strip().split())==4;
            line = line.strip().split()[1];
            assert line not in type_to_index, "duplicate type for %s" % line;
            type_to_index[line] = len(type_to_index);
            index_to_type[len(index_to_type)] = line;
            vocabulary.append(line);
        input_file.close();
    '''

    # '''
    type_to_index = {};
    index_to_type = {};
    # vocabulary = [];
    if (vocabulary_file != None):
        input_file = codecs.open(vocabulary_file, mode='r', encoding='utf-8');
        for line in input_file:
            # line = line.strip().split()[0];
            assert len(line.strip().split()) == 4;
            line = line.strip().split()[1];
            assert line not in type_to_index, "duplicate type for %s" % line;
            type_to_index[line] = len(type_to_index);
            index_to_type[len(index_to_type)] = line;
            # vocabulary.append(line);
        input_file.close();

    # type_to_index = {};
    # index_to_type = {};
    vocabulary = [];
    if (vocabulary_file != None):
        input_file = open(vocabulary_file, mode='r');
        for line in input_file:
            # line = line.strip().split()[0];
            assert len(line.strip().split()) == 4;
            line = line.strip().split()[1];
            # assert line not in type_to_index, "duplicate type for %s" % line;
            # type_to_index[line] = len(type_to_index);
            # index_to_type[len(index_to_type)] = line;
            vocabulary.append(line);
        input_file.close();
    # '''

    input_file = codecs.open(documents_file, mode="r", encoding="utf-8")
    doc_count = 0
    documents = []
    
    for line in input_file:
        line = line.strip().lower();

        contents = line.split("\t");

        document = [];
        for token in contents[-1].split():
            if token not in type_to_index:
                if vocabulary_file == None:
                    type_to_index[token] = len(type_to_index);
                    index_to_type[len(index_to_type)] = token;
                else:
                    continue;
                
            document.append(type_to_index[token]);
            # document.inc(type_to_index[token]);
            # document.append(type_to_index[token]);
        
        # assert len(document)>0, "document %d collapsed..." % doc_count;

        documents.append(document);
        
        doc_count += 1
        if doc_count % 10000 == 0:
            print "successfully import %d documents..." % doc_count;
    
    input_file.close();

    print "successfully import", len(documents), "documents..."
    return documents, type_to_index, index_to_type, vocabulary

from inferencer import Inferencer;
from inferencer import compute_dirichlet_expectation;
from variational_bayes import VariationalBayes;

class Hybrid(VariationalBayes):
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
        
        number_of_samples=10,
        burn_in_samples=5        
        '''
        
        VariationalBayes.__init__(self, hyper_parameter_optimize_interval);
        
        # Inferencer.__init__(self, update_hyper_parameter, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration, model_likelihood_threshold);
        
        # self._alpha_update_decay_factor = alpha_update_decay_factor;
        # self._alpha_maximum_decay = alpha_maximum_decay;
        # self._alpha_converge_threshold = alpha_converge_threshold;
        # self._alpha_maximum_iteration = alpha_maximum_iteration;
        
        # self._number_of_samples = number_of_samples;
        # self._burn_in_samples = burn_in_samples;
        
        # self._model_likelihood_threshold = model_likelihood_threshold;

    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    '''
    def _initialize(self, data, prior_tree, type_to_index, index_to_type, number_of_topics, alpha):
        self._counter = 0;
        
        self._type_to_index = type_to_index;
        self._index_to_type = index_to_type;
        
        # initialize the total number of topics.
        self._number_of_topics = number_of_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        #self._alpha_alpha = numpy.random.random((1, self._number_of_topics)) / self._number_of_topics;
        self._alpha_alpha = numpy.zeros((1, self._number_of_topics))+alpha;
        #self._eta = eta;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        self._parsed_corpus = data
        
        # initialize the size of the collection, i.e., total number of documents.
        self._number_of_documents = len(self._parsed_corpus)
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        #self._number_of_terms = len(self._type_to_index)
        
        self.update_tree_structure(prior_tree);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        #self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha + 1.0 * self._number_of_paths / self._number_of_topics;
        #self._gamma = numpy.tile(self._alpha_alpha + 1.0 * self._number_of_terms / self._number_of_topics, (self._number_of_documents, 1));
        #self._gamma = self._alpha_alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((self._number_of_documents, self._number_of_topics));
        
        # initialize a _E_log_beta variable, indexed by node, valued by a K-by-C matrix, where C stands for the number of children of that node
        self._E_log_beta = numpy.random.gamma(100., 1./100., (self._number_of_topics, self._number_of_edges));
        for node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[node_index];
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
    '''

    def parse_data(self, corpus=None):
        if corpus == None:
            corpus = self._corpus;
            
        doc_count = 0
        documents = []
        
        for document_line in corpus:
            document = [];
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue;
                document.append(self._type_to_index[token]);
            documents.append(document);
            
            doc_count += 1
            if doc_count % 10000 == 0:
                print "successfully import %d documents..." % doc_count;
        
        print "successfully import", len(documents), "documents..."
        return documents

    def e_step(self, parsed_corpus=None, number_of_samples=10, burn_in_samples=5):
        if parsed_corpus == None:
            documents = self._parsed_corpus
        else:
            documents = parsed_corpus
        
        number_of_documents = len(documents);

        document_log_likelihood = 0;
        words_log_likelihood = 0;

        # initialize a V-by-K matrix phi contribution
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_paths));
        
        # gamma_values = numpy.zeros((number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;
        gamma_values = self._alpha_alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((number_of_documents, self._number_of_topics));

        E_log_eta = numpy.copy(self._var_beta);
        for internal_node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[internal_node_index];
            assert numpy.min(E_log_eta[:, edge_index_list]) >= 0;
            E_log_eta[:, edge_index_list] = compute_dirichlet_expectation(E_log_eta[:, edge_index_list]);
        del internal_node_index, edge_index_list;

        # iterate over all documents
        for document_index in xrange(number_of_documents):
            document_gamma = numpy.zeros(self._alpha_alpha.shape);
            
            topic_path_assignment = {};
            topic_sum = numpy.zeros((1, self._number_of_topics));
            for word_index in xrange(len(documents[document_index])):
                topic_assignment = numpy.random.randint(0, self._number_of_topics);
                path_assignment = numpy.random.randint(0, len(self._word_index_to_path_indices[documents[document_index][word_index]]));
                topic_path_assignment[word_index] = (topic_assignment, path_assignment);
                topic_sum[0, topic_assignment] += 1;
            del word_index, topic_assignment, path_assignment;

            # update path_phi and phi_sum until phi_sum converges
            for sample_index in xrange(number_of_samples):
                # document_phi = numpy.zeros((self._number_of_topics, self._number_of_paths));
                
                phi_entropy = 0;
                phi_E_log_eta = 0;
                
                for word_index in xrange(len(documents[document_index])):
                    word_id = documents[document_index][word_index];
                    topic_sum[0, topic_path_assignment[word_index][0]] -= 1;
                    
                    paths_lead_to_current_word = self._word_index_to_path_indices[word_id];
                    assert len(paths_lead_to_current_word) > 0
                    
                    # path_phi = numpy.tile(scipy.special.psi(self._gamma[[document_index], :]).T, (1, len(paths_lead_to_current_word)));
                    path_phi = numpy.tile((topic_sum + self._alpha_alpha).T, (1, len(paths_lead_to_current_word)));
                    assert path_phi.shape == (self._number_of_topics, len(paths_lead_to_current_word));
                    
                    for path_index in xrange(len(paths_lead_to_current_word)):
                        path_phi[:, path_index] *= numpy.exp(numpy.sum(E_log_eta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1));
                    del path_index
                    
                    assert path_phi.shape == (self._number_of_topics, len(paths_lead_to_current_word));
                    # normalize path_phi over all topics
                    path_phi /= numpy.sum(path_phi);
                    
                    # compute the phi terms
                    phi_entropy += -numpy.sum(path_phi * numpy.log(path_phi + 1e-100));
                    
                    random_number = numpy.random.random();
                    for topic_index in xrange(self._number_of_topics):
                        for path_index in xrange(len(paths_lead_to_current_word)):
                            random_number -= path_phi[topic_index, path_index];
                            if random_number <= 0:
                                break;
                        if random_number <= 0:
                            break;
                    topic_sum[0, topic_index] += 1;
                    topic_path_assignment[word_index] = (topic_index, path_index);
                    
                    if sample_index >= burn_in_samples:
                        phi_sufficient_statistics[topic_index, paths_lead_to_current_word[path_index]] += 1;
                    
                    #
                    #
                    #
                    #
                    #
                        
                    for position_index in xrange(len(paths_lead_to_current_word)):
                        phi_E_log_eta += numpy.sum(path_phi[:, [position_index]] * numpy.sum(E_log_eta[:, self._edges_along_path[paths_lead_to_current_word[position_index]]], axis=1)[:, numpy.newaxis])
                    del position_index
                    
                del word_index, paths_lead_to_current_word
                
                if sample_index >= burn_in_samples:
                    document_gamma += self._alpha_alpha + topic_sum
                    
            # gamma_values[[document_index], :] = self._alpha_alpha + topic_sum;    
            gamma_values[[document_index], :] = document_gamma / (number_of_samples - burn_in_samples);
            
            # Note: all terms including E_q[p(\theta | \alpha)], i.e., terms involving \Psi(\gamma), are cancelled due to \gamma updates in E-step
            # document_log_likelihood += numpy.sum((self._alpha_alpha - 1) * compute_dirichlet_expectation(gamma_values[[document_index], :]));
            # document_log_likelihood += numpy.sum(numpy.sum(document_phi, axis=1)[:, numpy.newaxis].T * compute_dirichlet_expectation(gamma_values[[document_index], :]));
            # document_log_likelihood += -numpy.sum((gamma_values[[document_index], :] - 1) * compute_dirichlet_expectation(gamma_values[[document_index], :]));
            
            # compute the alpha terms
            document_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha_alpha)) - numpy.sum(scipy.special.gammaln(self._alpha_alpha));
            
            # compute the gamma terms
            document_log_likelihood += numpy.sum(scipy.special.gammaln(gamma_values[document_index, :])) - scipy.special.gammaln(numpy.sum(gamma_values[document_index, :]));
            
            # compute the phi terms
            # phi_entropy += -numpy.sum(path_phi * numpy.log(path_phi)) * documents[doc_id][word_id];
            document_log_likelihood += phi_entropy;
            
            # Note: all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates in M-step
            if parsed_corpus != None:
                # compute the p(w_{dn} | z_{dn}, \eta) terms, which will be cancelled during M-step during training
                words_log_likelihood += phi_E_log_eta;
            
            # phi_sufficient_statistics += document_phi;
        
            if (document_index + 1) % 1000 == 0:
                print "successfully processed %d documents..." % (document_index + 1);
                
            del document_index

        phi_sufficient_statistics /= (number_of_samples - burn_in_samples);
        assert phi_sufficient_statistics.shape == (self._number_of_topics, self._number_of_paths);
                
        if parsed_corpus == None:
            self._gamma = gamma_values;
            return document_log_likelihood, phi_sufficient_statistics
        else:
            return words_log_likelihood, gamma_values
        
    """
    def m_step(self, phi_sufficient_statistics):
        assert phi_sufficient_statistics.shape==(self._number_of_topics, self._number_of_paths);

        var_beta = numpy.tile(self._edge_prior, (self._number_of_topics, 1))
        assert var_beta.shape==(self._number_of_topics, self._number_of_edges);
        #for internal_node_index in self._edges_from_internal_node:
            #edges_indices_list = self._edges_from_internal_node[internal_node_index];
            #var_beta[:, edges_indices_list] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_internal_node[internal_node_index]], axis=1)[:, numpy.newaxis];
        
        for edge_index in self._index_to_edge:
            #print var_beta[:, edge_index].shape, numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis].shape;
            var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
        del edge_index
        assert(var_beta.shape == (self._number_of_topics, self._number_of_edges));

        #print "var_beta"
        #print var_beta
        #sys.exit()
        
        self._E_log_beta = var_beta;
        for internal_node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[internal_node_index];
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
        del internal_node_index, edge_index_list;

        
        corpus_level_log_likelihood = 0;
        '''
        for internal_node_index in self._edges_from_internal_node:
            edges_indices_list = self._edges_from_internal_node[internal_node_index];
            corpus_level_log_likelihood += (scipy.special.gammaln(numpy.sum(self._edge_prior[:, edges_indices_list])) - numpy.sum(scipy.special.gammaln(self._edge_prior[:, edges_indices_list]))) * self._number_of_topics;
            corpus_level_log_likelihood += numpy.sum(numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T));
            
            corpus_level_log_likelihood += numpy.sum(-scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)) + numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]), axis=1));
            corpus_level_log_likelihood += numpy.sum(-(var_beta[:, edges_indices_list]-1) * compute_dirichlet_expectation(var_beta[:, edges_indices_list]));
        '''
        
        # TODO: add in alpha updating
        # compute the sufficient statistics for alpha and update
        #alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        #alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
        #self.update_alpha(alpha_sufficient_statistics)
        
        return corpus_level_log_likelihood
    """

    """
    """
    '''
    def train(self):
        self._counter += 1;
        
        clock_e_step = time.time();        
        phi_sufficient_statistics, document_level_log_likelihood, gamma, alpha_sufficient_statistics = self.e_step();
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();        
        corpus_level_log_likelihood = self.m_step(phi_sufficient_statistics, alpha_sufficient_statistics);
        clock_m_step = time.time() - clock_m_step;

        # compute the log-likelihood of alpha terms
        #alpha_sum = numpy.sum(self._alpha_alpha, axis=1);
        #likelihood_alpha = -numpy.sum(scipy.special.gammaln(self._alpha_alpha), axis=1);
        #likelihood_alpha += scipy.special.gammaln(alpha_sum);
        #likelihood_alpha *= self._number_of_documents;
        
        #likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
        #likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));

        #new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
        
        new_likelihood = document_level_log_likelihood + corpus_level_log_likelihood;
        
        print "e_step and m_step of iteration %d finished in %d and %d seconds respectively with log likelihood %g" % (self._counter, clock_e_step, clock_m_step, new_likelihood)
        
        #if abs((new_likelihood - old_likelihood) / old_likelihood) < self._model_likelihood_threshold:
            #print "model likelihood converged..."
            #break
        #old_likelihood = new_likelihood;
        
        return new_likelihood
    '''

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """
    '''
    def update_alpha(self, alpha_sufficient_statistics):
        assert(alpha_sufficient_statistics.shape == (1, self._number_of_topics));        
        alpha_update = self._alpha_alpha;
        
        decay = 0;
        for alpha_iteration in xrange(self._alpha_maximum_iteration):
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

                step_size = numpy.power(self._alpha_update_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                #print "step size is", step_size
                assert(self._alpha_alpha.shape == step_size.shape);
                
                if numpy.any(self._alpha_alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self._alpha_alpha - step_size;
                
                if singular_hessian:
                    decay += 1;
                    if decay > self._alpha_maximum_decay:
                        break;
                else:
                    break;
                
            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - self._alpha_alpha));
            self._alpha_alpha = alpha_update;
            if mean_change <= self._alpha_converge_threshold:
                break;

        return
    '''

if __name__ == "__main__":
    raise NotImplementedError;
