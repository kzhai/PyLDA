"""
VariationalBayes for Vanilla LDA with Tree Prior
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

# from util.log_math import log_normalize
# from collections import defaultdict;

# import prior.tree.inferencer

"""
This is a python implementation of vanilla lda with tree prior, based on variational inference, with hyper parameter updating.

References:
[1] Y. Hu, J. Boyd-Graber, and B. Satinoff. Interactive Topic Modeling. Association for Computational Linguistics (ACL), 2011.
"""

from inferencer import Inferencer, compute_dirichlet_expectation

class VariationalBayes(Inferencer):
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
        model_likelihood_threshold=0.00001,
        
        gamma_converge_threshold=0.000001,
        gamma_maximum_iteration=20
        '''
        
        Inferencer.__init__(self, hyper_parameter_optimize_interval);
        
        # Inferencer.__init__(self, update_hyper_parameter, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration, model_likelihood_threshold);
        
        # self._alpha_update_decay_factor = alpha_update_decay_factor;
        # self._alpha_maximum_decay = alpha_maximum_decay;
        # self._alpha_converge_threshold = alpha_converge_threshold;
        # self._alpha_maximum_iteration = alpha_maximum_iteration;
        
        # self._gamma_maximum_iteration = gamma_maximum_iteration;
        # self._gamma_converge_threshold = gamma_converge_threshold;

        # self._model_likelihood_threshold = model_likelihood_threshold;

    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    # def _initialize(self, data, vocab, prior_tree, number_of_topics, alpha):
    def _initialize(self, corpus, vocab, prior_tree, number_of_topics, alpha_alpha):
        Inferencer._initialize(self, vocab, prior_tree, number_of_topics, alpha_alpha);

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        self._corpus = corpus;
        self._parsed_corpus = self.parse_data();
        
        # initialize the size of the collection, i.e., total number of documents.
        self._number_of_documents = len(self._parsed_corpus);

        '''
        # initialize a D-by-K matrix gamma, valued at N_d/K
        #self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;

        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        #self._eta = numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types));
        #self._E_log_eta = compute_dirichlet_expectation(self._eta);
        '''

        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        # self._number_of_terms = len(self._type_to_index)
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        # self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha + 1.0 * self._number_of_paths / self._number_of_topics;
        # self._gamma = numpy.tile(self._alpha_alpha + 1.0 * self._number_of_terms / self._number_of_topics, (self._number_of_documents, 1));
        self._gamma = self._alpha_alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((self._number_of_documents, self._number_of_topics));
        
        # initialize a _E_log_beta variable, indexed by node, valued by a K-by-C matrix, where C stands for the number of children of that node
        self._var_beta = numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_edges));
        # for edge_index in self._index_to_edge:
            # self._var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
            
        '''
        self._E_log_beta = numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_edges));
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
            document = {};
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue;
                
                type_id = self._type_to_index[token];
                if type_id not in document:
                    document[type_id] = 0;
                document[type_id] += 1;
                
            assert len(document) > 0, "document %d collapsed..." % doc_count;
            if len(document) == 0:
                sys.stderr.write("warning: document collapsed during parsing");
                continue;
            
            documents.append(document);
    
            doc_count += 1
            if doc_count % 10000 == 0:
                print "successfully parse %d documents..." % doc_count;
        
        print "successfully parse %d documents..." % len(documents)
        
        return documents

    def e_step(self, parsed_corpus=None, local_parameter_iteration=50, local_parameter_converge_threshold=1e-6):
        if parsed_corpus == None:
            documents = self._parsed_corpus
        
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
        for doc_id in xrange(number_of_documents):
            # update phi and gamma until gamma converges
            for gamma_iteration in xrange(local_parameter_iteration):
                document_phi = numpy.zeros((self._number_of_topics, self._number_of_paths));
                
                phi_entropy = 0;
                phi_E_log_eta = 0;

                # E_log_theta = scipy.special.psi(self._gamma[[doc_id], :]).T;
                # assert E_log_theta.shape==(self._number_of_topics, 1);
                
                E_log_theta = compute_dirichlet_expectation(gamma_values[[doc_id], :]).T;
                assert E_log_theta.shape == (self._number_of_topics, 1);
                
                for word_id in documents[doc_id]:  # word_ids:
                    paths_lead_to_current_word = self._word_index_to_path_indices[word_id];

                    # log_phi = numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]).T, (1, len(paths_lead_to_current_word)));
                    log_phi = numpy.tile(E_log_theta, (1, len(paths_lead_to_current_word)));
                    assert log_phi.shape == (self._number_of_topics, len(paths_lead_to_current_word));

                    for position_index in xrange(len(paths_lead_to_current_word)):
                        log_phi[:, position_index] += numpy.sum(E_log_eta[:, self._edges_along_path[paths_lead_to_current_word[position_index]]], axis=1);
                    del position_index

                    # log_phi -= scipy.misc.logsumexp(log_phi, axis=0)[numpy.newaxis, :]
                    log_phi -= scipy.misc.logsumexp(log_phi)
                    path_phi = numpy.exp(log_phi)
                    
                    # compute the phi terms
                    phi_entropy += -numpy.sum(path_phi * numpy.log(path_phi + 1e-100)) * documents[doc_id][word_id];
                    
                    for position_index in xrange(len(paths_lead_to_current_word)):
                        phi_E_log_eta += documents[doc_id][word_id] * numpy.sum(path_phi[:, [position_index]] * numpy.sum(E_log_eta[:, self._edges_along_path[paths_lead_to_current_word[position_index]]], axis=1)[:, numpy.newaxis])
                    del position_index
                    
                    # multiple path_phi with the count of current word
                    document_phi[:, paths_lead_to_current_word] += path_phi * documents[doc_id][word_id];
                
                del word_id, paths_lead_to_current_word
                # print doc_id, "before", self._gamma[[doc_id], :];
                gamma_values[[doc_id], :] = self._alpha_alpha + numpy.sum(document_phi, axis=1)[:, numpy.newaxis].T;
                
            # Note: all terms including E_q[p(\theta | \alpha)], i.e., terms involving \Psi(\gamma), are cancelled due to \gamma updates in E-step
            # document_log_likelihood += numpy.sum((self._alpha_alpha - 1) * compute_dirichlet_expectation(gamma_values[[doc_id], :]));
            # document_log_likelihood += numpy.sum(numpy.sum(document_phi, axis=1)[:, numpy.newaxis].T * compute_dirichlet_expectation(gamma_values[[doc_id], :]));
            # document_log_likelihood += -numpy.sum((gamma_values[[doc_id], :] - 1) * compute_dirichlet_expectation(gamma_values[[doc_id], :]));
            
            # compute the alpha terms
            document_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha_alpha)) - numpy.sum(scipy.special.gammaln(self._alpha_alpha));
            
            # compute the gamma terms
            document_log_likelihood += numpy.sum(scipy.special.gammaln(gamma_values[doc_id, :])) - scipy.special.gammaln(numpy.sum(gamma_values[doc_id, :]));
            
            # compute the phi terms
            document_log_likelihood += phi_entropy;
            
            # Note: all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates in M-step
            if parsed_corpus != None:
                # compute the p(w_{dn} | z_{dn}, \eta) terms, which will be cancelled during M-step during training
                words_log_likelihood += phi_E_log_eta;
            
            phi_sufficient_statistics += document_phi;
        
            if (doc_id + 1) % 1000 == 0:
                print "successfully processed %d documents..." % (doc_id + 1);
                
            del doc_id
        
        if parsed_corpus == None:
            self._gamma = gamma_values;
            return document_log_likelihood, phi_sufficient_statistics
        else:
            return words_log_likelihood, gamma_values

    def m_step(self, phi_sufficient_statistics):
        assert phi_sufficient_statistics.shape == (self._number_of_topics, self._number_of_paths);
        assert numpy.min(phi_sufficient_statistics) >= 0, phi_sufficient_statistics;
        
        self._var_beta = numpy.tile(self._edge_prior, (self._number_of_topics, 1))
        assert self._var_beta.shape == (self._number_of_topics, self._number_of_edges);
        assert numpy.min(self._var_beta) >= 0;
        # for internal_node_index in self._edges_from_internal_node:
            # edges_indices_list = self._edges_from_internal_node[internal_node_index];
            # _var_beta[:, edges_indices_list] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_internal_node[internal_node_index]], axis=1)[:, numpy.newaxis];
        
        for edge_index in self._index_to_edge:
            self._var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
        del edge_index
        assert(self._var_beta.shape == (self._number_of_topics, self._number_of_edges));
        assert numpy.min(self._var_beta) >= 0;

        topic_log_likelihood = 0;
        for internal_node_index in self._edges_from_internal_node:
            edges_indices_list = self._edges_from_internal_node[internal_node_index];
            
            # compute the beta terms
            topic_log_likelihood += self._number_of_topics * (scipy.special.gammaln(numpy.sum(self._edge_prior[:, edges_indices_list])) - numpy.sum(scipy.special.gammaln(self._edge_prior[:, edges_indices_list])))
            # topic_log_likelihood += numpy.sum(numpy.dot((self._edge_prior[:, edges_indices_list] - 1), _var_beta[:, edges_indices_list].T));
            
            # compute the eta terms
            topic_log_likelihood += numpy.sum(numpy.sum(scipy.special.gammaln(self._var_beta[:, edges_indices_list]), axis=1) - scipy.special.gammaln(numpy.sum(self._var_beta[:, edges_indices_list], axis=1)));
            # topic_log_likelihood += numpy.sum(-(_var_beta[:, edges_indices_list] - 1) * compute_dirichlet_expectation(_var_beta[:, edges_indices_list]));
        
        assert numpy.min(self._var_beta) >= 0;
        
        alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0);
        
        return topic_log_likelihood, alpha_sufficient_statistics

    """
    """
    def learning(self):
        self._counter += 1;
        
        clock_e_step = time.time();
        document_log_likelihood, phi_sufficient_statistics = self.e_step();
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();
        topic_log_likelihood, alpha_sufficient_statistics = self.m_step(phi_sufficient_statistics);
        if self._hyper_parameter_optimize_interval > 0 and self._counter % self._hyper_parameter_optimize_interval == 0:
            self.optimize_hyperparameters(alpha_sufficient_statistics);
        clock_m_step = time.time() - clock_m_step;
        
        joint_log_likelihood = document_log_likelihood + topic_log_likelihood;
        print document_log_likelihood, topic_log_likelihood;
        
        print "e_step and m_step of iteration %d finished in %d and %d seconds respectively with log likelihood %g" % (self._counter, clock_e_step, clock_m_step, joint_log_likelihood)
        
        return joint_log_likelihood

    def export_beta(self, exp_beta_path, top_display=-1):
        output = open(exp_beta_path, 'w');
        
        E_log_eta = numpy.copy(self._var_beta);
        assert E_log_eta.shape == (self._number_of_topics, self._number_of_edges)
        for internal_node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[internal_node_index];
            assert numpy.min(E_log_eta[:, edge_index_list]) >= 0;
            E_log_eta[:, edge_index_list] = compute_dirichlet_expectation(E_log_eta[:, edge_index_list]);
        del internal_node_index, edge_index_list;
        
        for topic_index in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (topic_index));
            
            freqdist = nltk.probability.FreqDist()
            for path_index in self._path_index_to_word_index:
                path_rank = 1;
                for edge_index in self._edges_along_path[path_index]:
                    path_rank *= numpy.exp(E_log_eta[topic_index, edge_index]);
                freqdist[path_index] += path_rank;
            
            i = 0;
            for (path_index, path_freq) in freqdist.most_common():
                i += 1;
                output.write("%s\t%g\n" % (self._index_to_type[self._path_index_to_word_index[path_index]], freqdist[path_index]));
                if top_display > 0 and i >= top_display:
                    break;
                
        output.close();

if __name__ == "__main__":
    raise NotImplementedError;
