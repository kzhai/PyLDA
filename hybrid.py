"""
Hybrid Update for Vanilla LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy;
import scipy
import nltk;
import sys;

from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer
from variational_bayes import VariationalBayes;

"""
This is a python implementation of vanilla lda, based on a lda approach of variational inference and gibbs sampling, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.

References:
[1] D. Mimno, M. Hoffman, D. Blei. Sparse Stochastic Inference for Latent Dirichlet Allocation. Internal Conference on Machine Learning, Jun 2012.
"""
class Hybrid(VariationalBayes, Inferencer):
    """
    """
    def __init__(self,
                 hyper_parameter_optimize_interval=1,
                 
                 #hyper_parameter_iteration=100,
                 #hyper_parameter_decay_factor=0.9,
                 #hyper_parameter_maximum_decay=10,
                 #hyper_parameter_converge_threshold=1e-6,
                 ):
        Inferencer.__init__(self, hyper_parameter_optimize_interval);
        #VariationalBayes.__init__(self, hyper_parameter_optimize_interval, hyper_parameter_iteration, hyper_parameter_decay_factor, hyper_parameter_maximum_decay, hyper_parameter_converge_threshold);
        
    """
    """
    def _initialize(self, corpus, vocab, number_of_topics, alpha_alpha, alpha_beta):
        Inferencer._initialize(self, vocab, number_of_topics, alpha_alpha, alpha_beta);
        
        # self._corpus = corpus;
        self._parsed_corpus = self.parse_data(corpus);
        
        # define the total number of document
        self._number_of_documents = len(self._parsed_corpus);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        #self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha + 1.0 * self._number_of_types / self._number_of_topics;
        self._gamma = numpy.tile(self._alpha_alpha + 1.0 * self._number_of_types / self._number_of_topics, (self._number_of_documents, 1));
        
        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._eta = numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types));
        #self._E_log_eta = compute_dirichlet_expectation(numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types)));
        #self._exp_E_log_beta = numpy.exp(compute_dirichlet_expectation(numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types))));
        
    def parse_data(self, corpus):
        doc_count = 0
        
        word_idss = [];
        
        for document_line in corpus:
            word_ids = [];
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue;
                
                type_id = self._type_to_index[token];
                word_ids.append(type_id);
            
            if len(word_ids)==0:
                sys.stderr.write("warning: document collapsed during parsing");
                continue;
            
            word_idss.append(word_ids);
            
            doc_count+=1
            if doc_count%10000==0:
                print "successfully parse %d documents..." % doc_count;
        
        print "successfully parse %d documents..." % (doc_count);
        
        return word_idss;
    
    def e_step(self, parsed_corpus=None, number_of_samples=10, burn_in_samples=5):
        if parsed_corpus==None:
            word_idss = self._parsed_corpus;
        else:
            word_idss = parsed_corpus;
        number_of_documents = len(word_idss);
        
        E_log_eta = compute_dirichlet_expectation(self._eta)
        exp_E_log_eta = numpy.exp(E_log_eta);
        
        document_log_likelihood = 0;
        words_log_likelihood = 0;

        # initialize a V-by-K matrix phi contribution
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_types));
        
        # initialize a D-by-K matrix gamma values
        gamma_values = numpy.zeros((number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;

        # iterate over all documents
        for doc_id in xrange(number_of_documents):
            phi = numpy.random.random((self._number_of_topics, len(word_idss[doc_id])));
            phi = phi / numpy.sum(phi, axis=0)[numpy.newaxis, :];
            phi_sum = numpy.sum(phi, axis=1)[:, numpy.newaxis];
            assert(phi_sum.shape == (self._number_of_topics, 1));
            
            document_phi = numpy.zeros((len(word_idss[doc_id]), self._number_of_topics));

            # collect phi samples from empirical distribution
            for it in xrange(number_of_samples):
                for word_pos in xrange(len(word_idss[doc_id])):
                    word_index = word_idss[doc_id][word_pos];
                    
                    phi_sum -= phi[:, word_pos][:, numpy.newaxis];
                    
                    # this is to get rid of the underflow error from the above summation, ideally, phi will become all integers after few iterations
                    phi_sum *= (phi_sum > 0);
                    #assert(numpy.all(phi_sum >= 0));

                    temp_phi = (phi_sum.T + self._alpha_alpha) * exp_E_log_eta[:, [word_index]].T;
                    assert(temp_phi.shape == (1, self._number_of_topics));
                    temp_phi /= numpy.sum(temp_phi);

                    # sample a topic for this word
                    temp_phi = numpy.random.multinomial(1, temp_phi[0])[:, numpy.newaxis];
                    assert(temp_phi.shape == (self._number_of_topics, 1));
                    
                    phi[:, word_pos][:, numpy.newaxis] = temp_phi;
                    phi_sum += temp_phi;

                    # discard the first few burn-in sweeps
                    if it < burn_in_samples:
                        continue;
                    
                    phi_sufficient_statistics[:, word_index] += temp_phi[:, 0];
                    document_phi[word_pos, :] += temp_phi[:, 0];

            gamma_values[doc_id, :] = self._alpha_alpha + phi_sum.T[0, :];
            #batch_document_topic_distribution[doc_id, :] = self._alpha_alpha + phi_sum.T[0, :];
            
            document_phi /= (number_of_samples - burn_in_samples);
            # this is to prevent 0 during log()
            document_phi += 1e-100;
            
            # Note: all terms including E_q[p(\theta|\alpha)], i.e., terms involving \Psi(\gamma), are cancelled due to \gamma updates
            # Note: all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates in M-step
            
            # compute the alpha terms
            document_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha_alpha)) - numpy.sum(scipy.special.gammaln(self._alpha_alpha))
            # compute the gamma terms
            document_log_likelihood += numpy.sum(scipy.special.gammaln(gamma_values[doc_id, :])) - scipy.special.gammaln(numpy.sum(gamma_values[doc_id, :]));
            # compute the phi terms
            document_log_likelihood -= numpy.sum(numpy.log(document_phi) * document_phi);
            
            # compute the p(w_{dn} | z_{dn}, \eta) terms, which will be cancelled during M-step
            words_log_likelihood += numpy.sum(document_phi * (E_log_eta[:, word_idss[doc_id]].T));
            
            if (doc_id+1) % 1000==0:
                print "successfully processed %d documents in hybrid mode..." % (doc_id+1);

        phi_sufficient_statistics /= (number_of_samples - burn_in_samples);
        
        if parsed_corpus==None:
            self._gamma = gamma_values;
            return document_log_likelihood, phi_sufficient_statistics
        else:
            return words_log_likelihood, gamma_values

if __name__ == "__main__":
    print "not implemented..."
