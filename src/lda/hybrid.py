"""
Hybrid Update for Vanilla LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy;
import scipy
import nltk;

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
        
        self._corpus = corpus;
        self.parse_data();
        
        # define the total number of document
        self._number_of_documents = len(self._word_idss);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        #self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha + 1.0 * self._number_of_types / self._number_of_topics;
        self._gamma = numpy.tile(self._alpha_alpha + 1.0 * self._number_of_types / self._number_of_topics, (self._number_of_documents, 1));
        
        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._E_log_beta = compute_dirichlet_expectation(numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types)));
        #self._exp_E_log_beta = numpy.exp(compute_dirichlet_expectation(numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types))));
        
    def parse_data(self):
        doc_count = 0
        
        self._word_idss = [];
        
        for document_line in self._corpus:
            word_ids = [];
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue;
                
                type_id = self._type_to_index[token];
                word_ids.append(type_id);
            
            self._word_idss.append(word_ids);
            
            doc_count+=1
            if doc_count%10000==0:
                print "successfully parse %d documents..." % doc_count;
        
        print "successfully parse %d documents..." % (doc_count);
    
    def e_step(self, number_of_samples=10, burn_in_samples=5):
        likelihood_phi = 0.0;

        # initialize a V-by-K matrix phi contribution
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_types));

        exp_E_log_beta = numpy.exp(self._E_log_beta);
        
        # Initialize the variational distribution q(theta|gamma) for the mini-batch
        #batch_document_topic_distribution = numpy.zeros((batchD, self._number_of_topics));

        # iterate over all documents
        for doc_id in xrange(self._number_of_documents):
            phi = numpy.random.random((self._number_of_topics, len(self._word_idss[doc_id])));
            phi = phi / numpy.sum(phi, axis=0)[numpy.newaxis, :];
            phi_sum = numpy.sum(phi, axis=1)[:, numpy.newaxis];
            assert(phi_sum.shape == (self._number_of_topics, 1));
            
            document_phi = numpy.zeros((len(self._word_idss[doc_id]), self._number_of_topics));

            # collect phi samples from empirical distribution
            for it in xrange(number_of_samples):
                for word_pos in xrange(len(self._word_idss[doc_id])):
                    word_index = self._word_idss[doc_id][word_pos];
                    
                    phi_sum -= phi[:, word_pos][:, numpy.newaxis];
                    
                    # this is to get rid of the underflow error from the above summation, ideally, phi will become all integers after few iterations
                    phi_sum *= (phi_sum > 0);
                    #assert(numpy.all(phi_sum >= 0));

                    temp_phi = (phi_sum.T + self._alpha_alpha) * exp_E_log_beta[:, [word_index]].T;
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

            self._gamma[doc_id, :] = self._alpha_alpha + phi_sum.T[0, :];
            #batch_document_topic_distribution[doc_id, :] = self._alpha_alpha + phi_sum.T[0, :];
            
            document_phi /= (number_of_samples - burn_in_samples);
            document_phi += 1e-100;
            likelihood_phi += numpy.sum(document_phi * (self._E_log_beta[:, self._word_idss[doc_id]].T - numpy.log(document_phi)));
            
            if (doc_id+1) % 1000==0:
                print "successfully processed %d documents in hybrid mode..." % (doc_id+1);

        phi_sufficient_statistics /= (number_of_samples - burn_in_samples);

        return phi_sufficient_statistics, likelihood_phi

    '''
    def export_beta(self, exp_beta_path, top_display=-1):
        output = open(exp_beta_path, 'w');
        for topic_index in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (topic_index));
            
            beta_probability = numpy.exp(self._E_log_beta[topic_index, :] - scipy.misc.logsumexp(self._E_log_beta[topic_index, :]));

            i = 0;
            for type_index in reversed(numpy.argsort(beta_probability)):
                i += 1;
                output.write("%s\t%g\n" % (self._index_to_type[type_index], beta_probability[type_index]));
                if top_display > 0 and i >= top_display:
                    break;
                
        output.close();
    '''
    
if __name__ == "__main__":
    print "not implemented..."