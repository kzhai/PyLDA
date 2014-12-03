"""
VariationalBayes for Vanilla LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy;
import scipy.misc;
import nltk;
import string;

from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer;

"""
This is a python implementation of vanilla lda, based on variational inference, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.

References:
[1] D. Blei, A. Ng, and M. Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3:993-1022, January 2003.
"""

def parse_data(corpus, vocab):
    doc_count = 0
    
    word_ids = [];
    word_cts = [];
            
    for document_line in corpus:
        #words = document_line.split();
        document_word_dict = []
        for token in document_line.split():
            if token in vocab:
                if token not in document_word_dict:
                    document_word_dict[token] = 0;
                document_word_dict[token] += 1;
            else:
                continue;
            
        word_ids.append(numpy.array(document_word_dict.keys()));
        word_cts.append(numpy.array(document_word_dict.values()));
        
        doc_count+=1
        if doc_count%10000==0:
            print "successfully import %d documents..." % doc_count;
    
    print "successfully import %d documents..." % (doc_count);
    
    return word_ids, word_cts

class VariationalBayes(Inferencer):
    """
    """
    def __init__(self,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 gamma_converge_threshold=0.000001,
                 gamma_maximum_iteration=100,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001
                 ):
        self._alpha_update_decay_factor = alpha_update_decay_factor;
        self._alpha_maximum_decay = alpha_maximum_decay;
        self._alpha_converge_threshold = alpha_converge_threshold;
        self._alpha_maximum_iteration = alpha_maximum_iteration;
        
        self._gamma_maximum_iteration = gamma_maximum_iteration;
        self._gamma_converge_threshold = gamma_converge_threshold;

        #self._model_likelihood_threshold = model_likelihood_threshold;

    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, corpus, vocab, number_of_topics, alpha_alpha, alpha_beta):
        Inferencer._initialize(self, vocab, number_of_topics, alpha_alpha, alpha_beta);
        
        self._corpus = corpus;
        self.parse_data();
        
        # define the total number of document
        self._number_of_documents = len(self._word_ids);

        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha + 1.0 * self._number_of_types / self._number_of_topics;

        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._E_log_beta = compute_dirichlet_expectation(numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types)));

    def parse_data(self):
        doc_count = 0
        
        self._word_ids = [];
        self._word_cts = [];
                
        for document_line in self._corpus:
            #words = document_line.split();
            document_word_dict = {}
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue;
                
                type_id = self._type_to_index[token];
                if type_id not in document_word_dict:
                    document_word_dict[type_id] = 0;
                document_word_dict[type_id] += 1;
                
            self._word_ids.append(numpy.array(document_word_dict.keys()));
            self._word_cts.append(numpy.array(document_word_dict.values())[numpy.newaxis, :]);
            
            doc_count+=1
            if doc_count%10000==0:
                print "successfully parse %d documents..." % doc_count;
        
        assert len(self._word_ids)==len(self._word_cts);
        print "successfully parse %d documents..." % (doc_count);
        
    def e_step(self):
        likelihood_phi = 0.0;

        # initialize a V-by-K matrix phi contribution
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_types));
        
        # iterate over all documents
        for doc_id in xrange(self._number_of_documents):
            # compute the total number of words
            #total_word_count = self._corpus[doc_id].N()
            total_word_count = numpy.sum(self._word_cts[doc_id]);

            # initialize gamma for this document
            self._gamma[[doc_id], :] = self._alpha_alpha + 1.0 * total_word_count / self._number_of_topics;
            
            #term_ids = numpy.array(self._corpus[doc_id].keys());
            #term_counts = numpy.array([self._corpus[doc_id].values()]);
            term_ids = self._word_ids[doc_id];
            term_counts = self._word_cts[doc_id];
            assert term_counts.shape == (1, len(term_ids));

            # update phi and gamma until gamma converges
            for gamma_iteration in xrange(self._gamma_maximum_iteration):
                assert self._E_log_beta.shape==(self._number_of_topics, self._number_of_types);
                #log_phi = self._E_log_beta[:, term_ids].T + numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]), (len(self._corpus[doc_id]), 1));
                log_phi = self._E_log_beta[:, term_ids].T + numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]), (self._word_ids[doc_id].shape[0], 1));
                assert log_phi.shape==(len(term_ids), self._number_of_topics);
                phi_normalizer = numpy.log(numpy.sum(numpy.exp(log_phi), axis=1)[:, numpy.newaxis]);
                assert(phi_normalizer.shape == (len(term_ids), 1));
                log_phi -= phi_normalizer;
                assert(log_phi.shape == (len(term_ids), self._number_of_topics));
                
                gamma_update = self._alpha_alpha + numpy.array(numpy.sum(numpy.exp(log_phi + numpy.log(term_counts.transpose())), axis=0));
                
                mean_change = numpy.mean(abs(gamma_update - self._gamma[doc_id, :]));
                self._gamma[[doc_id], :] = gamma_update;
                if mean_change <= self._gamma_converge_threshold:
                    break;
                
            likelihood_phi += numpy.sum(numpy.exp(log_phi) * ((self._E_log_beta[:, term_ids].T * term_counts.transpose()) - log_phi));
            assert(log_phi.shape == (len(term_ids), self._number_of_topics));
            phi_sufficient_statistics[:, term_ids] += numpy.exp(log_phi.T);
        
            if (doc_id+1) % 1000==0:
                print "successfully processed %d documents..." % (doc_id+1);
            
        return phi_sufficient_statistics, likelihood_phi

    def m_step(self, phi_sufficient_statistics):
        self._E_log_beta = compute_dirichlet_expectation(phi_sufficient_statistics+self._alpha_beta);
        assert(self._E_log_beta.shape == (self._number_of_topics, self._number_of_types));
        
        # compute the sufficient statistics for alpha and update
        alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
        self.update_alpha(alpha_sufficient_statistics)

    """
    """
    def learning(self):
        self._counter += 1;
        
        clock_e_step = time.time();        
        phi_sufficient_statistics, likelihood_phi = self.e_step();
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();        
        self.m_step(phi_sufficient_statistics);
        clock_m_step = time.time() - clock_m_step;
                
        # compute the log-likelihood of alpha terms
        alpha_sum = numpy.sum(self._alpha_alpha, axis=1);
        likelihood_alpha = -numpy.sum(scipy.special.gammaln(self._alpha_alpha), axis=1);
        likelihood_alpha += scipy.special.gammaln(alpha_sum);
        likelihood_alpha *= self._number_of_documents;
        
        likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
        likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));

        new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
        
        print "e_step and m_step of iteration %d finished in %d and %d seconds respectively with log likelihood %g" % (self._counter, clock_e_step, clock_m_step, new_likelihood)
        
        #if abs((new_likelihood - old_likelihood) / old_likelihood) < self._model_likelihood_threshold:
            #print "model likelihood converged..."
            #break
        #old_likelihood = new_likelihood;
        
        return new_likelihood

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """
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
        
if __name__ == "__main__":
    print "not implemented..."