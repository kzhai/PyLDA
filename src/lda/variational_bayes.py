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
                 hyper_parameter_optimize_interval=1,
                 
                 #hyper_parameter_iteration=100,
                 #hyper_parameter_decay_factor=0.9,
                 #hyper_parameter_maximum_decay=10,
                 #hyper_parameter_converge_threshold=1e-6,
                 
                 #model_converge_threshold=1e-6
                 ):
        Inferencer.__init__(self, hyper_parameter_optimize_interval);
        
        #self._hyper_parameter_iteration = hyper_parameter_iteration
        #self._hyper_parameter_decay_factor = hyper_parameter_decay_factor;
        #self._hyper_parameter_maximum_decay = hyper_parameter_maximum_decay;
        #self._hyper_parameter_converge_threshold = hyper_parameter_converge_threshold;
        
        #self._model_converge_threshold = model_converge_threshold;

    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, corpus, vocab, number_of_topics, alpha_alpha, alpha_beta):
        Inferencer._initialize(self, vocab, number_of_topics, alpha_alpha, alpha_beta);

        self._corpus = corpus;
        self._parsed_corpus = self.parse_data();
        
        # define the total number of document
        self._number_of_documents = len(self._parsed_corpus[0]);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;

        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._E_log_beta = compute_dirichlet_expectation(numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types)));

    def parse_data(self, corpus=None):
        if corpus==None:
            corpus=self._corpus;
            
        doc_count = 0
        
        word_ids = [];
        word_cts = [];
                
        for document_line in corpus:
            #words = document_line.split();
            document_word_dict = {}
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue;
                
                type_id = self._type_to_index[token];
                if type_id not in document_word_dict:
                    document_word_dict[type_id] = 0;
                document_word_dict[type_id] += 1;
                
            word_ids.append(numpy.array(document_word_dict.keys()));
            word_cts.append(numpy.array(document_word_dict.values())[numpy.newaxis, :]);
            
            doc_count+=1
            if doc_count%10000==0:
                print "successfully parse %d documents..." % doc_count;
        
        assert len(word_ids)==len(word_cts);
        print "successfully parse %d documents..." % (doc_count);
        
        return (word_ids, word_cts)
        
    def e_step(self, parsed_corpus=None, local_parameter_iteration=50, local_parameter_converge_threshold=1e-6):
        if parsed_corpus==None:
            word_ids = self._parsed_corpus[0];
            word_cts = self._parsed_corpus[1];
        else:
            word_ids = parsed_corpus[0]
            word_cts = parsed_corpus[1];
        
        assert len(word_ids)==len(word_cts);
        number_of_documents = len(word_ids);
        
        likelihood_phi = 0.0;

        # initialize a V-by-K matrix phi sufficient statistics
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_types));
        
        # initialize a D-by-K matrix gamma values
        gamma_values = numpy.zeros((number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;
        
        # iterate over all documents
        for doc_id in xrange(number_of_documents):
            # compute the total number of words
            #total_word_count = self._corpus[doc_id].N()
            total_word_count = numpy.sum(word_cts[doc_id]);

            # initialize gamma for this document
            gamma_values[doc_id, :] = self._alpha_alpha + 1.0 * total_word_count / self._number_of_topics;
            
            #term_ids = numpy.array(self._corpus[doc_id].keys());
            #term_counts = numpy.array([self._corpus[doc_id].values()]);
            term_ids = word_ids[doc_id];
            term_counts = word_cts[doc_id];
            assert term_counts.shape == (1, len(term_ids));

            # update phi and gamma until gamma converges
            for gamma_iteration in xrange(local_parameter_iteration):
                assert self._E_log_beta.shape==(self._number_of_topics, self._number_of_types);
                #log_phi = self._E_log_beta[:, term_ids].T + numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]), (len(self._corpus[doc_id]), 1));
                log_phi = self._E_log_beta[:, term_ids].T + numpy.tile(scipy.special.psi(gamma_values[[doc_id], :]), (word_ids[doc_id].shape[0], 1));
                assert log_phi.shape==(len(term_ids), self._number_of_topics);
                phi_normalizer = numpy.log(numpy.sum(numpy.exp(log_phi), axis=1)[:, numpy.newaxis]);
                assert(phi_normalizer.shape == (len(term_ids), 1));
                log_phi -= phi_normalizer;
                assert(log_phi.shape == (len(term_ids), self._number_of_topics));
                
                gamma_update = self._alpha_alpha + numpy.array(numpy.sum(numpy.exp(log_phi + numpy.log(term_counts.transpose())), axis=0));
                
                mean_change = numpy.mean(abs(gamma_update - gamma_values[doc_id, :]));
                gamma_values[doc_id, :] = gamma_update;
                if mean_change <= local_parameter_converge_threshold:
                    break;
                
            likelihood_phi += numpy.sum(numpy.exp(log_phi) * ((self._E_log_beta[:, term_ids].T * term_counts.transpose()) - log_phi));
            assert(log_phi.shape == (len(term_ids), self._number_of_topics));
            phi_sufficient_statistics[:, term_ids] += numpy.exp(log_phi.T);
            
            if (doc_id+1) % 1000==0:
                print "successfully processed %d documents..." % (doc_id+1);
            
        return gamma_values, phi_sufficient_statistics, likelihood_phi

    def m_step(self, phi_sufficient_statistics):
        self._E_log_beta = compute_dirichlet_expectation(phi_sufficient_statistics+self._alpha_beta);
        assert(self._E_log_beta.shape == (self._number_of_topics, self._number_of_types));
        
        # compute the sufficient statistics for alpha and update
        alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0);#[numpy.newaxis, :];
        
        return alpha_sufficient_statistics

    """
    """
    def learning(self):
        self._counter += 1;
        
        clock_e_step = time.time();        
        gamma_values, phi_sufficient_statistics, likelihood_phi = self.e_step();
        self._gamma = gamma_values;
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();        
        alpha_sufficient_statistics = self.m_step(phi_sufficient_statistics);
        if self._hyper_parameter_optimize_interval>0 and self._counter%self._hyper_parameter_optimize_interval==0:
            self.optimize_hyperparameters(alpha_sufficient_statistics);
        clock_m_step = time.time() - clock_m_step;
                
        # compute the log-likelihood of alpha terms
        alpha_sum = numpy.sum(self._alpha_alpha);
        likelihood_alpha = -numpy.sum(scipy.special.gammaln(self._alpha_alpha));
        likelihood_alpha += scipy.special.gammaln(alpha_sum);
        likelihood_alpha *= self._number_of_documents;
        
        likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
        likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));

        new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
        
        print "e_step and m_step of iteration %d finished in %d and %d seconds respectively with log likelihood %g" % (self._counter, clock_e_step, clock_m_step, new_likelihood)
        
        #if abs((new_likelihood - old_likelihood) / old_likelihood) < self._model_converge_threshold:
            #print "model likelihood converged..."
            #break
        #old_likelihood = new_likelihood;
        
        return new_likelihood

    def inference(self, corpus):
        parsed_corpus = self.parse_data(corpus);
        number_of_documents = len(parsed_corpus[0]);
        
        clock_e_step = time.time();        
        gamma_values, phi_sufficient_statistics, likelihood_phi = self.e_step(parsed_corpus);
        clock_e_step = time.time() - clock_e_step;
        
        # compute the log-likelihood of alpha terms
        alpha_sum = numpy.sum(self._alpha_alpha);
        likelihood_alpha = -numpy.sum(scipy.special.gammaln(self._alpha_alpha));
        likelihood_alpha += scipy.special.gammaln(alpha_sum);
        likelihood_alpha *= number_of_documents;
        
        likelihood_gamma = numpy.sum(scipy.special.gammaln(gamma_values));
        likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(gamma_values, axis=1)));

        new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
        
        #print "e_step finished in %d with log likelihood %g" % (clock_e_step, new_likelihood)
        
        #if abs((new_likelihood - old_likelihood) / old_likelihood) < self._model_converge_threshold:
            #print "model likelihood converged..."
            #break
        #old_likelihood = new_likelihood;
        
        return new_likelihood, gamma_values

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """
    def optimize_hyperparameters(self, alpha_sufficient_statistics, hyper_parameter_iteration=100, hyper_parameter_decay_factor=0.9, hyper_parameter_maximum_decay=10, hyper_parameter_converge_threshold=1e-6):
        #assert(alpha_sufficient_statistics.shape == (1, self._number_of_topics));
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