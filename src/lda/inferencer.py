"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy
import nltk;



"""
This is a python implementation of vanilla lda, based on a lda approach of variational inference and gibbs sampling, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.
"""

def compute_dirichlet_expectation(dirichlet_parameter):
    if (len(dirichlet_parameter.shape) == 1):
        return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
    return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:, numpy.newaxis]

def parse_vocabulary(vocab):
    type_to_index = {};
    index_to_type = {};
    for word in set(vocab):
        index_to_type[len(index_to_type)] = word;
        type_to_index[word] = len(type_to_index);
        
    return type_to_index, index_to_type;

class Inferencer():
    """
    """
    def __init__(self,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 #gamma_converge_threshold=0.000001,
                 number_of_samples=10,
                 burn_in_samples=5,
                 model_likelihood_threshold=0.00001
                 ):
        self._alpha_update_decay_factor = alpha_update_decay_factor;
        self._alpha_maximum_decay = alpha_maximum_decay;
        self._alpha_converge_threshold = alpha_converge_threshold;
        self._alpha_maximum_iteration = alpha_maximum_iteration;
        
        #self._gamma_converge_threshold = gamma_converge_threshold;
        self._number_of_samples = number_of_samples;
        self._burn_in_samples = burn_in_samples;

    """
    """
    def _initialize(self, vocab, number_of_topics, alpha_alpha, alpha_beta):
        self.parse_vocabulary(vocab);
        
        self._counter = 0;
                    
        # initialize the total number of topics.
        self._number_of_topics = number_of_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        #self._alpha_alpha = numpy.random.random((1, self._number_of_topics)) / self._number_of_topics;
        self._alpha_alpha = numpy.zeros((1, self._number_of_topics))+alpha_alpha;
        self._alpha_beta = alpha_beta;
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._number_of_types = len(self._type_to_index)

    def parse_vocabulary(self, vocab):
        self._type_to_index = {};
        self._index_to_type = {};
        for word in set(vocab):
            self._index_to_type[len(self._index_to_type)] = word;
            self._type_to_index[word] = len(self._type_to_index);
            
        self._vocab = self._type_to_index.keys();
        
    def parse_data(self):
        raise NotImplementedError;
    
    def e_step(self):
        # initialize a V-by-K matrix phi contribution
        sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_types));
        
        # Initialize the variational distribution q(theta|gamma) for the mini-batch
        #batch_document_topic_distribution = numpy.zeros((batchD, self._number_of_topics));

        # iterate over all documents
        for d in xrange(self._number_of_documents):
            phi = numpy.random.random((self._number_of_topics, len(self._corpus[d])));
            phi = phi / numpy.sum(phi, axis=0)[numpy.newaxis, :];
            phi_sum = numpy.sum(phi, axis=1)[:, numpy.newaxis];
            assert(phi_sum.shape == (self._number_of_topics, 1));

            # collect phi samples from empirical distribution
            for it in xrange(self._number_of_samples):
                for n in xrange(len(self._corpus[d])):
                    id = self._corpus[d][n];
                    
                    phi_sum -= phi[:, n][:, numpy.newaxis];
                    
                    # this is to get rid of the underflow error from the above summation, ideally, phi will become all integers after few iterations
                    phi_sum *= phi_sum > 0;
                    #assert(numpy.all(phi_sum >= 0));

                    temp_phi = (phi_sum.T + self._alpha_alpha) * self._exp_E_log_beta[:, [id]].T;
                    assert(temp_phi.shape == (1, self._number_of_topics));
                    temp_phi /= numpy.sum(temp_phi);

                    # sample a topic for this word
                    temp_phi = numpy.random.multinomial(1, temp_phi[0])[:, numpy.newaxis];
                    assert(temp_phi.shape == (self._number_of_topics, 1));
                    
                    phi[:, n][:, numpy.newaxis] = temp_phi;
                    phi_sum += temp_phi;

                    # discard the first few burn-in sweeps
                    if it < self._burn_in_samples:
                        continue;
                    
                    sufficient_statistics[:, id] += temp_phi[:, 0];

            self._gamma[d, :] = self._alpha_alpha + phi_sum.T[0, :];
            #batch_document_topic_distribution[d, :] = self._alpha_alpha + phi_sum.T[0, :];
            
            if (d+1) % 1000==0:
                print "successfully processed %d documents..." % (d+1);

        sufficient_statistics /= (self._number_of_samples - self._burn_in_samples);

        return sufficient_statistics

    def m_step(self, phi_sufficient_statistics):
        self._exp_E_log_beta = numpy.exp(self.compute_dirichlet_expectation(phi_sufficient_statistics+self._alpha_beta));
        assert(self._exp_E_log_beta.shape == (self._number_of_topics, self._number_of_types));
        
        # compute the sufficient statistics for alpha and update
        alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
        self.update_alpha(alpha_sufficient_statistics)

    """
    """
    def learning(self):
        raise NotImplementedError;

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
        raise NotImplementedError;
        
if __name__ == "__main__":
    raise NotImplementedError;