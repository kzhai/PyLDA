"""VariationalBayes
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import math, random;
import numpy, scipy;

"""
This is a python implementation of lda, based on variational inference, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.

References:
[1] D. Blei, A. Ng, and M. Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3:993-1022, January 2003.
"""
class VariationalBayes(object):
    """
    """
    def __init__(self, alpha_update_decay_factor=0.9, 
                 alpha_maximum_decay=10, 
                 gamma_converge_threshold=0.000001, 
                 gamma_maximum_iteration=100, 
                 alpha_converge_threshold = 0.000001, 
                 alpha_maximum_iteration = 100, 
                 converge_threshold = 0.00001,
                 maximum_iteration = 100,
                 snapshot_interval = 10):
        # initialize the iteration parameters
        self._alpha_update_decay_factor = alpha_update_decay_factor
        self._alpha_maximum_decay = alpha_maximum_decay
        self._alpha_converge_threshold = alpha_converge_threshold
        self._alpha_maximum_iteration = alpha_maximum_iteration
        
        self._gamma_converge_threshold = gamma_converge_threshold
        self._gamma_maximum_iteration = gamma_maximum_iteration
        
        self._maximum_iteration = maximum_iteration
        self._converge_threshold = converge_threshold
        
        self._gamma_title = "gamma-";
        self._beta_title = "beta-";
        
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, num_topics=10):
        # initialize the total number of topics.
        self._K = num_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        self._alpha = numpy.random.random((1, self._K)) / self._K;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        from util.input_parser import dict_list_2_dict_freqdist
        data = dict_list_2_dict_freqdist(data);
        self._data = data
        
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._data)
        
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = []
        for token_list in data.values():
            self._vocab += token_list
        self._vocab = list(set(self._vocab))
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._V = len(self._vocab)
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = numpy.tile(self._alpha + 1.0*self._V/self._K, (self._D, 1));
        
        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._beta = 1.0/self._V + numpy.random.random((self._V, self._K));
        self._beta = self._beta / numpy.sum(self._beta, axis=0)[numpy.newaxis, :];
        self._beta = numpy.log(self._beta);

    """
    """
    def update_phi(self, doc_id, phi_table):
        # update phi and gamma until gamma converges
        for gamma_iteration in xrange(self._gamma_maximum_iteration):
            term_ids = numpy.array(self._data[doc_id].keys());
            term_counts = numpy.array([self._data[doc_id].values()]);
            assert(term_counts.shape==(1, len(term_ids)));
            
            phi_contribution = self._beta[term_ids, :] + scipy.special.psi(self._gamma[[doc_id], :]);
            phi_normalizer = numpy.log(numpy.sum(numpy.exp(phi_contribution), axis=1)[:, numpy.newaxis]);
            assert(phi_normalizer.shape==(len(term_ids), 1));
            phi_contribution -= phi_normalizer;
            
            assert(phi_contribution.shape==(len(term_ids), self._K));
            phi_contribution += numpy.log(term_counts.transpose());
            
            gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(phi_contribution), axis=0));
            mean_change = numpy.mean(abs(gamma_update - self._gamma[doc_id, :]));
            self._gamma[[doc_id], :] = gamma_update;
            if mean_change<=self._gamma_converge_threshold:
                break;
            
        likelihood_phi = numpy.sum(numpy.exp(phi_contribution) * ((self._beta[term_ids, :] * term_counts.transpose()) - phi_contribution));
        assert(phi_contribution.shape==(len(term_ids), self._K));
        phi_table[[term_ids], :] += numpy.exp(phi_contribution);
                
        return phi_table, likelihood_phi

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    @attention: alpha_sufficient_statistics value will not be modified, however, alpha_vector will be updated during this function
    """
    def update_alpha(self, alpha_sufficient_statistics):
        assert(alpha_sufficient_statistics.shape==(1, self._K));        
        alpha_update = self._alpha;
        
        decay = 0;
        for alpha_iteration in xrange(self._alpha_maximum_iteration):
            alpha_sum = numpy.sum(self._alpha);
            alpha_gradient = self._D * (scipy.special.psi(alpha_sum) - scipy.special.psi(self._alpha)) + alpha_sufficient_statistics;
            alpha_hessian = -self._D * scipy.special.polygamma(1, self._alpha);

            if numpy.any(numpy.isinf(alpha_gradient)) or numpy.any(numpy.isnan(alpha_gradient)):
                print "illegal alpha gradient vector", alpha_gradient

            sum_g_h = numpy.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = self._D * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the alpha vector
            while True:
                singular_hessian = False

                step_size = numpy.power(self._alpha_update_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                assert(self._alpha.shape==step_size.shape);
                
                if numpy.any(self._alpha >= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self._alpha - step_size;
                
                if singular_hessian:
                    decay += 1;
                    if decay > self._alpha_maximum_decay:
                        break;
                else:
                    break;
                
            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - self._alpha));
            self._alpha = alpha_update;
            if mean_change<=self._alpha_converge_threshold:
                break;

        return

    """
    """
    def learning(self, iteration=0, directory="../../output/tmp-output"):
        if iteration<=0:
            iteration = self._maximum_iteration;
        
        old_likelihood = 1.0;
        for i in xrange(iteration):
            likelihood_phi = 0.0;

            # initialize a V-by-K matrix phi contribution
            phi_table = numpy.zeros((self._V, self._K));
            
            # iterate over all documents
            for doc in self._data.keys():
                
                # compute the total number of words
                total_word_count = self._data[doc].N()
    
                # initialize gamma for this document
                self._gamma[[doc], :] = self._alpha + 1.0 * total_word_count/self._K;
                
                # iterate till convergence
                phi_table, likelihood_phi_temp = self.update_phi(doc, phi_table);
                likelihood_phi+=likelihood_phi_temp;
            
            self._beta = phi_table / numpy.sum(phi_table, axis=0)[numpy.newaxis, :];
            assert(self._beta.shape==(self._V, self._K));
            self._beta = numpy.log(self._beta);
            
            # compute the log-likelihood of alpha terms
            alpha_sum = numpy.sum(self._alpha, axis=1);
            likelihood_alpha = numpy.sum(scipy.special.gammaln(self._alpha), axis=1);
            likelihood_alpha += scipy.special.gammaln(alpha_sum);
            likelihood_alpha += self._D;
            
            likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
            likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));
    
            new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
            print "em iteration is ", (i+1), " likelihood is ", new_likelihood
            
            if abs((new_likelihood - old_likelihood)/old_likelihood) < self._converge_threshold:
                break
            
            old_likelihood = new_likelihood;
            
            alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
            alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
            self.update_alpha(alpha_sufficient_statistics)
            #print "alpha vector is ", self._alpha
            
        print "learning finished..."

    """
    """
    def print_topics(self, term_mapping):
        output = open(term_mapping);
        vocab = {};
        i=0;
        for line in output:
            vocab[i] = line.strip();
            i+=1;

        display = self._beta > -numpy.log(self._V);
        assert(display.shape==(self._V, self._K));
        for k in xrange(self._K):
            output_str = str(k) + ": ";
            for v in xrange(self._V):
                if display[v, k]:
                    output_str += vocab[v] + "\t";
            print output_str
                    
if __name__ == "__main__":
    temp_directory = "../../data/test/";
    from util.input_parser import import_monolingual_data;
    d = import_monolingual_data(temp_directory+"doc.dat");
    
    lda = VariationalBayes();
    lda._initialize(d, 3);
    lda.learning(10);
    lda.print_topics(temp_directory+"voc.dat");