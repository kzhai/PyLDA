"""
UncollapsedVariationalBayes for Vanilla LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import math, random;
import numpy, scipy;
from lda.vb import VariationalBayes;

"""
This is a python implementation of vanilla lda, based on variational inference, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.

References:
[1] D. Blei, A. Ng, and M. Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3:993-1022, January 2003.
"""
class UncollapsedVariationalBayes(VariationalBayes):
    """
    """
    def __init__(self, alpha_update_decay_factor=0.9, 
                 alpha_maximum_decay=10, 
                 gamma_converge_threshold=0.000001, 
                 gamma_maximum_iteration=100, 
                 alpha_converge_threshold = 0.000001, 
                 alpha_maximum_iteration = 100, 
                 model_likelihood_threshold = 0.00001,
                 global_maximum_iteration = 100,
                 snapshot_interval = 10):
        super(UncollapsedVariationalBayes, self).__init__(alpha_update_decay_factor, 
                                                          alpha_maximum_decay, 
                                                          gamma_converge_threshold, 
                                                          gamma_maximum_iteration, 
                                                          alpha_converge_threshold, 
                                                          alpha_maximum_iteration, 
                                                          global_maximum_iteration,
                                                          snapshot_interval);
        self._model_likelihood_threshold = model_likelihood_threshold;

    
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, num_topics=10):
        super(UncollapsedVariationalBayes, self)._initialize(data, num_topics);

    """
    @deprecated: no longer useful
    """
    def update_phi(self, doc_id, phi_table):
        # update phi and gamma until gamma converges
        for gamma_iteration in xrange(self._gamma_maximum_iteration):
            term_ids = numpy.array(self._data[doc_id].keys());
            term_counts = numpy.array([self._data[doc_id].values()]);
            assert(term_counts.shape==(1, len(term_ids)));
            
            phi_contribution = self._log_beta[term_ids, :] + scipy.special.psi(self._gamma[[doc_id], :]);
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
            
        likelihood_phi = numpy.sum(numpy.exp(phi_contribution) * ((self._log_beta[term_ids, :] * term_counts.transpose()) - phi_contribution));
        assert(phi_contribution.shape==(len(term_ids), self._K));
        phi_table[[term_ids], :] += numpy.exp(phi_contribution);

        return phi_table, likelihood_phi

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """
    def update_alpha(self, alpha_sufficient_statistics):
        super(UncollapsedVariationalBayes, self).update_alpha(alpha_sufficient_statistics, 1);

        return

    """
    """
    def learning(self, iteration=0, directory="../../output/tmp-output"):
        if iteration<=0:
            iteration = self._global_maximum_iteration;
        
        old_likelihood = 1.0;
        for i in xrange(iteration):
            likelihood_phi = 0.0;

            # initialize a V-by-K matrix phi contribution
            phi_table = numpy.zeros((self._V, self._K));
            
            # iterate over all documents
            for doc_id in self._data.keys():
                
                # compute the total number of words
                #total_word_count = self._data[doc_id].N()
    
                # initialize gamma for this document
                #self._gamma[[doc_id], :] = self._alpha + 1.0 * total_word_count/self._K;
                
                # iterate till convergence
                #phi_table, likelihood_phi_temp = self.update_phi(doc_id, phi_table);
                #likelihood_phi+=likelihood_phi_temp;

                term_ids = numpy.array(self._data[doc_id].keys());
                term_counts = numpy.array([self._data[doc_id].values()]);
                assert(term_counts.shape==(1, len(term_ids)));

                # update phi and gamma until gamma converges
                for gamma_iteration in xrange(self._gamma_maximum_iteration):
                    phi_contribution = self._log_beta[term_ids, :] + scipy.special.psi(self._gamma[[doc_id], :]);
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
                    
                likelihood_phi += numpy.sum(numpy.exp(phi_contribution) * ((self._log_beta[term_ids, :] * term_counts.transpose()) - phi_contribution));
                assert(phi_contribution.shape==(len(term_ids), self._K));
                phi_table[[term_ids], :] += numpy.exp(phi_contribution);
                
            self._log_beta = phi_table / numpy.sum(phi_table, axis=0)[numpy.newaxis, :];
            assert(self._log_beta.shape==(self._V, self._K));
            self._log_beta = numpy.log(self._log_beta);
            
            # compute the log-likelihood of alpha terms
            alpha_sum = numpy.sum(self._alpha, axis=1);
            likelihood_alpha = numpy.sum(scipy.special.gammaln(self._alpha), axis=1);
            likelihood_alpha += scipy.special.gammaln(alpha_sum);
            likelihood_alpha += self._D;
            
            likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
            likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));
    
            new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
            print "em iteration is ", (i+1), " likelihood is ", new_likelihood
            
            if abs((new_likelihood - old_likelihood)/old_likelihood) < self._model_likelihood_threshold:
                break
            
            old_likelihood = new_likelihood;
            
            alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
            alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
            self.update_alpha(alpha_sufficient_statistics)
            print "alpha vector is ", self._alpha
            
        print "learning finished..."
                    
if __name__ == "__main__":
    temp_directory = "../../data/test/";
    from util.input_parser import import_monolingual_data;
    d = import_monolingual_data(temp_directory+"doc.dat");
    
    lda = UncollapsedVariationalBayes();
    lda._initialize(d, 3);
    lda.learning(10);
    lda.print_topics(temp_directory+"voc.dat");