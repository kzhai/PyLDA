"""
UncollapsedVariationalBayes for Vanilla LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

from lda.vb import VariationalBayes
import numpy
import scipy

"""
This is a python implementation of vanilla lda, based on variational inference, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.

References:
[1] D. Blei, A. Ng, and M. Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3:993-1022, January 2003.
"""
class UncollapsedVariationalBayes(VariationalBayes):
    """
    """
    def __init__(self,
                 snapshot_interval=10,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 gamma_converge_threshold=0.000001,
                 gamma_maximum_iteration=100,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001,
                 global_maximum_iteration=100):
        super(UncollapsedVariationalBayes, self).__init__(snapshot_interval,
                                                          alpha_update_decay_factor,
                                                          alpha_maximum_decay,
                                                          gamma_converge_threshold,
                                                          gamma_maximum_iteration,
                                                          alpha_converge_threshold,
                                                          alpha_maximum_iteration,
                                                          global_maximum_iteration);

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
            assert(term_counts.shape == (1, len(term_ids)));
            
            phi_contribution = self._E_log_beta[term_ids, :] + scipy.special.psi(self._gamma[[doc_id], :]);
            phi_normalizer = numpy.log(numpy.sum(numpy.exp(phi_contribution), axis=1)[:, numpy.newaxis]);
            assert(phi_normalizer.shape == (len(term_ids), 1));
            phi_contribution -= phi_normalizer;
            
            assert(phi_contribution.shape == (len(term_ids), self._K));
            phi_contribution += numpy.log(term_counts.transpose());
            
            gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(phi_contribution), axis=0));
            mean_change = numpy.mean(abs(gamma_update - self._gamma[doc_id, :]));
            self._gamma[[doc_id], :] = gamma_update;
            if mean_change <= self._gamma_converge_threshold:
                break;
            
        likelihood_phi = numpy.sum(numpy.exp(phi_contribution) * ((self._E_log_beta[term_ids, :] * term_counts.transpose()) - phi_contribution));
        assert(phi_contribution.shape == (len(term_ids), self._K));
        phi_table[[term_ids], :] += numpy.exp(phi_contribution);

        return phi_table, likelihood_phi

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """
    def update_alpha(self, alpha_sufficient_statistics):
        super(UncollapsedVariationalBayes, self).update_alpha(alpha_sufficient_statistics, 1);

        return

    def e_step(self):
        likelihood_phi = 0.0;

        # initialize a V-by-K matrix phi contribution
        phi_sufficient_statistics = numpy.zeros((self._V, self._K));
        
        # iterate over all documents
        for doc_id in self._data.keys():
            # compute the total number of words
            total_word_count = self._data[doc_id].N()

            # initialize gamma for this document
            self._gamma[[doc_id], :] = self._alpha + 1.0 * total_word_count / self._K;
            
            term_ids = numpy.array(self._data[doc_id].keys());
            term_counts = numpy.array([self._data[doc_id].values()]);
            assert(term_counts.shape == (1, len(term_ids)));

            # update phi and gamma until gamma converges
            for gamma_iteration in xrange(self._gamma_maximum_iteration):
                #_E_log_theta = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, 1))[:, numpy.newaxis];
                log_phi = self._E_log_beta[term_ids, :] + scipy.special.psi(self._gamma[[doc_id], :]);
                phi_normalizer = numpy.log(numpy.sum(numpy.exp(log_phi), axis=1)[:, numpy.newaxis]);
                assert(phi_normalizer.shape == (len(term_ids), 1));
                log_phi -= phi_normalizer;
                
                assert(log_phi.shape == (len(term_ids), self._K));
                
                #log_phi += numpy.log(term_counts.transpose());
                #gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(log_phi), axis=0));

                gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(log_phi + numpy.log(term_counts.transpose())), axis=0));
                
                mean_change = numpy.mean(abs(gamma_update - self._gamma[doc_id, :]));
                self._gamma[[doc_id], :] = gamma_update;
                if mean_change <= self._gamma_converge_threshold:
                    break;
                
            likelihood_phi += numpy.sum(numpy.exp(log_phi) * ((self._E_log_beta[term_ids, :] * term_counts.transpose()) - log_phi));
            assert(log_phi.shape == (len(term_ids), self._K));
            phi_sufficient_statistics[[term_ids], :] += numpy.exp(log_phi);
            
        return phi_sufficient_statistics, likelihood_phi

    def m_step(self, phi_sufficient_statistics):
        phi_sufficient_statistics += self._eta;
        self._E_log_beta = scipy.special.psi(phi_sufficient_statistics) - scipy.special.psi(numpy.sum(phi_sufficient_statistics, 0))[numpy.newaxis, :];
        assert(self._E_log_beta.shape == (self._V, self._K));

    """
    """
    def learning(self, iteration=0, directory="../../output/tmp-output"):
        if iteration <= 0:
            iteration = self._global_maximum_iteration;
        
        old_likelihood = 1.0;
        for i in xrange(iteration):
            '''
            likelihood_phi = 0.0;

            # initialize a V-by-K matrix phi contribution
            phi_table = numpy.zeros((self._V, self._K)) + self._eta;
            
            # iterate over all documents
            for doc_id in self._data.keys():
                # compute the total number of words
                total_word_count = self._data[doc_id].N()
    
                # initialize gamma for this document
                self._gamma[[doc_id], :] = self._alpha + 1.0 * total_word_count / self._K;
                
                term_ids = numpy.array(self._data[doc_id].keys());
                term_counts = numpy.array([self._data[doc_id].values()]);
                assert(term_counts.shape == (1, len(term_ids)));

                # update phi and gamma until gamma converges
                for gamma_iteration in xrange(self._gamma_maximum_iteration):
                    log_phi = self._E_log_beta[term_ids, :] + scipy.special.psi(self._gamma[[doc_id], :]);
                    phi_normalizer = numpy.log(numpy.sum(numpy.exp(log_phi), axis=1)[:, numpy.newaxis]);
                    assert(phi_normalizer.shape == (len(term_ids), 1));
                    log_phi -= phi_normalizer;
                    
                    assert(log_phi.shape == (len(term_ids), self._K));
                    
                    #log_phi += numpy.log(term_counts.transpose());
                    #gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(log_phi), axis=0));

                    gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(log_phi + numpy.log(term_counts.transpose())), axis=0));
                    
                    mean_change = numpy.mean(abs(gamma_update - self._gamma[doc_id, :]));
                    self._gamma[[doc_id], :] = gamma_update;
                    if mean_change <= self._gamma_converge_threshold:
                        break;
                    
                likelihood_phi += numpy.sum(numpy.exp(log_phi) * ((self._E_log_beta[term_ids, :] * term_counts.transpose()) - log_phi));
                assert(log_phi.shape == (len(term_ids), self._K));
                phi_table[[term_ids], :] += numpy.exp(log_phi);
                
            if numpy.isnan(likelihood_phi):
                break;
                
            self._E_log_beta = phi_table / numpy.sum(phi_table, axis=0)[numpy.newaxis, :];
            if self._truncate_beta:
                # truncate beta to the minimum value in the beta matrix
                self._E_log_beta[numpy.nonzero(self._E_log_beta <= 2. * numpy.mean(self._E_log_beta))] = numpy.min(self._E_log_beta);
            assert(self._E_log_beta.shape == (self._V, self._K));
            self._E_log_beta = numpy.log(self._E_log_beta);
            '''
            
            phi_sufficient_statistics, likelihood_phi = self.e_step();
            self.m_step(phi_sufficient_statistics);
            
            # compute the log-likelihood of alpha terms
            alpha_sum = numpy.sum(self._alpha, axis=1);
            likelihood_alpha = -numpy.sum(scipy.special.gammaln(self._alpha), axis=1);
            likelihood_alpha += scipy.special.gammaln(alpha_sum);
            likelihood_alpha *= self._D;
            
            likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
            likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));
    
            new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
            print "em iteration is ", (i + 1), " likelihood is ", new_likelihood
            
            if abs((new_likelihood - old_likelihood) / old_likelihood) < self._model_likelihood_threshold:
                print "model likelihood converged..."
                break
            
            old_likelihood = new_likelihood;
            
            # compute the sufficient statistics for alpha and update
            alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
            alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
            self.update_alpha(alpha_sufficient_statistics)
            print "alpha vector is ", self._alpha
            
        print "learning finished..."
                    
if __name__ == "__main__":
    temp_directory = "../../data/de-news/en/corpus-2/";
    number_of_topics = 5;
    number_of_iterations = 20;
    
    import sys
    if (len(sys.argv) > 1):
        temp_directory = sys.argv[1];
        number_of_topics = int(sys.argv[2]);
        number_of_iterations = int(sys.argv[3]);
        
    from util.input_parser import import_monolingual_data;
    d = import_monolingual_data(temp_directory + "doc.dat");

    lda = UncollapsedVariationalBayes(5, False);
    lda._initialize(d, number_of_topics);
    lda.learning(number_of_iterations);
    print lda._E_log_beta
    lda.print_topics(temp_directory + "voc.dat", 5);
