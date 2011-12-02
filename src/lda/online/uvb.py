"""
UncollapsedVariationalBayes for Online LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import numpy, scipy;
from lda.vb import VariationalBayes as VariationalBayes

"""
This is a python implementation of online lda, based on variational inference, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.

References:
[1] M. Hoffman, D. Blei, and F. Bach. Online Learning for Latent Dirichlet Allocation. Neural Information Processing Systems, 2010.
"""
class UncollapsedVariationalBayes(VariationalBayes):
    def __init__(self,
                 snapshot_interval=10,
                 truncate_beta=False,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 gamma_converge_threshold=0.000001,
                 gamma_maximum_iteration=100,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001,
                 global_maximum_iteration=100):
        super(UncollapsedVariationalBayes, self).__init__(snapshot_interval,
                                                          truncate_beta,
                                                          alpha_update_decay_factor,
                                                          alpha_maximum_decay,
                                                          gamma_converge_threshold,
                                                          gamma_maximum_iteration,
                                                          alpha_converge_threshold,
                                                          alpha_maximum_iteration,
                                                          global_maximum_iteration);
    
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(FreqDist) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, num_topics=10, batch_size=1, kappa=0, tau=1024):
        super(UncollapsedVariationalBayes, self)._initialize(data, num_topics);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = numpy.tile(self._alpha + 1.0 * self._V / self._K, (self._D, 1));
        self._gamma = numpy.exp((scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, 1))[:, numpy.newaxis]));
        
        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._lambda = 1 * numpy.random.gamma(100., 1. / 100., (self._V, self._K));
        self._log_beta = scipy.special.psi(self._lambda) - scipy.special.psi(numpy.sum(self._lambda, axis=0)[numpy.newaxis, :]);
        
        self._eta = 1.0 / self._K;
        self._batch_size = batch_size;
        
        self._tau = tau;
        self._kappa = kappa;
        
        print "alpha vector is", self._alpha
        
        #self._log_beta = 1.0/self._V + numpy.random.random((self._V, self._K));
        #self._log_beta = self._log_beta / numpy.sum(self._log_beta, axis=0)[numpy.newaxis, :];
        #self._log_beta = numpy.log(self._log_beta);

    """
    @deprecated: no longer useful
    """
    def update_phi(self, doc_id, phi_table):
        self._gamma[[doc_id], :] = numpy.random.gamma(100., 1. / 100., (1, self._K))
        self._gamma[[doc_id], :] = numpy.exp(scipy.special.psi(self._gamma[[doc_id], :]) - scipy.special.psi(numpy.sum(self._gamma[[doc_id], :])));
        
        # update phi and gamma until gamma converges
        for gamma_iteration in xrange(self._gamma_maximum_iteration):
            term_ids = numpy.array(self._data[doc_id].keys());
            term_counts = numpy.array([self._data[doc_id].values()]);
            assert(term_counts.shape == (1, len(term_ids)));
            
            phi_contribution = self._log_beta[term_ids, :] + numpy.log(self._gamma[[doc_id], :]);
            #phi_contribution = self._log_beta[term_ids, :] + scipy.special.psi(self._gamma[[doc_id], :]);
            phi_normalizer = numpy.log(numpy.sum(numpy.exp(phi_contribution), axis=1)[:, numpy.newaxis] + 1e-100);
            assert(phi_normalizer.shape == (len(term_ids), 1));
            phi_contribution -= phi_normalizer;
            
            assert(phi_contribution.shape == (len(term_ids), self._K));
            phi_contribution += numpy.log(term_counts.transpose());
            
            gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(phi_contribution), axis=0));
            gamma_update = numpy.exp(scipy.special.psi(gamma_update) - scipy.special.psi(numpy.sum(gamma_update)));
            mean_change = numpy.mean(abs(gamma_update - self._gamma[doc_id, :]));
            self._gamma[[doc_id], :] = gamma_update;
            if mean_change <= self._gamma_converge_threshold:
                break;
            
        likelihood_phi = numpy.sum(numpy.exp(phi_contribution) * ((self._log_beta[term_ids, :] * term_counts.transpose()) - phi_contribution));
        assert(phi_contribution.shape == (len(term_ids), self._K));
        #phi_table /= self._batch_size;
        #phi_table += self._eta;
        #phi_table = numpy.ones((self._V, self._K))*self._eta*self._batch_size;
        phi_table[[term_ids], :] += numpy.exp(phi_contribution);

        return phi_table, likelihood_phi

    """
    """
    def learning(self, iteration=0):
        if iteration <= 0:
            iteration = self._global_maximum_iteration;
        
        for i in xrange(iteration):
            # initialize a V-by-K matrix phi contribution
            phi_table = numpy.zeros((self._V, self._K)) + self._eta;
            
            # initialize alpha sufficient statistics
            #alpha_sufficient_statistics = numpy.zeros((1, self._K));
            
            # iterate over all documents
            for doc_id in self._data.keys():
                if numpy.random.random() > self._batch_size:
                    continue;
                
                # retrieve the term ids and their associated counts
                term_ids = numpy.array(self._data[doc_id].keys());
                term_counts = numpy.array([self._data[doc_id].values()]);
                assert(term_counts.shape == (1, len(term_ids)));
                
                # update phi and gamma until gamma converges
                for gamma_iteration in xrange(self._gamma_maximum_iteration):
                    # compute the phi contribution                    
                    # @attention: note self._gamma in this case is exp(E[log(theta)]), this is different from vanilla variational inference for lda.
                    phi_contribution = self._log_beta[term_ids, :] + numpy.log(self._gamma[[doc_id], :]);
                    phi_normalizer = numpy.log(numpy.sum(numpy.exp(phi_contribution), axis=1)[:, numpy.newaxis] + 1e-100);
                    assert(phi_normalizer.shape == (len(term_ids), 1));
                    phi_contribution -= phi_normalizer;
                    assert(phi_contribution.shape == (len(term_ids), self._K));
                    phi_contribution += numpy.log(term_counts.transpose());
                    
                    # compute the updated gamma
                    gamma_update = self._alpha + numpy.array(numpy.sum(numpy.exp(phi_contribution), axis=0));
                    # @attention: note we are computing the exp(E[log(theta)]) and store it to self._gamma.
                    gamma_update = numpy.exp(scipy.special.psi(gamma_update) - scipy.special.psi(numpy.sum(gamma_update)));
                    mean_change = numpy.mean(abs(gamma_update - self._gamma[doc_id, :]));
                    self._gamma[[doc_id], :] = gamma_update;
                    if mean_change <= self._gamma_converge_threshold:
                        break;
                    
                #likelihood_phi += numpy.sum(numpy.exp(phi_contribution) * ((self._log_beta[term_ids, :] * term_counts.transpose()) - phi_contribution));
                assert(phi_contribution.shape == (len(term_ids), self._K));
                phi_table[[term_ids], :] += numpy.exp(phi_contribution) / self._batch_size;
                #alpha_sufficient_statistics += (scipy.special.psi(self._gamma[[doc_id], :]) - scipy.special.psi(numpy.sum(self._gamma[[doc_id], :], axis=1)));
                            
            rho = numpy.power((self._tau + i), -self._kappa);
            # @attention: note we are updating lambda, which is a variational dirichet prior for beta matrix. unlike self._gamma, we need to explicitly store self._lambda and use it for update.
            self._lambda = (1.0 - rho) * self._lambda + rho * phi_table;
            if self._truncate_beta:
                # truncate beta to the minimum value in the beta matrix
                self._lambda[numpy.nonzero(self._lambda <= numpy.mean(self._lambda))] = numpy.min(self._lambda);
            #print numpy.max(rho * phi_table, axis=0), numpy.min(rho * phi_table, axis=0), numpy.mean(rho * phi_table, axis=0)
            #print numpy.max(self._lambda, axis=0), numpy.min(self._lambda, axis=0), numpy.mean(self._lambda, axis=0)
            # @attention: note we are computing exp(E[log(theta)]) and store it in log scale in this case.
            self._log_beta = (scipy.special.psi(self._lambda) - scipy.special.psi(numpy.sum(self._lambda, 0))[numpy.newaxis, :]);
            assert(self._log_beta.shape == (self._V, self._K));

            # compute the log-likelihood of alpha terms
            #alpha_sum = numpy.sum(self._alpha, axis=1);
            #likelihood_alpha = numpy.sum(scipy.special.gammaln(self._alpha), axis=1);
            #likelihood_alpha += scipy.special.gammaln(alpha_sum);
            #likelihood_alpha += self._D;
            
            #likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
            #likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));
    
            #new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
            #print "em iteration is ", (i+1), " likelihood is ", new_likelihood
            print "em iteration is", (i + 1), "rho is", rho
            
            alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
            alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
            self.update_alpha(alpha_sufficient_statistics, rho)
            print "alpha vector is", self._alpha
            
        print "learning finished..."

if __name__ == "__main__":
    temp_directory = "../../../data/de-news/en/corpus-2/";
    #temp_directory = "../../../data/test/";
    from util.input_parser import import_monolingual_data;
    d = import_monolingual_data(temp_directory + "doc.dat");
    
    lda = UncollapsedVariationalBayes(10, True);
    lda._initialize(d, 3, 0.1, 0.7);
    lda.learning(100);
    #print lda._log_beta
    #print lda._gamma
    lda.print_topics(temp_directory + "voc.dat");