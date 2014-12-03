"""
@author: Jordan Boyd-Graber (jbg@umiacs.umd.edu)
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import math, random, time;
import numpy;
import scipy;

#from collections import defaultdict
#from nltk import FreqDist
from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer;

"""
This is a python implementation of lda, based on collapsed Gibbs sampling, with hyper parameter updating.
It only supports symmetric Dirichlet prior over the topic simplex.

References:
[1] T. L. Griffiths & M. Steyvers. Finding Scientific Topics. Proceedings of the National Academy of Sciences, 101, 5228-5235, 2004.
"""
class MonteCarlo(Inferencer):
    """
    """
    def __init__(self,
                hyper_parameter_optimize_interval=10,
                 local_maximum_iteration=1, 

                 hyper_parameter_iteration=50,):

        self._hyper_parameter_iteration = hyper_parameter_iteration
        assert(self._hyper_parameter_iteration>0)
        
        self._hyper_parameter_optimize_interval = hyper_parameter_optimize_interval;
        assert(self._hyper_parameter_optimize_interval>0);
        
        self._local_maximum_iteration = local_maximum_iteration
        assert(self._local_maximum_iteration>0)

    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, corpus, vocab, number_of_topics, alpha_alpha, alpha_beta):
        Inferencer._initialize(self, vocab, number_of_topics, alpha_alpha, alpha_beta);
        
        self._corpus = corpus;
        self.parse_data();
        
        # define the total number of document
        self._number_of_documents = len(self._word_idss);
        
        # define the counts over different topics for all documents, first indexed by doc_id id, the indexed by topic id
        self._n_dk = numpy.zeros((self._number_of_documents, self._number_of_topics));
        # define the counts over words for all topics, first indexed by topic id, then indexed by token id
        self._n_kv = numpy.zeros((self._number_of_topics, self._number_of_types));
        self._n_k = numpy.zeros(self._number_of_topics);
        # define the topic assignment for every word in every document, first indexed by doc_id id, then indexed by word word_pos
        self._k_dn = {};
        
        self.random_initialize();

    def random_initialize(self):
        # initialize the vocabulary, i.e. a list of distinct tokens.
        for doc_id in xrange(self._number_of_documents):
            self._k_dn[doc_id] = numpy.zeros(len(self._word_idss[doc_id]));
            for word_pos in xrange(len(self._word_idss[doc_id])):
                type_index = self._word_idss[doc_id][word_pos];
                topic_index = numpy.random.randint(self._number_of_topics);
                
                self._k_dn[doc_id][word_pos] = topic_index;
                self._n_dk[doc_id, topic_index] += 1;
                self._n_kv[topic_index, type_index] += 1;
                self._n_k[topic_index] += 1;
        
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
        
    """
    """
    def optimize_hyperparameters(self, samples=5, step=3.0):
        old_hyper_parameters = [math.log(self._alpha_alpha), math.log(self._alpha_beta)]

        for ii in xrange(samples):
            log_likelihood_old = self.log_posterior(self._alpha_alpha, self._alpha_beta)
            log_likelihood_new = math.log(random.random()) + log_likelihood_old
            #print("OLD: %f\tNEW: %f at (%f, %f)" % (log_likelihood_old, log_likelihood_new, self._alpha_alpha, self._alpha_beta))

            l = [x - random.random() * step for x in old_hyper_parameters]
            r = [x + step for x in old_hyper_parameters]

            for jj in xrange(self._hyper_parameter_iteration):
                new_hyper_parameters = [l[x] + random.random() * (r[x] - l[x]) for x in xrange(len(old_hyper_parameters))]
                trial_alpha, trial_beta = [math.exp(x) for x in new_hyper_parameters]
                lp_test = self.log_posterior(trial_alpha, trial_beta)

                if lp_test > log_likelihood_new:
                    #print(jj)
                    self._alpha_alpha = math.exp(new_hyper_parameters[0])
                    self._alpha_beta = math.exp(new_hyper_parameters[1])
                    #self._alpha_sum = self._alpha_alpha * self._number_of_topics
                    #self._beta_sum = self._alpha_beta * self._number_of_types
                    old_hyper_parameters = [math.log(self._alpha_alpha), math.log(self._alpha_beta)]
                    break
                else:
                    for dd in xrange(len(new_hyper_parameters)):
                        if new_hyper_parameters[dd] < old_hyper_parameters[dd]:
                            l[dd] = new_hyper_parameters[dd]
                        else:
                            r[dd] = new_hyper_parameters[dd]
                        assert l[dd] <= old_hyper_parameters[dd]
                        assert r[dd] >= old_hyper_parameters[dd]

            #print("\nNew hyperparameters (%i): %f %f" % (jj, self._alpha_alpha, self._alpha_beta))

    """
    compute the log-likelihood of the model
    """
    def log_posterior(self, alpha, beta):
        assert self._n_dk.shape==(self._number_of_documents, self._number_of_topics);
        assert alpha.shape==(self._number_of_topics,);
        assert beta.shape==(self._number_of_types,);
        
        alpha_sum = numpy.sum(alpha);
        beta_sum = numpy.sum(beta);
        
        log_likelihood = 0.0
        # compute the log log_likelihood of the document
        log_likelihood += scipy.special.gammaln(numpy.sum(alpha)) * self._number_of_documents
        log_likelihood -= numpy.sum(scipy.special.gammaln(alpha)) * self._number_of_documents
        
        for doc_id in xrange(self._number_of_documents):
            log_likelihood += numpy.sum(scipy.special.gammaln(self._n_dk[doc_id, :] + alpha));
            log_likelihood -= scipy.special.gammaln(numpy.sum(self._n_dk[doc_id, :]) + alpha_sum);
        
        '''
        for ii in self._n_dk.keys():
            for jj in xrange(self._number_of_topics):
                log_likelihood += scipy.special.gammaln(alpha + self._n_dk[ii][jj])                    
            log_likelihood -= scipy.special.gammaln(alpha_sum + self._n_dk[ii].N())
        '''
        
        # compute the log log_likelihood of the topic
        log_likelihood += scipy.special.gammaln(numpy.sum(beta)) * self._number_of_topics;
        log_likelihood -= numpy.sum(scipy.special.gammaln(beta)) * self._number_of_topics;

        for topic_id in xrange(self._number_of_topics):
            log_likelihood += numpy.sum(scipy.special.gammaln(self._n_kv[topic_id, :] + beta));
            log_likelihood -= scipy.special.gammaln(self._n_k[topic_id] + beta_sum);
        
        '''
        for ii in self._n_kv.keys():
            for jj in self._type_to_index:
                log_likelihood += scipy.special.gammaln(beta + self._n_kv[ii][jj])
            log_likelihood -= scipy.special.gammaln(beta_sum + self._n_kv[ii].N())
        '''
        
        return log_likelihood

    """
    this method samples the word at position in document, by covering that word and compute its new topic distribution, in the end, both self._k_dn, self._n_dk and self._n_kv will change
    @param doc_id: a document id
    @param position: the position in doc_id, ranged as range(self._word_idss[doc_id])
    """
    def sample_document(self, doc_id):
        for position in xrange(len(self._word_idss[doc_id])):
            assert position >= 0 and position < len(self._word_idss[doc_id])
            
            #retrieve the word_id
            word_id = self._word_idss[doc_id][position]
        
            #get the old topic assignment to the word_id in doc_id at position
            old_topic = self._k_dn[doc_id][position]
            if old_topic != -1:
                #this word_id already has a valid topic assignment, decrease the topic|doc_id counts and word_id|topic counts by covering up that word_id
                self._n_dk[doc_id, old_topic] -= 1
                self._n_kv[old_topic, word_id] -= 1;
                self._n_k[old_topic] -= 1;
    
            #compute the topic probability of current word_id, given the topic assignment for other words
            log_probability = numpy.log(self._n_dk[doc_id, :] + self._alpha_alpha); 
            log_probability += numpy.log(self._n_kv[:, word_id] + self._alpha_beta[word_id]);
            log_probability -= numpy.log(self._n_k + numpy.sum(self._alpha_beta));

            log_probability -= scipy.misc.logsumexp(log_probability)
            
            #sample a new topic out of a distribution according to log_probability
            temp_probability = numpy.exp(log_probability);
            temp_topic_probability = numpy.random.multinomial(1, temp_probability)[numpy.newaxis, :]
            new_topic = numpy.nonzero(temp_topic_probability == 1)[1][0];
    
            #after we draw a new topic for that word_id, we will change the topic|doc_id counts and word_id|topic counts, i.e., add the counts back
            self._n_dk[doc_id, new_topic] += 1
            self._n_kv[new_topic, word_id] += 1;
            self._n_k[new_topic] += 1;
            #assign the topic for the word_id of current document at current position
            self._k_dn[doc_id][position] = new_topic

    """
    sample the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def learning(self):
        #sample the total corpus
        #for iter1 in xrange(number_of_iterations):
        self._counter += 1;
        
        processing_time = time.time();

        #sample every document
        for doc_id in xrange(self._number_of_documents):
            for iter in xrange(self._local_maximum_iteration):
                self.sample_document(doc_id)

            if (doc_id+1) % 1000==0:
                print "successfully sampled %d documents" % (doc_id+1)

        if self._counter % self._hyper_parameter_optimize_interval == 0:
            self.optimize_hyperparameters()

        processing_time = time.time() - processing_time;                
        print("iteration %i finished in %d seconds with log-likelihood %g" % (self._counter, processing_time, self.log_posterior(self._alpha_alpha, self._alpha_beta)))

    def export_beta(self, exp_beta_path, top_display=-1):
        output = open(exp_beta_path, 'w');
        for topic_index in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (topic_index));
            
            beta_probability = self._n_kv[topic_index, :] + self._alpha_beta;
            beta_probability /= numpy.sum(beta_probability);
            
            i = 0;
            for type_index in reversed(numpy.argsort(beta_probability)):
                i += 1;
                output.write("%s\t%g\n" % (self._index_to_type[type_index], beta_probability[type_index]));
                if top_display > 0 and i >= top_display:
                    break;
                
        output.close();

if __name__ == "__main__":
    print "not implemented"