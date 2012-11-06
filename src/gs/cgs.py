"""
@author: Jordan Boyd-Graber (jbg@umiacs.umd.edu)
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import math, random, time;
import scipy;
import util.log_math;

from collections import defaultdict
from nltk import FreqDist

"""
This is a python implementation of lda, based on collapsed Gibbs sampling, with hyper parameter updating.
It only supports symmetric Dirichlet prior over the topic simplex.

References:
[1] T. L. Griffiths & M. Steyvers. Finding Scientific Topics. Proceedings of the National Academy of Sciences, 101, 5228-5235, 2004.
"""
class CollapsedGibbsSampling:
    """
    """
    def __init__(self,
                 #snapshot_interval=10,
                 local_maximum_iteration=5, 
                 alpha_maximum_iteration=50,
                 hyper_parameter_sampling_interval=20):

        #self._snapshot_iterval = snapshot_interval;

        self._alpha_maximum_iteration = alpha_maximum_iteration
        assert(self._alpha_maximum_iteration>0)
        
        self._local_maximum_iteration = local_maximum_iteration
        assert(self._local_maximum_iteration>0)

        self._hyper_parameter_sampling_interval = hyper_parameter_sampling_interval;
        assert(self._hyper_parameter_sampling_interval>0);
        
    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, data, type_to_index, index_to_type, number_of_topics=10, alpha=0.5, beta=0.1):
        self._counter=0;
        
        # set the document smooth factor
        self._alpha = alpha
        # set the vocabulary smooth factor
        self._beta = beta
        
        # define the counts over different topics for all documents, first indexed by doc_id id, the indexed by topic id
        self._document_topic_counts = defaultdict(FreqDist)
        # define the counts over words for all topics, first indexed by topic id, then indexed by token id
        self._topic_term_counts = defaultdict(FreqDist)
        # define the topic assignment for every word in every document, first indexed by doc_id id, then indexed by word position
        self._topic_assignment = defaultdict(dict)
        
        self._number_of_topics = number_of_topics;
    
        self._alpha_sum = self._alpha * self._number_of_topics

        # define the input data
        self._data = data
        # define the total number of document
        self._number_of_documents = len(data)
    
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._type_to_index = type_to_index
        self._index_to_type = index_to_type

        for doc_id in xrange(self._number_of_documents):
            for position in xrange(len(self._data[doc_id])):
                # initialize the state to unassigned
                self._topic_assignment[doc_id][position] = -1;
                
        self._number_of_terms = len(self._type_to_index);
        
    """
    """
    def optimize_hyperparameters(self, samples=5, step=3.0):
        old_hyper_parameters = [math.log(self._alpha), math.log(self._beta)]

        for ii in xrange(samples):
            log_likelihood_old = self.compute_likelihood(self._alpha, self._beta)
            log_likelihood_new = math.log(random.random()) + log_likelihood_old
            #print("OLD: %f\tNEW: %f at (%f, %f)" % (log_likelihood_old, log_likelihood_new, self._alpha, self._beta))

            l = [x - random.random() * step for x in old_hyper_parameters]
            r = [x + step for x in old_hyper_parameters]

            for jj in xrange(self._alpha_maximum_iteration):
                new_hyper_parameters = [l[x] + random.random() * (r[x] - l[x]) for x in xrange(len(old_hyper_parameters))]
                trial_alpha, trial_beta = [math.exp(x) for x in new_hyper_parameters]
                lp_test = self.compute_likelihood(trial_alpha, trial_beta)

                if lp_test > log_likelihood_new:
                    #print(jj)
                    self._alpha = math.exp(new_hyper_parameters[0])
                    self._beta = math.exp(new_hyper_parameters[1])
                    self._alpha_sum = self._alpha * self._number_of_topics
                    self._beta_sum = self._beta * self._number_of_terms
                    old_hyper_parameters = [math.log(self._alpha), math.log(self._beta)]
                    break
                else:
                    for dd in xrange(len(new_hyper_parameters)):
                        if new_hyper_parameters[dd] < old_hyper_parameters[dd]:
                            l[dd] = new_hyper_parameters[dd]
                        else:
                            r[dd] = new_hyper_parameters[dd]
                        assert l[dd] <= old_hyper_parameters[dd]
                        assert r[dd] >= old_hyper_parameters[dd]

            #print("\nNew hyperparameters (%i): %f %f" % (jj, self._alpha, self._beta))

    """
    compute the log-likelihood of the model
    """
    def compute_likelihood(self, alpha, beta):
        assert len(self._document_topic_counts) == self._number_of_documents
        
        alpha_sum = alpha * self._number_of_topics
        beta_sum = beta * self._number_of_terms

        likelihood = 0.0
        # compute the log likelihood of the document
        likelihood += scipy.special.gammaln(alpha_sum) * self._number_of_documents
        likelihood -= scipy.special.gammaln(alpha) * self._number_of_topics * self._number_of_documents
           
        for ii in self._document_topic_counts.keys():
            for jj in xrange(self._number_of_topics):
                likelihood += scipy.special.gammaln(alpha + self._document_topic_counts[ii][jj])                    
            likelihood -= scipy.special.gammaln(alpha_sum + self._document_topic_counts[ii].N())
            
        # compute the log likelihood of the topic
        likelihood += scipy.special.gammaln(beta_sum) * self._number_of_topics
        likelihood -= scipy.special.gammaln(beta) * self._number_of_terms * self._number_of_topics

        for ii in self._topic_term_counts.keys():
            for jj in self._type_to_index:
                likelihood += scipy.special.gammaln(beta + self._topic_term_counts[ii][jj])
            likelihood -= scipy.special.gammaln(beta_sum + self._topic_term_counts[ii].N())

        return likelihood

    """
    compute the conditional distribution
    @param doc_id: doc_id id
    @param word: word id
    @param topic: topic id  
    @return: the probability value of the topic for that word in that document
    """
    def log_prob(self, doc_id, word, topic):
        val = math.log(self._document_topic_counts[doc_id][topic] + self._alpha)
        #this is constant across a document, so we don't need to compute this term
        # val -= math.log(self._document_topic_counts[doc_id].N() + self._alpha_sum)
        
        val += math.log(self._topic_term_counts[topic][word] + self._beta)
        val -= math.log(self._topic_term_counts[topic].N() + self._number_of_terms * self._beta)
    
        return val

    """
    this method samples the word at position in document, by covering that word and compute its new topic distribution, in the end, both self._topic_assignment, self._document_topic_counts and self._topic_term_counts will change
    @param doc_id: a document id
    @param position: the position in doc_id, ranged as range(self._data[doc_id])
    """
    def sample_document(self, doc_id):
        for position in xrange(len(self._data[doc_id])):
            assert position >= 0 and position < len(self._data[doc_id])
            
            #retrieve the word_id
            word_id = self._data[doc_id][position]
        
            #get the old topic assignment to the word_id in doc_id at position
            old_topic = self._topic_assignment[doc_id][position]
            if old_topic != -1:
                #this word_id already has a valid topic assignment, decrease the topic|doc_id counts and word_id|topic counts by covering up that word_id
                self.change_count(doc_id, word_id, old_topic, -1)
    
            #compute the topic probability of current word_id, given the topic assignment for other words
            probs = [self.log_prob(doc_id, self._data[doc_id][position], x) for x in xrange(self._number_of_topics)]
    
            #sample a new topic out of a distribution according to probs
            new_topic = util.log_math.log_sample(probs)
    
            #after we draw a new topic for that word_id, we will change the topic|doc_id counts and word_id|topic counts, i.e., add the counts back
            self.change_count(doc_id, word_id, new_topic, 1)
            #assign the topic for the word_id of current document at current position
            self._topic_assignment[doc_id][position] = new_topic

    """
    this methods change the count of a topic in one doc_id and a word of one topic by delta
    this values will be used in the computation
    @param doc_id: the doc_id id
    @param word: the word id
    @param topic: the topic id
    @param delta: the change in the value
    """
    def change_count(self, doc_id, word_id, topic_id, delta):
        self._document_topic_counts[doc_id].inc(topic_id, delta)
        self._topic_term_counts[topic_id].inc(word_id, delta)

    """
    sample the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def sample(self):
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

        if self._counter % self._hyper_parameter_sampling_interval == 0:
            self.optimize_hyperparameters()

        processing_time = time.time() - processing_time;                
        print("iteration %i finished in %d seconds with log-likelihood %g" % (self._counter, processing_time, self.compute_likelihood(self._alpha, self._beta)))

    def export_topic_term_distribution(self, exp_beta_path):
        output = open(exp_beta_path, 'w');
        for k in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (k));

            i = 0;
            for key in self._topic_term_counts[k]:
                i += 1;
                output.write("%s\t%g\n" % ( self._index_to_type[key], (self._topic_term_counts[k][key]+self._beta)/(self._topic_term_counts[k].N()+self._beta*self._number_of_terms)));
                
        output.close();
        
if __name__ == "__main__":
    temp_directory = "../data/ap/";
    from util.input_parser import import_monolingual_data;
    doc_id = import_monolingual_data(temp_directory+"doc_id.dat");
    
    lda = CollapsedGibbsSampling()
    lda._initialize(doc_id, 10)

    lda.sample()
    lda.print_topics(2)