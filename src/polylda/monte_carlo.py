"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import codecs
import collections
import math, random, time;
import nltk;
import numpy;
import os;
import scipy;

#from collections import defaultdict
#from nltk import FreqDist

"""
This is a python implementation of polylingual lda, based on collapsed Gibbs sampling, with hyper parameter updating.
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
                 alpha_maximum_iteration=10,
                 hyper_parameter_sampling_interval=20):

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
    def _initialize(self, data, language_type_index, language_index_type, number_of_topics=10, alpha=0.5, beta=0.1):
        self._counter=0;
        
        # set the document smooth factor
        self._alpha = alpha
        # set the vocabulary smooth factor
        self._beta = beta
        
        # define the counts over different topics for all documents, first indexed by document id, the indexed by topic id
        self._document_topic_counts = collections.defaultdict(nltk.probability.FreqDist)
        #self._document_topic_counts = numpy.zeros((self._number_of_documents, self._number_of_topics));
        # define the counts over words for all languages and topics, indexed by language id, topic id, and token id
        self._language_topic_type_counts = {};
        # define the topic assignment for every word in every language of every document, indexed by document id, language id, and word position
        self._document_language_position = {};
        
        self._number_of_topics = number_of_topics;
        
        # define the input data
        self._data = data
        # define the number of document
        self._number_of_documents = len(data)
        # define the number of languages
        self._number_of_languages = len(data[0]);
        
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._language_type_index = language_type_index
        self._language_index_type = language_index_type
        
        assert self._number_of_languages==len(self._language_index_type);
        assert self._number_of_languages==len(self._language_type_index);
        
        self._number_of_language_types = {}
        
        for language_id in xrange(self._number_of_languages):
            self._number_of_language_types[language_id] = len(self._language_type_index[language_id]);
            #self._language_topic_type_counts[language_id] = numpy.zeros((self._number_of_topics, self._number_of_language_types[language_id]));
            self._language_topic_type_counts[language_id] = collections.defaultdict(nltk.probability.FreqDist);

        for doc_id in xrange(self._number_of_documents):
            self._document_language_position[doc_id] = collections.defaultdict(dict);
            
            for lang_id in xrange(self._number_of_languages):
                for position in xrange(len(self._data[doc_id][lang_id])):
                    # initialize the state to unassigned
                    self._document_language_position[doc_id][lang_id][position] = -1;
                    
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
                    #self._alpha_sum = self._alpha * self._number_of_topics
                    #self._beta_sum = self._beta * self._number_of_language_types
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
        beta_sum = numpy.zeros(self._number_of_languages);
        for language_id in xrange(self._number_of_languages):
            beta_sum[language_id] = beta * self._number_of_language_types[language_id];

        log_likelihood = 0.0
        # compute the log log_likelihood of the document
        log_likelihood += scipy.special.gammaln(alpha_sum) * self._number_of_documents
        log_likelihood -= scipy.special.gammaln(alpha) * self._number_of_topics * self._number_of_documents
           
        for jj in self._document_topic_counts.keys():
            for kk in xrange(self._number_of_topics):
                log_likelihood += scipy.special.gammaln(alpha + self._document_topic_counts[jj][kk])                    
            log_likelihood -= scipy.special.gammaln(alpha_sum + self._document_topic_counts[jj].N())
            
        # compute the log log_likelihood of the topic
        for ii in xrange(self._number_of_languages):
            log_likelihood += scipy.special.gammaln(beta_sum[ii]) * self._number_of_topics
            log_likelihood -= scipy.special.gammaln(beta) * self._number_of_language_types[ii] * self._number_of_topics

            for jj in self._language_topic_type_counts[ii].keys():
                for kk in self._language_type_index[ii]:
                    log_likelihood += scipy.special.gammaln(beta + self._language_topic_type_counts[ii][jj][kk])
                log_likelihood -= scipy.special.gammaln(beta_sum[ii] + self._language_topic_type_counts[ii][jj].N())

        return log_likelihood

    """
    compute the conditional distribution
    @param doc_id: doc_id id
    @param word: word id
    @param topic: topic id  
    @return: the probability value of the topic for that word in that document
    """
    def log_prob(self, doc_id, lang_id, topic_id, word_id):
        val = math.log(self._document_topic_counts[doc_id][topic_id] + self._alpha)
        #this is constant across a document, so we don't need to compute this term
        # val -= math.log(self._document_topic_counts[doc_id].N() + self._alpha_sum)
        
        val += math.log(self._language_topic_type_counts[lang_id][topic_id][word_id] + self._beta)
        val -= math.log(self._language_topic_type_counts[lang_id][topic_id].N() + self._number_of_language_types[lang_id] * self._beta)
        
        return val

    """
    this method samples the word at position in document, by covering that word and compute its new topic distribution, in the end, both self._document_language_position, self._document_topic_counts and self._language_topic_type_counts will change
    @param doc_id: a document id
    @param position: the position in doc_id, ranged as range(self._data[doc_id])
    """
    def sample_document(self, doc_id):
        for lang_id in xrange(self._number_of_languages):
            for position in xrange(len(self._data[doc_id][lang_id])):
                assert position >= 0 and position < len(self._data[doc_id][lang_id])
                
                #retrieve the word_id
                word_id = self._data[doc_id][lang_id][position]
            
                #get the old topic assignment to the word_id in doc_id at position
                old_topic = self._document_language_position[doc_id][lang_id][position]
                if old_topic != -1:
                    #this word_id already has a valid topic assignment, decrease the topic|doc_id counts and word_id|topic counts by covering up that word_id
                    #self.change_count(doc_id, lang_id, word_id, old_topic, -1)
                    
                    self._document_topic_counts[doc_id].inc(old_topic, -1)
                    self._language_topic_type_counts[lang_id][old_topic].inc(word_id, -1)
        
                #compute the topic probability of current word_id, given the topic assignment for other words
                topic_log_probability = [self.log_prob(doc_id, lang_id, topic_id, self._data[doc_id][lang_id][position]) for topic_id in xrange(self._number_of_topics)]
                
                #learning a new topic out of a distribution according to topic_log_probability
                #new_topic = util.log_math.log_sample(topic_log_probability)
                topic_log_probability = numpy.asarray(topic_log_probability);
                topic_log_probability -= scipy.misc.logsumexp(topic_log_probability);
                topic_probability = numpy.exp(topic_log_probability);
                temp_topic_probability = numpy.random.multinomial(1, topic_probability)[numpy.newaxis, :]
                new_topic = numpy.nonzero(temp_topic_probability==1)[1][0];
        
                #after we draw a new topic for that word_id, we will change the topic|doc_id counts and word_id|topic counts, i.e., add the counts back
                #self.change_count(doc_id, lang_id, word_id, new_topic, 1)
                self._document_topic_counts[doc_id].inc(new_topic, 1)
                self._language_topic_type_counts[lang_id][new_topic].inc(word_id, 1)
                
                #assign the topic for the word_id of current document at current position
                self._document_language_position[doc_id][lang_id][position] = new_topic

    """
    learning the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def learning(self):
        #learning the total corpus
        #for iter1 in xrange(number_of_iterations):
        self._counter += 1;
        
        processing_time = time.time();

        #learning every document
        for doc_id in xrange(self._number_of_documents):
            for iter in xrange(self._local_maximum_iteration):
                self.sample_document(doc_id);

            if (doc_id+1) % 1000==0:
                print "successfully sampled %d documents" % (doc_id+1)

        if self._counter % self._hyper_parameter_sampling_interval == 0:
            self.optimize_hyperparameters();

        processing_time = time.time() - processing_time;                
        print("iteration %i finished in %d seconds with log-likelihood %g" % (self._counter, processing_time, self.compute_likelihood(self._alpha, self._beta)))

    def export_exp_beta(self, exp_beta_directory):
        for language_id in xrange(self._number_of_languages):
            exp_beta_path = os.path.join(exp_beta_directory, "exp_beta-%d-language-%d.dat" % (self._counter, language_id));
            
            output = codecs.open(exp_beta_path, mode="w", encoding="utf-8")
            
            for k in xrange(self._number_of_topics):
                output.write("==========\t%d\t==========\n" % (k));
    
                i = 0;
                for key in self._language_topic_type_counts[language_id][k]:
                    i += 1;
                    output.write("%s\t%g\n" % (self._language_index_type[language_id][key], (self._language_topic_type_counts[language_id][k][key]+self._beta)/(self._language_topic_type_counts[language_id][k].N()+self._beta*self._number_of_language_types[language_id])));
                
            output.close();

        exp_theta_path = os.path.join(exp_beta_directory, "exp_theta-%d.dat" % (self._counter));
        output = codecs.open(exp_theta_path, mode="w", encoding="utf-8");
        for doc_id in xrange(self._number_of_documents):
            for topic_id in xrange(self._number_of_topics):
                output.write("%g\t" % (self._document_topic_counts[doc_id][topic_id]));
            output.write("\n");
        
    """
    this methods change the count of a topic in one doc_id and a word of one topic by delta
    this values will be used in the computation
    @param doc_id: the doc_id id
    @param word: the word id
    @param topic: the topic id
    @param delta: the change in the value
    @deprecated:
    """
    '''
    def change_count(self, doc_id, lang_id, word_id, topic_id, delta):
        self._document_topic_counts[doc_id].inc(topic_id, delta)
        self._language_topic_type_counts[lang_id][topic_id].inc(word_id, delta)
    '''
        
if __name__ == "__main__":
    print "not implemented"