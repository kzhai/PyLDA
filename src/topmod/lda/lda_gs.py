# Original Author: Jordan Boyd-Graber
# Email: jbg@umiacs.umd.edu
# Modification: Ke Zhai
# Email: zhaike@cs.umd.edu

from collections import defaultdict
from math import log, exp
from random import random
from nltk import FreqDist
from scipy.special import psi, gammaln, polygamma;

class LDAGibbsSampling:
    def __init__(self, ):
        self._docs = defaultdict(FreqDist)
        self._topics = defaultdict(FreqDist)
        
        self._state = None

        self._alpha = 0.1
        self._lambda = 0.01
        
        self._alpha_update_decay_factor = 0.9
        self._alpha_maximum_decay = 10
        
        self._alpha_converge = 0.000001
        self._alpha_maximum_iteration = 100
        
        self._maximum_iteration = 100
        self._converge = 0.00001

        # pending for further changing~
        self._gamma_converge = 0.000001
        self._gamma_maximum_iteration = 400
        
    # data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    def _initialize(self, num_topics=10, data):
        self._K = num_topics
        
        self._to_be_deleted_alpha_sum = self._alpha * self._K
        self._state = defaultdict(dict)
    
        self._data = data
        
        self._D = len(data)
    
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = set([])
        for doc in self._data:
            for position in xrange(self._data[doc]):
                # learn all the words we'll see
                self._vocab.add(self._data[doc][position])
            
                # initialize the state to unassigned
                self._state[doc][position] = -1
        self._V = len(self._vocab)
        
        self._to_be_deleted_lambda_sum = float(self._V) * self._lambda

    def optimize_hyperparameters(self, samples=5, step=3.0):
        rawParam = [log(self._alpha), log(self._lambda)]

        for ii in xrange(samples):
            lp_old = self.lhood(self._alpha, self._lambda)
            lp_new = log(random()) + lp_old
            print("OLD: %f\tNEW: %f at (%f, %f)" % (lp_old, lp_new, self._alpha, self._lambda))

            l = [x - random() * step for x in rawParam]
            r = [x + step for x in rawParam]

            for jj in xrange(100):
                rawParamNew = [l[x] + random() * (r[x] - l[x]) for x in xrange(len(rawParam))]
                trial_alpha, trial_lambda = [exp(x) for x in rawParamNew]
                lp_test = self.lhood(trial_alpha, trial_lambda)
                #print("TRYING: %f (need %f) at (%f, %f)" % (lp_test - lp_old, lp_new - lp_old, trial_alpha, trial_lambda))

                if lp_test > lp_new:
                    print(jj)
                    self._alpha = exp(rawParamNew[0])
                    self._lambda = exp(rawParamNew[1])
                    self._to_be_deleted_alpha_sum = self._alpha * self._K
                    self._to_be_deleted_lambda_sum = self._lambda * self._V
                    rawParam = [log(self._alpha), log(self._lambda)]
                    break
                else:
                    for dd in xrange(len(rawParamNew)):
                        if rawParamNew[dd] < rawParam[dd]:
                            l[dd] = rawParamNew[dd]
                        else:
                            r[dd] = rawParamNew[dd]
                        assert l[dd] <= rawParam[dd]
                        assert r[dd] >= rawParam[dd]

            print("\nNew hyperparameters (%i): %f %f" % (jj, self._alpha, self._lambda))

    def lhood(self, doc_smoothing, voc_smoothing):
        doc_sum = doc_smoothing * self._K
        voc_sum = voc_smoothing * self._V

        val = 0.0
        val += lgammln(doc_sum) * len(self._docs)
        val -= lgammln(doc_smoothing) * self._K * len(self._docs)
        for ii in self._docs:
            for jj in xrange(self._K):
                val += lgammln(doc_smoothing + self._docs[ii][jj])
            val -= lgammln(doc_sum + self._docs[ii].N())
      
            val += lgammln(voc_sum) * self._K
            val -= lgammln(voc_smoothing) * self._V * self._K
            for ii in self._topics:
                for jj in self._vocab:
                    val += lgammln(voc_smoothing + self._topics[ii][jj])
                val -= lgammln(voc_sum + self._topics[ii].N())
            return val

    def prob(self, doc, word, topic):
        val = log(self._docs[doc][topic] + self._alpha)
        # This is constant across a document, so we don't need to compute this term
        # val -= log(self._docs[doc].N() + self._to_be_deleted_alpha_sum)
        
        val += log(self._topics[topic][word] + self._lambda)
        val -= log(self._topics[topic].N() + self._to_be_deleted_lambda_sum)
    
        return val

    def sample_word(self, doc, position):
        word = self._data[doc][position]
    
        old_topic = self._state[doc][position]
        if old_topic != -1:
            self.change_count(doc, word, old_topic, -1)
    
        probs = [self.prob(doc, self._data[doc][position], x) for x in xrange(self._K)]
        new_topic = log_sample(probs)
    
        self.change_count(doc, word, new_topic, 1)
        self._state[doc][position] = new_topic

    def change_count(self, doc, word, topic, delta):
        self._docs[doc].inc(topic, delta)
        self._topics[topic].inc(word, delta)

    def sample(self, hyper_delay=10):
        assert self._state
        for iter in xrange(self._maximum_iteration):
            for doc in self._data:
                for position in xrange(len(self._data[doc])):
                    self.sample_word(doc, position)
                    
            print("Iteration %i %f" % (iter, self.lhood(self._alpha, self._lambda)))
            if hyper_delay >= 0 and iter % hyper_delay == 0:
                self.optimize_hyperparameters()

    def print_topics(self, num_words=15):
        for ii in self._topics:
            print("%i:%s\n" % (ii, "\t".join(self._topics[ii].keys()[:num_words])))

if __name__ == "__main__":
    #d = create_data("/nfshomes/jbg/sentop/topicmod/data/de_news/txt/*.en.txt", doc_limit=50, delimiter="<doc")
    from topmod.io.de_news_io import parse_de_news_gs
    d = parse_de_news_gs("/windows/d/Data/de-news/txt/*.en.txt", 1, true, false)
    
    lda = LDAGibbsSampling(5)
    lda.initialize(d)

    lda.sample(100)
    lda.print_topics()