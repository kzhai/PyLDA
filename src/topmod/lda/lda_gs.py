#from glob import glob
from collections import defaultdict
from math import log, exp
from random import random

from nltk import FreqDist

class LDAGibbsSampling:
    def __init__(self, num_topics, doc_smoothing=0.1, topic_smoothing=0.01):
        self._docs = defaultdict(FreqDist)
        self._topics = defaultdict(FreqDist)
        self._K = num_topics
        self._state = None

        self._alpha = doc_smoothing
        self._lambda = topic_smoothing

    def initialize(self, data):
        """
        Data should be keyed by doc-id, values should be iterable
        """

        self._alpha_sum = self._alpha * self._K
        self._state = defaultdict(dict)
    
        self._vocab = set([])
        for dd in data:
              for ww in xrange(len(data[dd])):
                    # Learn all the words we'll see
                    self._vocab.add(data[dd][ww])
            
                    # Initialize the state to unassigned
                    self._state[dd][ww] = -1
    
        self._W = len(self._vocab)
        self._lambda_sum = float(self._W) * self._lambda
    
        self._data = data
    
        print("Initialized vocab of size %i" % len(self._vocab))

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
                    self._alpha_sum = self._alpha * self._K
                    self._lambda_sum = self._lambda * self._W
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
        voc_sum = voc_smoothing * self._W

        val = 0.0
        val += lgammln(doc_sum) * len(self._docs)
        val -= lgammln(doc_smoothing) * self._K * len(self._docs)
        for ii in self._docs:
            for jj in xrange(self._K):
                val += lgammln(doc_smoothing + self._docs[ii][jj])
            val -= lgammln(doc_sum + self._docs[ii].N())
      
            val += lgammln(voc_sum) * self._K
            val -= lgammln(voc_smoothing) * self._W * self._K
            for ii in self._topics:
                for jj in self._vocab:
                    val += lgammln(voc_smoothing + self._topics[ii][jj])
                val -= lgammln(voc_sum + self._topics[ii].N())
            return val



    def prob(self, doc, word, topic):
        val = log(self._docs[doc][topic] + self._alpha)
        # This is constant across a document, so we don't need to compute this term
        # val -= log(self._docs[doc].N() + self._alpha_sum)
        
        val += log(self._topics[topic][word] + self._lambda)
        val -= log(self._topics[topic].N() + self._lambda_sum)
    
        #print doc, word, topic, self._docs[doc][topic], self._topics[topic][word]
        
        return val

    def sample_word(self, doc, position):
        word = self._data[doc][position]
    
        old_topic = self._state[doc][position]
        if old_topic != -1:
            self.change_count(doc, word, old_topic, -1)
    
        probs = [self.prob(doc, self._data[doc][position], x) for x in xrange(self._K)]
        new_topic = log_sample(probs)
        #print doc, word, new_topic
    
        self.change_count(doc, word, new_topic, 1)
        self._state[doc][position] = new_topic

    def change_count(self, doc, word, topic, delta):
        self._docs[doc].inc(topic, delta)
        self._topics[topic].inc(word, delta)

    def sample(self, iterations=100, hyper_delay=10):
        assert self._state
        for ii in xrange(iterations):
              for dd in self._data:
                    for ww in xrange(len(self._data[dd])):
                        self.sample_word(dd, ww)
              print("Iteration %i %f" % (ii, self.lhood(self._alpha, self._lambda)))
              if hyper_delay >= 0 and ii % hyper_delay == 0:
                  self.optimize_hyperparameters()

    def print_topics(self, num_words=15):
        for ii in self._topics:
            print("%i:%s\n" % (ii, "\t".join(self._topics[ii].keys()[:num_words])))

if __name__ == "__main__":
    #d = create_data("/nfshomes/jbg/sentop/topicmod/data/de_news/txt/*.en.txt", doc_limit=50, delimiter="<doc")
    from topmod.io.de_news_io import parse_de_news, parse_data
    d = parse_de_news("/windows/d/Data/de-news/txt/*.en.txt", doc_limit=1)
    d = parse_data(d)
    
    lda = LDAGibbsSampling(5)
    lda.initialize(d)

    lda.sample(100)
    lda.print_topics()