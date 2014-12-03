#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;

def main():
    import option_parser;
    options = option_parser.parse_args();

    # parameter set 2
    assert(options.number_of_topics>0);
    number_of_topics = options.number_of_topics;
    assert(options.number_of_iterations>0);
    number_of_iterations = options.number_of_iterations;

    # parameter set 3
    alpha_alpha = 1.0/number_of_topics;
    if options.alpha>0:
        alpha_alpha=options.alpha;
    assert(options.eta>0);
    alpha_eta = options.eta;
    
    # parameter set 4
    #disable_alpha_theta_update = options.disable_alpha_theta_update;
    #inference_type = options.hybrid_mode;
    assert(options.snapshot_interval>0);
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    
    # parameter set 1
    #assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    corpus_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);

    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%b%d-%H%M%S") + "";
    suffix += "-%s" % ("lda");
    suffix += "-I%d" % (number_of_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-K%d" % (number_of_topics);
    suffix += "-aa%f" % (alpha_alpha);
    suffix += "-ae%f" % (alpha_eta);
    # suffix += "-%s" % (resample_topics);
    # suffix += "-%s" % (hash_oov_words);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));

    #dict_file = options.dictionary;
    #if dict_file != None:
        #dict_file = dict_file.strip();
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    #options_output_file.write("dictionary_file=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("number_of_iteration=%d\n" % (number_of_iterations));
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("alpha_eta=" + str(alpha_eta) + "\n");
    # parameter set 4
    #options_output_file.write("inference_type=%s\n" % (inference_type));
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");

    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    #print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "number_of_iterations=%d" %(number_of_iterations);
    print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha_alpha=" + str(alpha_alpha)
    print "alpha_eta=" + str(alpha_eta)
    # parameter set 4
    #print "inference_type=%s" % (inference_type)
    print "snapshot_interval=" + str(snapshot_interval);
    print "========== ========== ========== ========== =========="

    # Document
    train_docs = [];
    input_doc_stream = open(os.path.join(input_directory, 'doc.dat'), 'r');
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print "successfully load all training train_docs..."
    
    # Vocabulary
    dictionary_file = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(dictionary_file, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (dictionary_file);
    
    
    import lda
    lda_inference = lda.Hybrid();
    lda_inference._initialize(train_docs, vocab, number_of_topics, alpha_alpha, alpha_eta);
    
    for iteration in xrange(number_of_iterations):
        lda_inference.learning();
        
        if (lda_inference._counter % snapshot_interval == 0):
            lda_inference.export_topic_term_distribution(output_directory + 'exp_beta-' + str(lda_inference._counter));
    
if __name__ == '__main__':
    main()