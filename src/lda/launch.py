#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import optparse;

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        #dictionary=None,
                        
                        # parameter set 2
                        training_iterations=-1,
                        snapshot_interval=10,
                        number_of_topics=-1,

                        # parameter set 3
                        alpha_alpha=-1,
                        alpha_eta=1e-12,
                        
                        # parameter set 4
                        #disable_alpha_theta_update=False,
                        inference_mode=-1,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    #parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      #help="the corpus name [None]")
    #parser.add_option("--dictionary", type="string", dest="dictionary",
                      #help="the dictionary file [None]")
    
    # parameter set 2
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="total number of topics [-1]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="total number of iterations [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [vocab_prune_interval]");                      
                      
    # parameter set 3
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="hyper-parameter for Dirichlet distribution of topics [1.0/number_of_topics]")
    parser.add_option("--alpha_eta", type="float", dest="alpha_eta",
                      help="hyper-parameter for Dirichlet distribution of vocabulary [1e-12]")
    
    # parameter set 4
    #parser.add_option("--disable_alpha_theta_update", action="store_true", dest="disable_alpha_theta_update",
                      #help="disable alpha (hyper-parameter for Dirichlet distribution of topics) update");
    parser.add_option("--inference_mode", type="int", dest="inference_mode",
                      help="inference mode [ " + 
                            "0: hybrid inference, " + 
                            "1: monte carlo, " + 
                            "2: variational bayes " + 
                            "]");
    #parser.add_option("--inference_mode", action="store_true", dest="inference_mode",
    #                  help="run latent Dirichlet allocation in lda mode");

    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();

    # parameter set 2
    assert(options.number_of_topics>0);
    number_of_topics = options.number_of_topics;
    assert(options.training_iterations>0);
    training_iterations = options.training_iterations;
    assert(options.snapshot_interval>0);
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
        
    # parameter set 3
    alpha_alpha = 1.0/number_of_topics;
    if options.alpha_alpha>0:
        alpha_alpha=options.alpha_alpha;
    assert(options.alpha_eta>0);
    alpha_eta = options.alpha_eta;
    
    # parameter set 4
    #disable_alpha_theta_update = options.disable_alpha_theta_update;
    inference_mode = options.inference_mode;
    
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
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("lda");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-K%d" % (number_of_topics);
    suffix += "-aa%f" % (alpha_alpha);
    suffix += "-ae%f" % (alpha_eta);
    suffix += "-im%d" % (inference_mode);
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
    #options_output_file.write("vocabulary_path=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("training_iterations=%d\n" % (training_iterations));
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("alpha_eta=" + str(alpha_eta) + "\n");
    # parameter set 4
    options_output_file.write("inference_mode=%d\n" % (inference_mode));
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    #print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "training_iterations=%d" %(training_iterations);
    print "snapshot_interval=" + str(snapshot_interval);
    print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha_alpha=" + str(alpha_alpha)
    print "alpha_eta=" + str(alpha_eta)
    # parameter set 4
    print "inference_mode=%d" % (inference_mode)
    print "========== ========== ========== ========== =========="

    # Document
    train_docs_path = os.path.join(input_directory, 'doc.dat')
    input_doc_stream = open(train_docs_path, 'r');
    train_docs = [];
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print "successfully load all training train docs from %s..." % (os.path.abspath(train_docs_path));
    
    # Vocabulary
    vocabulary_path = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(vocabulary_path, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (os.path.abspath(vocabulary_path));
    
    if inference_mode==0:
        import hybrid
        lda_inference = hybrid.Hybrid();
    elif inference_mode==1:
        import monte_carlo
        lda_inference = monte_carlo.MonteCarlo();
    elif inference_mode==2:
        import variational_bayes
        lda_inference = variational_bayes.VariationalBayes();
    else:
        sys.stderr.write("error: unrecognized inference mode %d...\n" % (inference_mode));
        return;
    
    lda_inference._initialize(train_docs, vocab, number_of_topics, alpha_alpha, alpha_eta);
    
    for iteration in xrange(training_iterations):
        lda_inference.learning();
        
        if (lda_inference._counter % snapshot_interval == 0):
            lda_inference.export_beta(output_directory + 'exp_beta-' + str(lda_inference._counter));
    
if __name__ == '__main__':
    main()