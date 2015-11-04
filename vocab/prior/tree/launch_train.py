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
                        # corpus_name=None,
                        tree_name=None,
                        
                        # parameter set 2
                        training_iterations=-1,
                        number_of_topics=-1,

                        # parameter set 3
                        alpha_alpha=-1,
                        # default_correlation_prior=1e-2,
                        # positive_correlation_prior=1e2,
                        # negative_correlation_prior=1e-12,
                        # eta=1e-12,
                        
                        # parameter set 4
                        inference_mode=0,
                        #update_hyperparameter=False,
                        # inference_type="cgs",
                        
                        # parameter set 5
                        snapshot_interval=10
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    #parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      #help="the corpus name [None]")
    parser.add_option("--tree_name", type="string", dest="tree_name",
                      help="the tree_name file [None]")
    
    # parameter set 2
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="total number of topics [-1]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="total number of iterations [-1]");
                      
    # parameter set 3
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="hyper-parameter for Dirichlet distribution of topics [1.0/number_of_topics]")
    # parser.add_option("--positive_correlation_prior", type="float", dest="positive_correlation_prior",
                      # help="positive correlation prior between words [1e2]")
    # parser.add_option("--negative_correlation_prior", type="float", dest="negative_correlation_prior",
                      # help="negative correlation prior between words [1e-12]")
    # parser.add_option("--default_correlation_prior", type="float", dest="default_correlation_prior",
                      # help="normal correlation prior between words [1e-2]")
    # parser.add_option("--eta", type="float", dest="eta",
                      # help="hyper-parameter for Dirichlet distribution of vocabulary [1e-12]")
    
    # parameter set 4
    parser.add_option("--inference_mode", type="int", dest="inference_mode",
                      help="inference mode [ " + 
                            "0 (default): hybrid inference, " + 
                            "1: monte carlo, " + 
                            "2: variational bayes " + 
                            "]");
    #parser.add_option("--update_hyperparameter", action="store_true", dest="update_hyperparameter",
                      #help="update alpha_alpha (hyper-parameter for topic Dirichlet distribution)");
    # parser.add_option("--inference_type", type="string", dest="inference_type",
                      # help="inference type [cgs] cgs-CollapsedGibbsSampling uvb-UncollapsedVariationalBayes hybrid-HybridMode");
    # parser.add_option("--inference_type", action="store_true", dest="inference_type",
    #                  help="run latent Dirichlet allocation in hybrid mode");
    
    # parameter set 5
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [vocab_prune_interval]");

    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();

    # parameter set 2
    assert(options.number_of_topics > 0);
    number_of_topics = options.number_of_topics;
    assert(options.training_iterations > 0);
    training_iterations = options.training_iterations;

    # parameter set 3
    alpha_alpha = 1.0 / number_of_topics;
    if options.alpha_alpha > 0:
        alpha_alpha = options.alpha_alpha;
    
    # assert options.default_correlation_prior>0;
    # default_correlation_prior = options.default_correlation_prior;
    # assert options.positive_correlation_prior>0;
    # positive_correlation_prior = options.positive_correlation_prior;
    # assert options.negative_correlation_prior>0;
    # negative_correlation_prior = options.negative_correlation_prior;
    
    # parameter set 4
    # disable_alpha_theta_update = options.disable_alpha_theta_update;
    inference_mode = options.inference_mode;
    #update_hyperparameter = options.update_hyperparameter;
    
    # parameter set 5
    assert(options.snapshot_interval > 0);
    if options.snapshot_interval > 0:
        snapshot_interval = options.snapshot_interval;
    
    # parameter set 1
    # assert(options.corpus_name!=None);
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    assert(options.tree_name != None);
    tree_name = options.tree_name;
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    corpus_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    # Document
    train_docs_path = os.path.join(input_directory, 'train.dat')
    input_doc_stream = open(train_docs_path, 'r');
    train_docs = [];
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print "successfully load all training docs from %s..." % (os.path.abspath(train_docs_path));
    
    # Vocabulary
    vocabulary_path = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(vocabulary_path, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (os.path.abspath(vocabulary_path));

    '''
    # create output directory
    now = datetime.datetime.now();
    output_directory += now.strftime("%y%b%d-%H%M%S") + "";
    output_directory += "-prior_tree-K%d-I%d-a%g-S%d-%s-%s-%s/" \
                        % (number_of_topics,
                           training_iterations,
                           alpha_alpha,
                           snapshot_interval,
                           tree_name,
                           inference_mode,
                           update_hyperparameter);
    '''

    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("lda");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-K%d" % (number_of_topics);
    suffix += "-aa%f" % (alpha_alpha);
    #suffix += "-ab%f" % (alpha_beta);
    suffix += "-im%d" % (inference_mode);
    # suffix += "-%s" % (resample_topics);
    # suffix += "-%s" % (hash_oov_words);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));

    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("tree_name=" + str(tree_name) + "\n");
    # parameter set 2
    options_output_file.write("number_of_iteration=%d\n" % (training_iterations));
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    # options_output_file.write("default_correlation_prior=" + str(default_correlation_prior) + "\n");
    # options_output_file.write("positive_correlation_prior=" + str(positive_correlation_prior) + "\n");
    # options_output_file.write("negative_correlation_prior=" + str(negative_correlation_prior) + "\n");
    # parameter set 4
    options_output_file.write("inference_mode=%d\n" % (inference_mode));
    #options_output_file.write("update_hyperparameter=%s\n" % (update_hyperparameter));
    # parameter set 5
    #options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");

    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    print "tree prior file=" + str(tree_name)
    # parameter set 2
    print "training_iterations=%d" % (training_iterations);
    print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha_alpha=" + str(alpha_alpha)
    # print "default_correlation_prior=" + str(default_correlation_prior)
    # print "positive_correlation_prior=" + str(positive_correlation_prior)
    # print "negative_correlation_prior=" + str(negative_correlation_prior)
    # parameter set 4
    print "inference_mode=%d" % (inference_mode)
    #print "update_hyperparameter=%s" % (update_hyperparameter);
    # parameter set 5
    #print "snapshot_interval=" + str(snapshot_interval);
    print "========== ========== ========== ========== =========="

    if inference_mode==0:
        import hybrid;
        lda_inferencer = hybrid.Hybrid();
        #lda_inferencer = hybrid.Hybrid(update_hyperparameter);
        #import hybrid.parse_data as parse_data
    elif inference_mode==1:
        #import monte_carlo
        #lda_inferencer = monte_carlo.MonteCarlo();
        sys.stderr.write("warning: monte carlo inference method is not implemented yet...\n");
        pass
    elif inference_mode==2:
        #from prior.tree import variational_bayes
        #lda_inferencer = variational_bayes.VariationalBayes();
        import variational_bayes
        lda_inferencer = variational_bayes.VariationalBayes();
        #lda_inferencer = variational_bayes.VariationalBayes(update_hyperparameter);
        #from variational_bayes import parse_data
    else:
        sys.stderr.write("error: unrecognized inference mode %d...\n" % (inference_mode));
        return;
    
    # initialize tree
    import priortree
    prior_tree = priortree.VocabTreePrior();
    # from vb.prior.tree.priortree import VocabTreePrior;
    # prior_tree = VocabTreePrior();
    # prior_tree._initialize(input_directory+"tree.wn.*", vocab, default_correlation_prior, positive_correlation_prior, negative_correlation_prior);
    prior_tree._initialize(os.path.join(input_directory, tree_name + ".wn.*"), os.path.join(input_directory, tree_name + ".hyperparams"), vocab)

    lda_inferencer._initialize(train_docs, vocab, prior_tree, number_of_topics, alpha_alpha);
    
    for iteration in xrange(training_iterations):
        lda_inferencer.learning();
        
        if (lda_inferencer._counter % snapshot_interval == 0):
            lda_inferencer.export_beta(os.path.join(output_directory, 'exp_beta-' + str(lda_inferencer._counter)));
            model_snapshot_path = os.path.join(output_directory, 'model-' + str(lda_inferencer._counter));
            cPickle.dump(lda_inferencer, open(model_snapshot_path, 'wb'));
    
    model_snapshot_path = os.path.join(output_directory, 'model-' + str(lda_inferencer._counter));
    cPickle.dump(lda_inferencer, open(model_snapshot_path, 'wb'));

if __name__ == '__main__':
    main()
