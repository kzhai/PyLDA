#!/usr/bin/python

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import codecs

try:
    import cPickle as pickle
except:
    import pickle

def load_vocabulary(vocabulary_file):
    index_to_type = {};
    type_to_index = {};

    # generate unicode vocabulary
    input_file = codecs.open(vocabulary_file, mode='r', encoding='utf-8');
    for line in input_file:
        #line = line.strip().split()[0];
        assert len(line.strip().split())==4;
        line = line.strip().split()[1];
        assert line not in type_to_index, "duplicate type for %s" % line;
        #if line in type_to_index:
            #continue;

        # add in to force the string to be utf-8 coded...
        line = unicode(line).encode('utf-8');

        type_to_index[line] = len(type_to_index);
        index_to_type[len(index_to_type)] = line;
        #vocabulary.append(line);
    input_file.close();
    
    # generate non-unicode vocabulary
    vocabulary = [];
    input_file = open(vocabulary_file, mode='r');
    for line in input_file:
        #line = line.strip().split()[0];
        assert len(line.strip().split())==4;
        line = line.strip().split()[1];
        assert line not in vocabulary, "duplicate type for %s" % line;
        #if line in vocabulary:
            #continue;
        
        #type_to_index[line] = len(type_to_index);
        #index_to_type[len(index_to_type)] = line;
        vocabulary.append(line);
    input_file.close();

    assert len(type_to_index)==len(vocabulary);
    assert len(index_to_type)==len(vocabulary);

    return type_to_index, index_to_type, vocabulary

'''
def add_document_id(input_file, output_file):
    counter=1;
    input_stream = codecs.open(input_file, 'r', 'utf-8');
    output_stream = codecs.open(output_file, 'w', 'utf-8');
    for line in input_stream:
        output_stream.write("%d\t%s\n" % (counter, line.strip()));
        counter += 1;
'''

def main():
    #from vb.prior.tree.backup.option_parser import option_parser
    import vb.prior.tree.option_parser as option_parser
    options = option_parser.parse_args();

    # parameter set 2
    assert(options.number_of_topics>0);
    number_of_topics = options.number_of_topics;

    # parameter set 3
    alpha = 1.0/number_of_topics;
    if options.alpha>0:
        alpha=options.alpha;
    
    # parameter set 4
    inference_mode = options.hybrid_mode;
    update_hyperparameter = options.update_hyperparameter;

    # parameter set 1
    #assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);
    assert(options.tree_name!=None);

    input_directory = options.input_directory;
    if not input_directory.endswith('/'):
        input_directory += '/';
    #input_directory += corpus_name+'/';

    tree_name = options.tree_name.strip();
        
    output_directory = options.output_directory;
    #if not os.path.exists(output_directory):
        #os.mkdir(output_directory);
    if not output_directory.endswith('/'):
        output_directory += '/';
    #output_directory += corpus_name+'/';
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);

    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    #options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("tree_name=" + str(tree_name) + "\n");
    # parameter set 2
    #options_output_file.write("number_of_iteration=%d\n" % (number_of_iterations));
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha=" + str(alpha) + "\n");
    #options_output_file.write("default_correlation_prior=" + str(default_correlation_prior) + "\n");
    #options_output_file.write("positive_correlation_prior=" + str(positive_correlation_prior) + "\n");
    #options_output_file.write("negative_correlation_prior=" + str(negative_correlation_prior) + "\n");
    # parameter set 4
    options_output_file.write("inference_mode=%s\n" % (inference_mode));
    options_output_file.write("update_hyperparameter=%s\n" % (update_hyperparameter));
    #options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");

    options_output_file.close()

    type_to_index, index_to_type, vocabulary = load_vocabulary(input_directory+'voc.dat');

    from vb.prior.tree.priortree import VocabTreePrior;
    prior_tree = VocabTreePrior();
    prior_tree._initialize(input_directory+tree_name+".wn.*", input_directory+tree_name+".hyperparams", vocabulary);

    import inferencer
    lda_inferencer = inferencer.Inferencer(update_hyperparameter, inference_mode);
    lda_inferencer._initialize(prior_tree, type_to_index, index_to_type, number_of_topics, alpha);
    lda_inferencer.dump_tree(os.path.join(output_directory, "current-tree"));
    lda_inferencer.dump_parameters(os.path.join(output_directory, "current-params"));
    #lda_inferencer.load_params(os.path.join(output_directory, "current-params"));
    
    lda_inferencer.dump_E_log_beta(os.path.join(output_directory, "current-E-log-beta"));
    
    #if not inference_mode:
        #lda_inferencer.dump_gamma(os.path.join(output_directory, "current-gamma"));

if __name__ == '__main__':
    main()
