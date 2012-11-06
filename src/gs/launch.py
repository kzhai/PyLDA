#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;

from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

def parse_data(documents_file, vocabulary_file=None):
    import codecs
    
    type_to_index = {};
    index_to_type = {};
    if (vocabulary_file!=None):
        input_file = codecs.open(vocabulary_file, mode='r', encoding='utf-8');
        for line in input_file:
            line = line.strip().split()[0];
            assert line not in type_to_index, "duplicate type for %s" % line;
            type_to_index[line] = len(type_to_index);
            index_to_type[len(index_to_type)] = line;
        input_file.close();

    input_file = codecs.open(documents_file, mode="r", encoding="utf-8")

    doc_count = 0
    documents = [];
    #documents = {};
    
    for line in input_file:
        line = line.strip().lower();

        contents = line.split("\t");

        document = [];
        for token in contents[-1].split():
            if token not in type_to_index:
                if vocabulary_file==None:
                    type_to_index[token] = len(type_to_index);
                    index_to_type[len(index_to_type)] = token;
                else:
                    continue;
                    
            document.append(type_to_index[token]);
        
        assert len(document)>0, "document %d collapsed..." % doc_count;
        
        documents.append(document);
        #if len(contents)==2:
            #documents[int(contents[0])] = document;
        #elif len(contents)==1:
            #documents[doc_count] = document;
        #else:
            #print ""  

        doc_count+=1
        if doc_count%10000==0:
            print "successfully import %d documents..." % doc_count;
    
    input_file.close();

    print "successfully import", len(documents), "documents..."
    return documents, type_to_index, index_to_type;

def main():
    import option_parser;
    options = option_parser.parse_args();

    # parameter set 2
    assert(options.number_of_topics>0);
    number_of_topics = options.number_of_topics;
    assert(options.number_of_iterations>0);
    number_of_iterations = options.number_of_iterations;

    # parameter set 3
    alpha = 1.0/number_of_topics;
    if options.alpha>0:
        alpha=options.alpha;
    assert(options.beta>0);
    beta = options.beta;
    
    # parameter set 4
    #disable_alpha_theta_update = options.disable_alpha_theta_update;
    #inference_type = options.hybrid_mode;
    assert(options.snapshot_interval>0);
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    
    # parameter set 1
    assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);

    corpus_name = options.corpus_name;

    input_directory = options.input_directory;
    if not input_directory.endswith('/'):
        input_directory += '/';
    input_directory += corpus_name+'/';
        
    output_directory = options.output_directory;
    if not output_directory.endswith('/'):
        output_directory += '/';
    output_directory += corpus_name+'/';
     
    # create output directory
    now = datetime.datetime.now();
    output_directory += now.strftime("%y%b%d-%H%M%S")+"";
    #output_directory += "-" + str(now.microsecond) + "/";
    output_directory += "-K%d-I%d-a%g-b%g-S%d/" \
                        % (number_of_topics,
                           number_of_iterations,
                           alpha,
                           beta,
                           snapshot_interval);

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
    options_output_file.write("alpha=" + str(alpha) + "\n");
    options_output_file.write("beta=" + str(beta) + "\n");
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
    print "alpha=" + str(alpha)
    print "beta=" + str(beta)
    # parameter set 4
    #print "inference_type=%s" % (inference_type)
    print "snapshot_interval=" + str(snapshot_interval);
    print "========== ========== ========== ========== =========="

    documents, type_to_index, index_to_type = parse_data(input_directory+'doc.dat', input_directory+'voc.dat');
    print "successfully load all training documents..."

    import cgs;
    lda_inference = cgs.CollapsedGibbsSampling()
    lda_inference._initialize(documents, type_to_index, index_to_type, number_of_topics, alpha, beta);
    
    for iteration in xrange(number_of_iterations):
        lda_inference.sample();
        
        if (lda_inference._counter % snapshot_interval == 0):
            lda_inference.export_topic_term_distribution(output_directory + 'exp_beta-' + str(lda_inference._counter));

if __name__ == '__main__':
    main()