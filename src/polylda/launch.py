#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;
import collections;

import scipy.io;
import nltk;
import numpy;

def parse_data(documents_file):
    import codecs
    
    language_type_to_index = collections.defaultdict(dict);
    language_index_to_type = collections.defaultdict(dict);
    
    input_file = codecs.open(documents_file, mode="r", encoding="utf-8")
    
    doc_count = 0
    documents = [];
    
    for line in input_file:
        line = line.strip().lower();
        
        contents = line.split("\t");
        
        document = [];
        for language_id in xrange(len(contents)):
            language_content = contents[language_id];
            document_language = [];
            for token in language_content.split():
                if token not in language_type_to_index[language_id]:
                    language_type_to_index[language_id][token] = len(language_type_to_index[language_id]);
                    language_index_to_type[language_id][len(language_index_to_type[language_id])] = token;
                        
                document_language.append(language_type_to_index[language_id][token]);
                
            document.append(document_language);
        
        assert len(document)>0, "document %d collapsed..." % doc_count;
        
        documents.append(document);
        
        doc_count+=1
        if doc_count%10000==0:
            print "successfully import %d documents..." % doc_count;
    
    input_file.close();
    
    print "successfully import", len(documents), "documents..."
    return documents, language_type_to_index, language_index_to_type;

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
    if options.alpha_alpha>0:
        alpha_alpha=options.alpha_alpha;
    assert(options.alpha_beta>0);
    alpha_beta = options.alpha_beta;
    
    # parameter set 4
    #disable_alpha_theta_update = options.disable_alpha_theta_update;
    #inference_type = options.hybrid_mode;
    assert(options.snapshot_interval>0);
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    
    # parameter set 1
    #assert(options.dataset_name!=None);
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
    suffix = now.strftime("%y%b%d-%H%M%S");
    #output_directory += "-" + str(now.microsecond) + "/";
    suffix += "-polylda"
    suffix += "-K%d" % (number_of_topics)
    suffix += "-I%d" % (number_of_iterations)
    suffix += "-a%g" % (alpha_alpha)
    suffix += "-b%g" % (alpha_beta)
    suffix += "-S%d" % (snapshot_interval)
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    # store all the options to a file
    options_output_file = open(os.path.join(output_directory, "option.txt"), 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    #options_output_file.write("dictionary_file=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("number_of_iteration=%d\n" % (number_of_iterations));
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("alpha_beta=" + str(alpha_beta) + "\n");
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
    print "alpha_beta=" + str(alpha_beta)
    # parameter set 4
    #print "inference_type=%s" % (inference_type)
    print "snapshot_interval=" + str(snapshot_interval);
    print "========== ========== ========== ========== =========="
    
    documents, language_type_to_index, language_index_to_type = parse_data(os.path.join(input_directory, 'doc.dat'));
    print "successfully load all training documents..."
    
    from polylda import monte_carlo
    lda_inference = monte_carlo.CollapsedGibbsSampling()
    lda_inference._initialize(documents, language_type_to_index, language_index_to_type, number_of_topics, alpha_alpha, alpha_beta);
    
    for iteration in xrange(number_of_iterations):
        lda_inference.learning();
        
        if (lda_inference._counter % snapshot_interval == 0):
            lda_inference.export_exp_beta(output_directory);

if __name__ == '__main__':
    main()