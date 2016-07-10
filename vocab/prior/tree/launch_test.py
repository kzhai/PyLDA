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
                        model_directory=None,
                        snapshot_index=-1,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="model directory [None]");
    parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
                      help="snapshot index [-: evaluate on all available snapshots]");
    
    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();
# parameter set 1
    #assert(options.input_corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.model_directory!=None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    input_corpus_name = os.path.basename(input_directory);
    
    model_directory = options.model_directory;
    model_directory = model_directory.rstrip("/");
    if not os.path.exists(model_directory):
        sys.stderr.write("error: model directory %s does not exist...\n" % (os.path.abspath(model_directory)));
        return;
    corpus_directory = os.path.split(os.path.abspath(model_directory))[0];
    model_corpus_name = os.path.split(os.path.abspath(corpus_directory))[1]
    if input_corpus_name!=model_corpus_name:
        sys.stderr.write("error: corpus name does not match for input (%s) and model (%s)...\n" % (input_corpus_name, model_corpus_name));
        return;
    
    
    

    snapshot_index=options.snapshot_index;

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + input_corpus_name
    print "snapshot_index=" + str(snapshot_index);
    print "========== ========== ========== ========== =========="

    # Document
    test_docs_path = os.path.join(input_directory, 'test.dat')
    input_doc_stream = open(test_docs_path, 'r');
    test_docs = [];
    for line in input_doc_stream:
        test_docs.append(line.strip().lower());
    print "successfully load all testing docs from %s..." % (os.path.abspath(test_docs_path));
    
    if snapshot_index>=0:
        input_snapshot_path = os.path.join(model_directory, ("model-%d" % (snapshot_index)))
        if not os.path.exists(input_snapshot_path):
            sys.stderr.write("error: model snapshot %s does not exist...\n" % (os.path.abspath(input_snapshot_path)));
            return;
        
        output_gamma_path = os.path.join(model_directory, "test-%d" % snapshot_index);
        
        evaluate_snapshot(input_snapshot_path, test_docs, output_gamma_path);
    else:
        for model_snapshot in os.listdir(model_directory):
            if not model_snapshot.startswith("model-"):
                continue;
            
            snapshot_index = int(model_snapshot.split("-")[-1]);
            
            input_snapshot_path = os.path.join(model_directory, model_snapshot);
            output_gamma_path = os.path.join(model_directory, "test-%d" % snapshot_index);
            
            evaluate_snapshot(input_snapshot_path, test_docs, output_gamma_path)

def evaluate_snapshot(input_snapshot_path, test_docs, output_gamma_path):
    #import hybrid, monte_carlo, variational_bayes;
    lda_inferencer = cPickle.load(open(input_snapshot_path, "rb" ));
    #print 'successfully load model snapshot %s...' % (os.path.abspath(input_snapshot_path));
    
    log_likelihood, gamma_values = lda_inferencer.inference(test_docs);
    print "held-out likelihood of snapshot %s is %g" % (os.path.abspath(input_snapshot_path), log_likelihood);
    numpy.savetxt(output_gamma_path, gamma_values);

if __name__ == '__main__':
    main()
