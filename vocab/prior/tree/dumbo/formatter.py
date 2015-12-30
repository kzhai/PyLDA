#!/usr/bin/python

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import codecs
import glob

try:
    import cPickle as pickle
except:
    import pickle

def main():
    input_directory = sys.argv[1];
    output_directory = sys.argv[2];
    if len(sys.argv)==4 and len(sys.argv[3])>0:
        log_beta_path = sys.argv[3]
    else:
        log_beta_path = None;

    from inferencer import Inferencer;
    lda_inferencer = Inferencer();
    lda_inferencer.load_params(os.path.join(output_directory, "current-params"));
    lda_inferencer.load_tree(os.path.join(output_directory, "current-tree"));
    lda_inferencer.format_output(input_directory);
    
    if lda_inferencer._update_hyper_parameter:
        lda_inferencer.dump_parameters(os.path.join(output_directory, "current-params"));
    
    lda_inferencer.dump_E_log_beta(os.path.join(output_directory, "current-E-log-beta"));

    if log_beta_path!=None:
        lda_inferencer.export_E_log_beta(log_beta_path);

    #if not hybrid_mode:
        #lda_inferencer.dump_gamma(os.path.join(output_directory, "current-gamma"));

    lda_inferencer.export_gamma(os.path.join(output_directory, "gamma"));

if __name__ == '__main__':
    main()
