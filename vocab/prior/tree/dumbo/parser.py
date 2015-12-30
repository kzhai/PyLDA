#!/usr/bin/python
import cPickle, string, sys;
import datetime, os;

import codecs;

from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

def main():
    input_file = sys.argv[1];
    output_file = sys.argv[2];
    
    counter=1;
    input_stream = codecs.open(input_file, 'r', 'utf-8');
    output_stream = codecs.open(output_file, 'w', 'utf-8');
    for line in input_stream:
        output_stream.write("%d\t%s\n" % (counter, line.strip()));
        counter += 1;

if __name__ == '__main__':
    main()