from glob import glob;
from collections import defaultdict;

# this method reads in the data from de-news dataset/corpus
# output a dict data type, indexed by the document id
def parse_de_news(glob_expression, doc_limit= -1, title = True, path = False):
    from nltk.tokenize.punkt import PunktWordTokenizer 
    tokenizer = PunktWordTokenizer()

    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer();
    
    #if lang.lower() == "english":
    #    stop = stopwords.words('english')
    #    from nltk.stem.porter import PorterStemmer
    #    stemmer = PorterStemmer();
    #elif lang.lower() == "german":
    #    stop = stopwords.words('german')
    #    from nltk.stem.snowball import GermanStemmer
    #    stemmer = GermanStemmer();
    #else:
    #    print "language option unspecified, default to english..."
          
    from string import ascii_lowercase
  
    docs = {}
    files = glob(glob_expression)
    print("Found %i files" % len(files))
    for ii in files:
        #import codecs
        #input = codecs.open(ii, "utf-8");
        text = open(ii).read().lower()
        
        sections = text.split("<doc")
        
        for section in sections:
            if section != None and len(section) != 0:
                index_content = section.split(">\n<h1>\n")
                title_content = index_content[1].split("</h1>")
                # not x in stop: to remove the stopwords
                # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                words = [stemmer.stem(x) for x in tokenizer.tokenize(title_content[1]) if (min(y in ascii_lowercase for y in x))];
                words = [x for x in words if not x in stop];
                if path:
                    if title:
                        docs["%s\t%s\t%s" % (ii, index_content[0].strip(), title_content[0].strip())] = words
                    else:
                        docs["%s\t%s" % (ii, index_content[0].strip())] = words
                else:
                    if title:
                        docs["%s\t%s" % (index_content[0].strip(), title_content[0].strip())] = words
                    else:
                        docs["%s" % (index_content[0].strip())] = words
                                        
        if doc_limit > 0 and len(docs) > doc_limit:
            print("Passed doc limit %i" % len(docs))
            break
    
    return docs

"""
generated vocabuary from a corpus
"""
def generate_vocab(corpus, min_df=0, max_df=1):
    from nltk.probability import FreqDist;
    vocab = FreqDist();

    for doc_id in corpus.keys():
        for term in list(set(corpus[doc_id])):
            vocab.inc(term);
    
    return [item for item in vocab.keys() if vocab[item] < max_df*len(corpus.keys()) and vocab[item] > min_df*len(corpus.keys())];

"""
"""
def output_parsed_corpus(corpus, vocab, directory="../../data/tmp-corpus/"):
    assert(directory.endswith("/"));
    vocab_mapping = {};
    output = open(directory + "voc.dat", "w");
    output.write("\n".join(vocab).strip());
    
    index=0;
    for term in vocab:
        vocab_mapping[term] = index;
        index+=1;
        
    output = open(directory + "doc.dat", "w");
    index = 0;
    for doc_id in corpus.keys():
        valid_terms = [term for term in corpus[doc_id] if term in vocab_mapping.keys()];
        if len(valid_terms)==0:
            continue;
        output.write(str(index) + "\t");
        output.write(" ".join([str(vocab_mapping[term]) for term in valid_terms]));
        index+=1;
        if index<len(corpus.keys()):
            output.write("\n");

if __name__ == "__main__":
    doc = parse_de_news("../../data/de-news-raw/txt/*.en.txt",
                  100, True, False)
    voc = generate_vocab(doc);
    output_parsed_corpus(doc, voc, "../../data/de-news/en/");