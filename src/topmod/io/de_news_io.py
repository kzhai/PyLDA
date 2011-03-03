from glob import glob;
from collections import defaultdict;
from topmod.facility.output_function import output_defaultdict_dict, output_dict

#def parse_input(glob_expression, lang="english", doc_limit= -1, delimiter=""):
#    from nltk.tokenize.treebank import TreebankWordTokenizer
#    tokenizer = TreebankWordTokenizer()
#
#    from nltk.corpus import stopwords
#    stop = stopwords.words('english')
#  
#    from string import ascii_lowercase
#  
#    docs = {}
#    files = glob(glob_expression)
#    print("Found %i files" % len(files))
#    for ii in files:
#        text = open(ii).read().lower()
#        if delimiter:
#            sections = text.split(delimiter)
#        else:
#            sections = [text]
#            
#        if doc_limit > 0 and len(docs) > doc_limit:
#            print("Passed doc limit %i" % len(docs))
#            break
#            #print(ii, len(sections))
#
#        for jj in xrange(len(sections)):
#            words = [x for x in tokenizer.tokenize(sections[jj]) \
#                                  if (not x in stop) and (min(y in ascii_lowercase for y in x))]
#            if len(words) != 0:
#                docs["%s-%i" % (ii, jj)] = words
#        
#    return docs

# this method reads in the data from de-news dataset/corpus
# output a dict data type, indexed by the document id
def parse_de_news(glob_expression, lang="english", doc_limit= -1, title = True, path = False):
    from nltk.tokenize.treebank import TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()

    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    if lang.lower() == "english":
        stop = stopwords.words('english')
    elif lang.lower() == "german":
        stop = stopwords.words('german')
    else:
        print "language option unspecified, default to english..."
          
    from string import ascii_lowercase
  
    docs = {}
    files = glob(glob_expression)
    print("Found %i files" % len(files))
    for ii in files:
        text = open(ii).read().lower()
        
        sections = text.split("<doc")
        
        for section in sections:
            if section != None and len(section) != 0:
                index_content = section.split(">\n<h1>\n")
                title_content = index_content[1].split("</h1>")
                # not x in stop: to remove the stopwords
                # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                words = [x for x in tokenizer.tokenize(title_content[1]) if (not x in stop) and (min(y in ascii_lowercase for y in x))]
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

# this method convert a corpus into proper format for lda model
# output a defaultdict(dict) data type, first indexed by the document id, then indexed by the unique tokens
# corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
def parse_data(corpus):
    docs = defaultdict(dict)
    
    for doc in corpus.keys():
        content = {}
        for term in corpus[doc]:
            if term in content.keys():
                content[term] = content[term] + 1
            else:
                content[term] = 1
        docs[doc] = content
    
    return docs

def map_corpus(corpus_a, corpus_b):
    common_docs = (set(corpus_a.keys()) & set(corpus_b.keys()));
   
    for doc in corpus_a.keys():
        if doc not in common_docs:
            del corpus_a[doc]
            
    for doc in corpus_b.keys():
        if doc not in common_docs:
            del corpus_b[doc]
            
    return corpus_a, corpus_b

def output_param(alpha, beta, gamma, dir, index=-1):
    if index!=-1:
        postfix = str(index)
    else:
        postfix = ""

    alpha_path = dir + "alpha" + postfix
    f = open(alpha_path, "w");
    for k in alpha.keys():
        f.write(str(k) + "\t" + str(alpha[k]) + "\n")
        
    beta_path = dir + "beta" + postfix
    f = open(beta_path, "w");
    for term in beta.keys():
        for k in beta[term].keys():
            f.write(str(term) + "\t" + str(k) + "\t" + str(beta[term][k]) + "\n")
        
    gamma_path = dir + "gamma" + postfix
    f = open(gamma_path, "w");
    for doc in gamma.keys():
        for k in gamma[doc].keys():
            f.write(str(doc) + "\t" + str(k) + "\t" + str(gamma[doc][k]) + "\n")
            
def output(d):
    f = open("/windows/d/Workspace/data/test_data", "w");
    
    terms = [];
    for value in d.values():
        terms = terms+value;
    terms = set(terms);
    
    termID = {}
    i = 0;
    for t in terms:
        termID[t] = i;
        i+=1;
    
    i = 1;
    for doc in d.keys():
        if len(d[doc])==0:
            continue;
        f.write(str(i) + "\t");
        temp = set(d[doc]);
        for t in temp:
            f.write(str(termID[t])+ "\t"+str(d[doc].count(t))+"\t");
        f.write("\n");
        i+=1;

if __name__ == "__main__":
    data_en = parse_de_news("/windows/d/Data/de-news/txt/*.en.txt", "english",
                  500, False)
    data_en = parse_data(data_en)
    data_de = parse_de_news("/windows/d/Data/de-news/txt/*.de.txt", "german",
                  500, False)
    data_de = parse_data(data_de)
    print len(data_en), "\t", len(data_de)
    
    [data_en, data_de] = map_corpus(data_en, data_de)
    print len(data_en), "\t", len(data_de)
    
#lda.initialize(d)

#lda.sample(100)
#lda.print_topics()