import sys, os

#sys.path.append("/fs/clip-sw/rhel6/Scipy_Stack-1.0/lib/python2.7/site-packages/");
sys.path.append("/opt/local/stow/python-commonmodules-2.7.2.0/lib/python2.7/site-packages/");
#sys.path.append("/fs/clip-software/python-contrib-2.7.1.0/lib/python2.7/site-packages/");
#sys.path.append("/fs/cliplab/software/python-contrib-2.7.1.0/lib/python2.7/site-packages/")
sys.path.append("/chomes/jbg/nltk_svn");
sys.path.append("/fs/clip-software/protobuf-2.3.0b/python/lib/python2.6/site-packages");
sys.path.append("/fs/cliplab/software/protobuf-2.3.0b/python/lib/python2.6/site-packages");
sys.path.append("/fs/cliplab/software/PyStemmer/`arch`/lib/python2.5/site-packages/");
sys.path.append("/fs/cliplab/software/pyeditdistance/`arch`/lib/python2.5/site-packages/");
sys.path.append("/cliphomes/zhaike/.local/lib/python-2.7/site-packages/");
sys.path.append("/cliphomes/zhaike/Workspace/topicmod/lib/python_lib");
sys.path.append("/cliphomes/zhaike/Workspace/topicmod/projects/variational/src/");
sys.path.append("/fs/clip-software/yamcha-0.33/`arch`/bin/");
sys.path.append("/usr/lib/");
sys.path.append("/usr/lib64/");
sys.path.append("/opt/UMtorq_number_of_topicsath.app_number_of_topicsftware/protobuf-2.3.0b/`arch`/bin/");
sys.path.append("/fs/clip-software/gsl-1.13/`arch`/bin/");
sys.path.append("/fs/clip-software/yamcha-0.33/`arch`/bin/");
sys.path.append("/fs/cliplab/software/svm_light-6.02/`arch`/");
sys.path.append("/cliphomes/zhaike/.local/bin/");

import time
import numpy
import scipy
import nltk;
import sys;
import codecs;
import glob;

try:
    import cPickle as pickle
except:
    import pickle

from collections import defaultdict;

class Inferencer():
    def __init__(self,
                 update_hyper_parameter=False,
                 inference_mode=True,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001
                 ):
        self._update_hyper_parameter = update_hyper_parameter;
        self._hybrid_mode = inference_mode;
        
        self._alpha_update_decay_factor = alpha_update_decay_factor;
        self._alpha_maximum_decay = alpha_maximum_decay;
        self._alpha_converge_threshold = alpha_converge_threshold;
        self._alpha_maximum_iteration = alpha_maximum_iteration;
        
        self._model_likelihood_threhold = model_likelihood_threshold;
    
    def compute_dirichlet_expectation(self, dirichlet_parameter):
        if (len(dirichlet_parameter.shape) == 1):
            return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
        return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:, numpy.newaxis]
    
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, prior_tree, type_to_index, index_to_type, number_of_topics, alpha):
        self._counter = 0;
        
        #self._hybrid_mode = inference_mode;
        
        self._type_to_index = type_to_index;
        self._index_to_type = index_to_type;
        
        # initialize the total number of topics.
        self._number_of_topics = number_of_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        #self._alpha_alpha = numpy.random.random((1, self._number_of_topics)) / self._number_of_topics;
        self._alpha_alpha = numpy.zeros((1, self._number_of_topics)) + alpha;
        #self._eta = eta;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        #self._data = data
        
        # initialize the size of the collection, i.e., total number of documents.
        #self._number_of_documents = len(self._data)
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        #self._number_of_terms = len(self._type_to_index)
        
        self.update_tree_structure(prior_tree);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        #self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha + 1.0 * self._number_of_paths / self._number_of_topics;
        #self._gamma = numpy.tile(self._alpha_alpha + 1.0 * self._number_of_terms / self._number_of_topics, (self._number_of_documents, 1));
        #self._gamma = self._alpha_alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((self._number_of_documents, self._number_of_topics));
        self._gamma = {};
        
        # initialize a _E_log_beta variable, indexed by node, valued by a K-by-C matrix, where C stands for the number of children of that node
        self._E_log_beta = numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_edges));
        for node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[node_index];
            self._E_log_beta[:, edge_index_list] = self.compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);

    def update_tree_structure(self, prior_tree):
        self._maximum_depth = prior_tree._max_depth
        
        self._edge_to_index = {}
        self._index_to_edge = {}

        self._edges_from_internal_node = defaultdict(list);
        
        self._edge_prior = [];

        for parent_node in prior_tree._nodes.keys():
            node = prior_tree._nodes[parent_node]
            
            # if the node is an internal node, compute the prior scalar for every edge
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                for position_index in xrange(len(node._children_offsets)):
                    child_node = node._children_offsets[position_index];
                    self._edge_to_index[(parent_node, child_node)] = len(self._edge_to_index);
                    self._index_to_edge[len(self._index_to_edge)] = (parent_node, child_node);
                    
                    self._edges_from_internal_node[parent_node].append(self._edge_to_index[(parent_node, child_node)]);
                    
                    self._edge_prior.append(node._transition_prior[position_index]);

        self._number_of_edges = len(self._edge_prior);
        
        assert(len(self._edge_to_index) == self._number_of_edges);
        assert(len(self._index_to_edge) == self._number_of_edges);
                    
        self._edge_prior = numpy.array(self._edge_prior)[numpy.newaxis, :];
        assert(self._edge_prior.shape == (1, self._number_of_edges));
        
        self._edges_along_path = defaultdict(list);
        self._paths_through_edge = defaultdict(list);

        # word list indexed by path node_index 
        self._path_index_to_word_index = [];
        self._word_index_to_path_indices = defaultdict(list);
        
        # compute the prior for every path, initialize self._path_prior_sum_of_word, self._path_prior_of_word,        
        path_index = 0;
        # iterate over all leaf nodes in the prior_tree
        for word in prior_tree._word_paths.keys():
            
            # iterate over all paths in the path of the current word (leaf node)
            for path in xrange(len(prior_tree._word_paths[word])):
                self._path_index_to_word_index.append(word)
                self._word_index_to_path_indices[word].append(path_index);

                nodes_along_word_path = prior_tree._word_paths[word][path]._nodes
                
                for position_index in xrange(len(nodes_along_word_path) - 1):
                    parent_node = nodes_along_word_path[position_index];
                    child_node = nodes_along_word_path[position_index + 1];

                    self._edges_along_path[path_index].append(self._edge_to_index[(parent_node, child_node)]);
                    self._paths_through_edge[self._edge_to_index[(parent_node, child_node)]].append(path_index);
                    
                path_index += 1;
                    
        self._number_of_paths = len(self._path_index_to_word_index);
    
    def update_alpha(self, alpha_sufficient_statistics, number_of_documents):
        assert(alpha_sufficient_statistics.shape == (1, self._number_of_topics));        
        alpha_update = self._alpha_alpha;
        
        decay = 0;
        for alpha_iteration in xrange(self._alpha_maximum_iteration):
            alpha_sum = numpy.sum(self._alpha_alpha);
            alpha_gradient = number_of_documents * (scipy.special.psi(alpha_sum) - scipy.special.psi(self._alpha_alpha)) + alpha_sufficient_statistics;
            alpha_hessian = -number_of_documents * scipy.special.polygamma(1, self._alpha_alpha);

            if numpy.any(numpy.isinf(alpha_gradient)) or numpy.any(numpy.isnan(alpha_gradient)):
                print "illegal alpha gradient vector", alpha_gradient

            sum_g_h = numpy.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = number_of_documents * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the alpha vector
            while True:
                singular_hessian = False

                step_size = numpy.power(self._alpha_update_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                #print "step size is", step_size
                assert(self._alpha_alpha.shape == step_size.shape);
                
                if numpy.any(self._alpha_alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self._alpha_alpha - step_size;
                
                if singular_hessian:
                    decay += 1;
                    if decay > self._alpha_maximum_decay:
                        break;
                else:
                    break;
                
            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - self._alpha_alpha));
            self._alpha_alpha = alpha_update;
            if mean_change <= self._alpha_converge_threshold:
                break;
            
        return
    
    def dump_tree(self, output_tree_file):
        output_stream = codecs.open(output_tree_file, 'w', 'utf-8');

        pickle.dump(self._maximum_depth, output_stream);

        pickle.dump(self._edge_to_index, output_stream);
        pickle.dump(self._index_to_edge, output_stream);

        #pickle.dump(self._number_of_edges, output_stream);

        pickle.dump(self._edges_from_internal_node, output_stream);

        pickle.dump(self._edge_prior, output_stream);
        
        output_stream.flush();

        pickle.dump(self._path_index_to_word_index, output_stream);
        pickle.dump(self._word_index_to_path_indices, output_stream);

        #pickle.dump(self._number_of_paths, output_stream);

        pickle.dump(self._edges_along_path, output_stream);
        pickle.dump(self._paths_through_edge, output_stream);
        
        output_stream.flush();
        output_stream.close();

    def dump_parameters(self, output_parameter_file):
        # this pops up error
        output_stream = codecs.open(output_parameter_file, 'w', 'utf-8');
        
        pickle.dump(self._counter, output_stream);
        pickle.dump(self._type_to_index, output_stream);
        pickle.dump(self._index_to_type, output_stream);

        #print >> sys.stderr, "dump", self._index_to_type[0], self._index_to_type[1], self._index_to_type[2], self._index_to_type[3]

        pickle.dump(self._number_of_topics, output_stream);
        pickle.dump(self._number_of_edges, output_stream);
        pickle.dump(self._number_of_paths, output_stream);
        
        pickle.dump(self._update_hyper_parameter, output_stream);
        pickle.dump(self._alpha_alpha, output_stream);
        pickle.dump(self._hybrid_mode, output_stream);
        
        output_stream.flush();
        output_stream.close();
        
    def load_params(self, input_parameter_file):
        input_stream = codecs.open(input_parameter_file, 'r');

        self._counter = pickle.load(input_stream);
        self._type_to_index = pickle.load(input_stream);
        self._index_to_type = pickle.load(input_stream);
        
        self._number_of_topics = pickle.load(input_stream);
        self._number_of_edges = pickle.load(input_stream);
        self._number_of_paths = pickle.load(input_stream);
        
        self._update_hyper_parameter = pickle.load(input_stream);
        self._alpha_alpha = pickle.load(input_stream);
        self._hybrid_mode = pickle.load(input_stream);
        
        input_stream.close();
        
        print >> sys.stderr, "successfully load params from %s..." % input_parameter_file
        
    def load_tree(self, input_tree_file):
        input_stream = codecs.open(input_tree_file, 'r');

        self._maximum_depth = pickle.load(input_stream);

        self._edge_to_index = pickle.load(input_stream);
        self._index_to_edge = pickle.load(input_stream);
        
        #self._number_of_edges = pickle.load(input_stream);

        self._edges_from_internal_node = pickle.load(input_stream);

        self._edge_prior = pickle.load(input_stream);
        
        self._path_index_to_word_index = pickle.load(input_stream);
        self._word_index_to_path_indices = pickle.load(input_stream);
        
        #self._number_of_paths = pickle.load(input_stream);

        self._edges_along_path = pickle.load(input_stream);
        self._paths_through_edge = pickle.load(input_stream);
        
        input_stream.close();
        
        print >> sys.stderr, "successfully load tree from %s..." % input_tree_file

    def dump_E_log_beta(self, output_beta_file):
        numpy.savetxt(output_beta_file, self._E_log_beta);
        
    def load_E_log_beta(self, input_beta_file):
        self._E_log_beta = numpy.loadtxt(input_beta_file);
        print >> sys.stderr, "successfully load E-log-beta from %s..." % input_beta_file
        
    def dump_gamma(self, output_gamma_file):
        output_stream = codecs.open(output_gamma_file, 'w');
        pickle.dump(self._gamma, output_stream);
        output_stream.flush();
        output_stream.close();
        
        #numpy.savetxt(output_gamma_file, self._gamma);
        
    def load_gamma(self, input_gamma_file):
        input_stream = codecs.open(input_gamma_file, 'r');
        self._gamma = pickle.load(input_stream);
        input_stream.close();
        
        print >> sys.stderr, "successfully load gamma from %s..." % input_gamma_file

        #self._gamma = numpy.loadtxt(input_gamma_file);

    def export_gamma(self, output_gamma_file):
        output_stream = codecs.open(output_gamma_file, 'w', 'utf-8');
        for index in self._gamma:
            output_stream.write("%d %s\n" % (index, " ".join(["%f" % value for value in self._gamma[index][0, :]])));

    def export_E_log_beta(self, exp_beta_path):
        #output = codecs.open(exp_beta_path, 'w', 'utf-8');
        output = codecs.open(exp_beta_path, 'w');
        
        #term_path_ranking = defaultdict(nltk.probability.FreqDist);
        freqdist = nltk.probability.FreqDist()
        for k in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (k));

            freqdist.clear();
            for path_index in self._path_index_to_word_index:
                path_rank = 1;
                for edge_index in self._edges_along_path[path_index]:
                    path_rank *= numpy.exp(self._E_log_beta[k, edge_index]);
                    
                freqdist.inc(path_index, path_rank);
            
            for key in freqdist.keys():
                output.write("%s\t%g\n" % (self._index_to_type[self._path_index_to_word_index[key]], freqdist[key]));
                
        output.close();

    def format_output(self, input_directory):
        self._E_log_beta = numpy.zeros((self._number_of_topics, self._number_of_edges));

        self._gamma = {};
        
        if self._update_hyper_parameter:
            alpha_sufficient_statistics = numpy.zeros((1, self._number_of_topics));

        for corpus_name in os.listdir(input_directory):
            if os.path.isdir(os.path.join(input_directory, corpus_name)):
                continue;

            if not corpus_name.startswith('part'):
                continue;

            file_input_stream = codecs.open(os.path.join(input_directory, corpus_name), 'r', 'utf-8');
            for line in file_input_stream:
                line = line.strip();
                contents = line.split("\t");
                assert len(contents) == 2;

                indices = contents[0].split();
                if len(indices) == 2:
                    index_1 = int(indices[0]);
                    index_2 = int(indices[1]);
                    value = float(contents[1]);
                    
                    if index_1 < 0 and index_2 < 0:
                        # if this value is a gamma term
                        index_1 = -index_1-1;
                        index_2 = -index_2-1;
                        if index_1 not in self._gamma:
                            self._gamma[index_1] = numpy.zeros((1, self._number_of_topics))
                        self._gamma[index_1][0, index_2] = value;
                    elif index_1 == 0 and index_2 == 0:
                        print >> sys.stderr, "output likelihood..."
                    elif index_1 == 0 and index_2 > 0:
                        # if this value is a alpha term
                        alpha_sufficient_statistics[0, index_2-1] = value;
                    elif index_1 == 0 and index_2 < 0:
                        # if this value is a system setting
                        number_of_documents = value;
                    else:
                        print >> sys.stderr, "unexpected key..."
                        
                elif len(indices) == 1:
                    values = contents[1].split();
                    
                    topic_index = int(indices[0]);
                    assert len(values) == self._number_of_edges;
                        
                    for edge_index in xrange(self._number_of_edges):
                        self._E_log_beta[topic_index, edge_index] = float(values[edge_index]);

            print "successfully processed file %s..." % corpus_name
            
        if self._update_hyper_parameter:
            assert number_of_documents>0;
            self.update_alpha(alpha_sufficient_statistics, number_of_documents);
            print "successfully updated hyperparameter alpha to", self._alpha_alpha

class Mapper(Inferencer):
    def __init__(self,
                 update_hyper_parameter=False,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001
                 
                 ):
        Inferencer.__init__(self, update_hyper_parameter, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration, model_likelihood_threshold);
        
        self.load_params('current-params');
        self.load_tree('current-tree');
        self.load_E_log_beta('current-E-log-beta');
        
        if self._hybrid_mode:
            number_of_samples = 10;
            burn_in_samples = 5;
            self._number_of_samples = number_of_samples;
            self._burn_in_samples = burn_in_samples;
        else:
            gamma_maximum_iteration = 10;
            gamma_converge_threshold = 1e-6;
            self._gamma_maximum_iteration = gamma_maximum_iteration;
            self._gamma_converge_threshold = gamma_converge_threshold;

    def __call__(self, data):
        '''
        for key, value in data:
            #doc_id, content = self.parse_document(value);
            #self.optimize(doc_id, content);
            
            self.optimize(self.parse_document(value));
        self.finalize();
        '''
        
        if self._update_hyper_parameter:
            alpha_sufficient_statistics = numpy.zeros((1, self._number_of_topics));
            number_of_documents = 0;

        gamma = {};
        phi_sufficient_statistics = nltk.probability.FreqDist();
                    
        if self._hybrid_mode:
            document_level_log_likelihood = 0;

            for key, value in data:
                document = value.strip().lower();
                contents = document.split("\t");
                assert len(contents) == 2, str(len(contents)) + "\t" + document;
                
                if self._update_hyper_parameter:
                    number_of_documents += 1;
        
                content = [];
    
                '''
                # this causes error when dealing with aligned corpus
                for token in contents[-1].split():
                    #print >> sys.stderr, "token:", token
                    if token not in self._type_to_index:
                        continue;
                    content.append(self._type_to_index[token]);
                '''
    
                for temp_content in contents[1:]:
                    for token in temp_content.split():
                        if token not in self._type_to_index:
                            continue;
                        content.append(self._type_to_index[token]);
    
                if len(content) == 0:
                    print >> sys.stderr, "document %d collapsed..." % int(contents[0]);
                    continue;
    
                doc_id = int(contents[0]);
            
                #phi_sufficient_statistics = nltk.probability.FreqDist();
                #phi_sufficient_statistics = {};
                
                topic_path_assignment = {};
                topic_sum = numpy.zeros((1, self._number_of_topics));
                for word_pos in xrange(len(content)):
                    topic_assignment = numpy.random.randint(0, self._number_of_topics);
                    path_assignment = numpy.random.randint(0, len(self._word_index_to_path_indices[content[word_pos]]));
                    topic_path_assignment[word_pos] = (topic_assignment, path_assignment);
                    topic_sum[0, topic_assignment] += 1;
                del word_pos, topic_assignment, path_assignment;
                    
                # update path_phi and phi_sum until phi_sum converges
                for iteration_index in xrange(self._number_of_samples):
                    #document_phi = numpy.zeros((self._number_of_topics, self._number_of_paths));
                    
                    phi_entropy = 0;
                    phi_E_log_beta = 0;
                    
                    for word_pos in xrange(len(content)):
                        word_id = content[word_pos];
                        topic_sum[0, topic_path_assignment[word_pos][0]] -= 1;
                        
                        paths_lead_to_current_word = self._word_index_to_path_indices[word_id];
                        assert len(paths_lead_to_current_word) > 0
        
                        #path_phi = numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]).T, (1, len(paths_lead_to_current_word)));
                        path_phi = numpy.tile((topic_sum + self._alpha_alpha).T, (1, len(paths_lead_to_current_word)));
                        assert path_phi.shape == (self._number_of_topics, len(paths_lead_to_current_word));
                        
                        for path_index in xrange(len(paths_lead_to_current_word)):
                            path_phi[:, path_index] *= numpy.exp(numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1));
                        del path_index
                        
                        assert path_phi.shape == (self._number_of_topics, len(paths_lead_to_current_word));
                        # normalize path_phi over all topics
                        path_phi /= numpy.sum(path_phi);
                        
                        random_number = numpy.random.random();
                        for topic_index in xrange(self._number_of_topics):
                            for path_index in xrange(len(paths_lead_to_current_word)):
                                random_number -= path_phi[topic_index, path_index];
                                if random_number <= 0:
                                    break;
                            if random_number <= 0:
                                break;
                        topic_sum[0, topic_index] += 1;
                        topic_path_assignment[word_pos] = (topic_index, path_index);
                        
                        if iteration_index >= self._burn_in_samples: 
                            phi_sufficient_statistics.inc((topic_index, paths_lead_to_current_word[path_index]), 1)
                            
                            '''
                            if (topic_index, paths_lead_to_current_word[path_index]) not in phi_sufficient_statistics:
                                phi_sufficient_statistics[(topic_index, paths_lead_to_current_word[path_index])] = 1;
                            else:
                                phi_sufficient_statistics[(topic_index, paths_lead_to_current_word[path_index])] += 1;
                            '''
        
                        '''
                        #phi_entropy += - numpy.sum(path_phi * numpy.log(path_phi));
                        phi_entropy += -numpy.sum(path_phi * numpy.log(path_phi + 1e-50));
                        for path_index in xrange(len(paths_lead_to_current_word)):
                            phi_E_log_beta += numpy.sum(path_phi[:, [path_index]] * numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1)[:, numpy.newaxis])
                        del path_index
                        '''
        
                    del word_pos, paths_lead_to_current_word  
                    
                gamma[doc_id] = self._alpha_alpha + topic_sum;
                
                if self._update_hyper_parameter:
                    alpha_sufficient_statistics += self.compute_dirichlet_expectation(gamma[doc_id]);
                
                #gamma[doc_id] = self._alpha_alpha + topic_sum;
                #for topic_index in xrange(self._number_of_topics):
                    #yield (-doc_id-1, -topic_index-1), self._alpha_alpha[0, topic_index] + topic_sum[0, topic_index];
                #yield (-doc_id-1, -topic_index-1), " ".join(["%f" % item for item in (self._alpha_alpha + topic_sum)[0, :]]);
                
                '''
                document_level_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha_alpha)) - numpy.sum(scipy.special.gammaln(self._alpha_alpha));
                document_level_log_likelihood += numpy.sum((self._alpha_alpha - 1) * self.compute_dirichlet_expectation(gamma[[doc_id], :]));
        
                document_level_log_likelihood += numpy.sum(topic_sum / len(content) * self.compute_dirichlet_expectation(gamma[[doc_id], :]));
                
                document_level_log_likelihood += phi_E_log_beta;
                
                document_level_log_likelihood += -scipy.special.gammaln(numpy.sum(gamma[[doc_id], :])) + numpy.sum(scipy.special.gammaln(gamma[[doc_id], :]))
                document_level_log_likelihood += -numpy.sum((gamma[[doc_id], :] - 1) * self.compute_dirichlet_expectation(gamma[[doc_id], :]));
                
                document_level_log_likelihood += phi_entropy;
                '''
        
                '''    
                for (topic_index, path_index) in phi_sufficient_statistics:
                    yield (topic_index+1, path_index+1), phi_sufficient_statistics[(topic_index, path_index)]/(self._number_of_samples - self._burn_in_samples);
                '''
    
            for doc_id in gamma:
                for topic_index in xrange(self._number_of_topics):
                    yield (-doc_id - 1, -topic_index - 1), float(gamma[doc_id][0, topic_index]);
            
            for (topic_index, path_index) in phi_sufficient_statistics:
                yield (topic_index + 1, path_index + 1), float(phi_sufficient_statistics[(topic_index, path_index)] / (self._number_of_samples - self._burn_in_samples));
    
            if self._update_hyper_parameter:
                for topic_id in xrange(self._number_of_topics):
                    yield (0, topic_id + 1), float(alpha_sufficient_statistics[0, topic_id]);
                
                yield (0, -1), number_of_documents;
        else:
            for key, value in data:
                document = value.strip().lower();
                contents = document.split("\t");
                assert len(contents)==2, str(len(contents)) + "\t" + document;

                if self._update_hyper_parameter:
                    number_of_documents += 1;
                
                content = dict();
                for temp_content in contents[1:]:
                    for token in contents[-1].split():
                        if token not in self._type_to_index:
                            continue;
                        token_index = self._type_to_index[token]
                        if token_index not in content:
                            content[token_index] = 0;
                        content[token_index] += 1;
                if len(content) == 0:
                    print >> sys.stderr, "document %d collapsed..." % int(contents[0]);
                    continue;
                
                doc_id = int(contents[0]);
                
                '''
                topic_path_assignment = {};
                topic_sum = numpy.zeros((1, self._number_of_topics));
                for word_pos in xrange(len(content)):
                    topic_assignment = numpy.random.randint(0, self._number_of_topics);
                    path_assignment = numpy.random.randint(0, len(self._word_index_to_path_indices[content[word_pos]]));
                    topic_path_assignment[word_pos] = (topic_assignment, path_assignment);
                    topic_sum[0, topic_assignment] += 1;
                del word_pos, topic_assignment, path_assignment;
                '''
                
                topic_sum = numpy.random.random((1, self._number_of_topics));
                
                # update path_phi and phi_sum until phi_sum converges
                for iteration_index in xrange(self._gamma_maximum_iteration):
                    E_log_gamma = self.compute_dirichlet_expectation(topic_sum + self._alpha_alpha);
                    topic_sum = numpy.zeros((1, self._number_of_topics));
                    
                    for word_id in content:
                        paths_lead_to_current_word = self._word_index_to_path_indices[word_id];
                        assert len(paths_lead_to_current_word)>0
                        
                        #path_phi = numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]).T, (1, len(paths_lead_to_current_word)));
                        path_phi = numpy.tile(E_log_gamma.T, (1, len(paths_lead_to_current_word)));
                        #assert path_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));
                        for path_index in xrange(len(paths_lead_to_current_word)):
                            path_phi[:, path_index] += numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1);
                        del path_index
                        assert path_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));
                        
                        # normalize path_phi over all topics
                        log_normalizer = scipy.misc.logsumexp(path_phi);
                        #print >> sys.stderr, "log normalizer=%f" % log_normalizer
                        #if numpy.isnan(log_normalizer):
                            #print >> sys.stderr, "\n%s\n%s" % (path_phi, E_log_gamma)
                        path_phi -= log_normalizer
                        path_phi = numpy.exp(path_phi);
                        # this is a trick to avoid overflow error
                        #path_phi *= path_phi>0;
                        #path_phi /= numpy.sum(path_phi);
                        assert numpy.all(path_phi>=0);
                        topic_sum += numpy.sum(path_phi, axis=1)[:, numpy.newaxis].T * content[word_id];
                        
                        if iteration_index==self._gamma_maximum_iteration-1:
                            for topic_index in xrange(self._number_of_topics):
                                for path_index in xrange(len(paths_lead_to_current_word)):
                                    phi_sufficient_statistics.inc((topic_index, paths_lead_to_current_word[path_index]), path_phi[topic_index, path_index]);
                        
                    del word_id, paths_lead_to_current_word
                gamma[doc_id] = self._alpha_alpha + topic_sum
                
                if self._update_hyper_parameter:
                    alpha_sufficient_statistics += self.compute_dirichlet_expectation(gamma[doc_id]);
                
                #for topic_index in xrange(self._number_of_topics):
                    #yield (-doc_id-1, -topic_index-1), self._alpha_alpha[0, topic_index] + topic_sum[0, topic_index];
                    
                #yield (-doc_id-1, -topic_index-1), " ".join(["%f" % item for item in (self._alpha_alpha + topic_sum)[0, :]]);
                
                '''
                document_level_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha_alpha)) - numpy.sum(scipy.special.gammaln(self._alpha_alpha));
                document_level_log_likelihood += numpy.sum((self._alpha_alpha - 1) * self.compute_dirichlet_expectation(self._gamma[[doc_id], :]));
        
                document_level_log_likelihood += numpy.sum(numpy.sum(document_phi, axis=1)[:, numpy.newaxis].T * self.compute_dirichlet_expectation(self._gamma[[doc_id], :]));
                
                document_level_log_likelihood += phi_E_log_beta;
                
                document_level_log_likelihood += - scipy.special.gammaln(numpy.sum(self._gamma[[doc_id], :])) + numpy.sum(scipy.special.gammaln(self._gamma[[doc_id], :]))
                document_level_log_likelihood += - numpy.sum((self._gamma[[doc_id], :] - 1) * self.compute_dirichlet_expectation(self._gamma[[doc_id], :]));
        
                document_level_log_likelihood += phi_entropy;
                '''
        
                '''    
                for (topic_index, path_index) in phi_sufficient_statistics:
                    yield (topic_index+1, path_index+1), phi_sufficient_statistics[(topic_index, path_index)]/(self._number_of_samples - self._burn_in_samples);
                '''
    
            for doc_id in gamma:
                for topic_index in xrange(self._number_of_topics):
                    yield (-doc_id-1, -topic_index-1), float(gamma[doc_id][0, topic_index]);
            
            for (topic_index, path_index) in phi_sufficient_statistics:
                yield (topic_index+1, path_index+1), float(phi_sufficient_statistics[(topic_index, path_index)]);

            if self._update_hyper_parameter:
                for topic_id in xrange(self._number_of_topics):
                    yield (0, topic_id + 1), float(alpha_sufficient_statistics[0, topic_id]);
                
                yield (0, -1), number_of_documents;

class Combiner(Inferencer):
    def __call__(self, key, values):
        yield key, sum(values);

        #value_sum = 0;
        #for value in values:
            #value_sum += float(value);
        #yield key, value_sum;
        
class Reducer(Inferencer):
    def __init__(self,
                 update_hyper_parameter=False,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001
                 ):
        Inferencer.__init__(self, update_hyper_parameter, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration, model_likelihood_threshold);
        
        self.load_params('current-params');
        self.load_tree('current-tree');
        #self.load_E_log_beta('current-E-log-beta');
        
        #self._phi_sufficient_statistics = numpy.zeros((1, self._number_of_paths));
        #self._current_topic_index = -1;
        
    def __call__(self, data):
        phi_sufficient_statistics = None;
        current_topic_index = -1;
        
        for key, values in data:
            #value_sum = 0;
            #for value in values:
                #value_sum += float(value);

            value_sum = sum(values);
                
            #(index_1, index_2) = key;
            index_1 = int(key[0]);
            index_2 = int(key[1]);
            
            if index_1 < 0 and index_2 < 0:
                # if this value is a gamma term
                yield "%d %d" % (index_1, index_2), value_sum;
            elif index_1 > 0 and index_2 > 0:
                # if this value is a phi term
                topic_index = index_1 - 1;
                path_index = index_2 - 1;
                
                if current_topic_index != topic_index:
                    if current_topic_index == -1:
                        phi_sufficient_statistics = numpy.zeros((1, self._number_of_paths));
                        
                    else:
                        var_beta = numpy.copy(self._edge_prior);
                        assert var_beta.shape == (1, self._number_of_edges);
                        assert numpy.min(var_beta) >= 0;

                        for edge_index in self._index_to_edge:
                            var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
                        del edge_index
                        assert(var_beta.shape == (1, self._number_of_edges));
                        assert numpy.min(var_beta) >= 0;
                
                        E_log_beta = numpy.copy(var_beta);
                        for internal_node_index in self._edges_from_internal_node:
                            edge_index_list = self._edges_from_internal_node[internal_node_index];
                            assert numpy.min(E_log_beta[:, edge_index_list]) >= 0;
                            E_log_beta[:, edge_index_list] = self.compute_dirichlet_expectation(E_log_beta[:, edge_index_list]);
                        del internal_node_index, edge_index_list;
                        assert numpy.min(var_beta) >= 0;
                        
                        assert E_log_beta.shape == (1, self._number_of_edges);
                        
                        yield current_topic_index, " ".join(["%f" % (item) for item in E_log_beta[0, :]]);
                        
                        #yield self.current_topic_index, " ".join(["%f" %(item) for item in self.optimize_lambda_vector()[0, :]]);
                        
                        '''
                        corpus_level_log_likelihood = 0;
                        for internal_node_index in self._edges_from_internal_node:
                            edges_indices_list = self._edges_from_internal_node[internal_node_index];
                            
                            # term 3
                            tmp = corpus_level_log_likelihood
                            corpus_level_log_likelihood += (scipy.special.gammaln(numpy.sum(self._edge_prior[:, edges_indices_list])) - numpy.sum(scipy.special.gammaln(self._edge_prior[:, edges_indices_list]))) * self._number_of_topics;
                            corpus_level_log_likelihood += numpy.sum(numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T));
                            
                            #print numpy.sum(var_beta[:, edges_indices_list].T)
                            #print numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T)
                            
                            # term 6
                            #corpus_level_log_likelihood += numpy.sum( - scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)))
                            #corpus_level_log_likelihood += numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]));
                            corpus_level_log_likelihood += numpy.sum( - scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)) + numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]), axis=1));
                            corpus_level_log_likelihood += numpy.sum( - (var_beta[:, edges_indices_list]-1) * self.compute_dirichlet_expectation(var_beta[:, edges_indices_list]));
                                    
                        assert numpy.min(var_beta)>=0;
                
                        # TODO: add in alpha updating
                        # compute the sufficient statistics for alpha and update
                        #alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
                        #alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
                        #self.update_alpha(alpha_sufficient_statistics)
                        '''
                        
                    current_topic_index = topic_index;
                    
                phi_sufficient_statistics[0, path_index] = value_sum; #sum(values);
                
            elif index_1 == 0 and index_2 == 0:
                print >> sys.stderr, "output likelihood..."
            elif index_1 == 0 and index_2 > 0:
                # if this value is a alpha term
                yield "%d %d" % (index_1, index_2), value_sum;
            elif index_1 == 0 and index_2 < 0:
                # if this value is a system setting
                yield "%d %d" % (index_1, index_2), value_sum;
            else:
                print >> sys.stderr, "unexpected key sequence...";
        
        if phi_sufficient_statistics == None:
            return;
        
        var_beta = self._edge_prior
        #print "=====>", var_beta
        assert var_beta.shape == (1, self._number_of_edges);
        assert numpy.min(var_beta) >= 0;
        #for internal_node_index in self._edges_from_internal_node:
            #edges_indices_list = self._edges_from_internal_node[internal_node_index];
            #var_beta[:, edges_indices_list] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_internal_node[internal_node_index]], axis=1)[:, numpy.newaxis];
        
        for edge_index in self._index_to_edge:
            #print var_beta[:, edge_index].shape, numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis].shape;
            var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
        del edge_index
        assert(var_beta.shape == (1, self._number_of_edges));
        assert numpy.min(var_beta) >= 0;

        E_log_beta = numpy.copy(var_beta);
        for internal_node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[internal_node_index];
            assert numpy.min(E_log_beta[:, edge_index_list]) >= 0;
            E_log_beta[:, edge_index_list] = self.compute_dirichlet_expectation(E_log_beta[:, edge_index_list]);
        del internal_node_index, edge_index_list;
        assert numpy.min(var_beta) >= 0;

        '''
        corpus_level_log_likelihood = 0;
        for internal_node_index in self._edges_from_internal_node:
            edges_indices_list = self._edges_from_internal_node[internal_node_index];
            
            # term 3
            tmp = corpus_level_log_likelihood
            corpus_level_log_likelihood += (scipy.special.gammaln(numpy.sum(self._edge_prior[:, edges_indices_list])) - numpy.sum(scipy.special.gammaln(self._edge_prior[:, edges_indices_list]))) * self._number_of_topics;
            corpus_level_log_likelihood += numpy.sum(numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T));
            
            #print numpy.sum(var_beta[:, edges_indices_list].T)
            #print numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T)
            
            # term 6
            #corpus_level_log_likelihood += numpy.sum( - scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)))
            #corpus_level_log_likelihood += numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]));
            corpus_level_log_likelihood += numpy.sum( - scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)) + numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]), axis=1));
            corpus_level_log_likelihood += numpy.sum( - (var_beta[:, edges_indices_list]-1) * self.compute_dirichlet_expectation(var_beta[:, edges_indices_list]));
                    
        assert numpy.min(var_beta)>=0;

        # TODO: add in alpha updating
        # compute the sufficient statistics for alpha and update
        #alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        #alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
        #self.update_alpha(alpha_sufficient_statistics)
        
        #print numpy.sum(numpy.exp(self.E_log_beta), axis=1);
        
        return corpus_level_log_likelihood
        '''

        yield current_topic_index, " ".join(["%f" % (item) for item in E_log_beta[0, :]]);

if __name__ == '__main__':
    import dumbo;
    dumbo.run(Mapper, Reducer, combiner=Combiner);
