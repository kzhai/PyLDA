class HybridMapper(Inferencer):
    def __init__(self,
                 update_hyper_parameter=False,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001,
                 number_of_samples=10,
                 burn_in_samples=5
                 ):
        Inferencer.__init__(self, update_hyper_parameter, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration, model_likelihood_threshold);
        
        self._number_of_samples = number_of_samples;
        self._burn_in_samples = burn_in_samples;
        
        self.load_params('current-params');
        self.load_tree('current-tree');
        self.load_E_log_beta('current-E-log-beta');

        #self._gamma = {}
        #self._phi_sufficient_statistics = nltk.probability.FreqDist();
        #if self._hybrid_mode:
            #self.load_gamma(self.params['current-gamma']);
            
    def __call__(self, data):
        '''
        for key, value in data:
            #doc_id, content = self.parse_document(value);
            #self.optimize(doc_id, content);
            
            self.optimize(self.parse_document(value));
        self.finalize();
        '''
        
        gamma = {};
        phi_sufficient_statistics = nltk.probability.FreqDist();
        document_level_log_likelihood = 0;
        
        if self._update_hyper_parameter:
            alpha_sufficient_statistics = numpy.zeros((1, self._number_of_topics));
            number_of_documents = 0;
                    
        for key, value in data:
            document = value.strip().lower();
            #print >> sys.stderr, "document is", document
            contents = document.split("\t");
            # this causes error when dealing with aligned corpus
            assert len(contents) == 2, len(contents);
            #print >> sys.stderr, contents
            
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
                    #print >> sys.stderr, "token:", token
                    if token not in self._type_to_index:
                        continue;
                    
                    content.append(self._type_to_index[token]);

            #assert len(content)>0, "document %d collapsed..." % int(contents[0]);
            if len(content) == 0:
                print >> sys.stderr, "document %d collapsed..." % int(contents[0]);
                continue;

            doc_id = int(contents[0]);
        
            #phi_sufficient_statistics = nltk.probability.FreqDist();
            #phi_sufficient_statistics = {};
            
            topic_path_assignment = {};
            topic_sum = numpy.zeros((1, self._number_of_topics));
            for word_index in xrange(len(content)):
                topic_assignment = numpy.random.randint(0, self._number_of_topics);
                path_assignment = numpy.random.randint(0, len(self._word_index_to_path_indices[content[word_index]]));
                topic_path_assignment[word_index] = (topic_assignment, path_assignment);
                topic_sum[0, topic_assignment] += 1;
            del word_index, topic_assignment, path_assignment;
                
            # update path_phi and phi_sum until phi_sum converges
            for sample_index in xrange(self._number_of_samples):
                #document_phi = numpy.zeros((self._number_of_topics, self._number_of_paths));
                
                phi_entropy = 0;
                phi_E_log_beta = 0;
                
                for word_index in xrange(len(content)):
                    word_id = content[word_index];
                    topic_sum[0, topic_path_assignment[word_index][0]] -= 1;
                    
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
                    topic_path_assignment[word_index] = (topic_index, path_index);
                    
                    if sample_index >= self._burn_in_samples: 
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
    
                del word_index, paths_lead_to_current_word  
                
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

        #yield (0, 0), document_level_log_likelihood;

        if self._update_hyper_parameter:
            for topic_id in xrange(self._number_of_topics):
                yield (0, topic_id + 1), float(alpha_sufficient_statistics[0, topic_id]);
            
            yield (0, -1), number_of_documents;

class VBMapper(Inferencer):    
    def __init__(self,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 #gamma_converge_threshold=0.000001,
                 #gamma_maximum_iteration=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 #model_likelihood_threshold=0.00001
                 gamma_maximum_iteration,
                 gamma_converge_threshold
                 ):
        Inferencer.__init__(self, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration);
        
        self._gamma_maximum_iteration = gamma_maximum_iteration;
        self._gamma_converge_threshold = gamma_converge_threshold;
        
        self.load_params('current-params');
        self.load_tree('current-tree');
        self.load_E_log_beta('current-E-log-beta');

        #self._gamma = {}
        #self._phi_sufficient_statistics = nltk.probability.FreqDist();
        #if self._hybrid_mode:
            #self.load_gamma(self.params['current-gamma']);
            
    def __call__(self, data):
        '''
        for key, value in data:
            #doc_id, content = self.parse_document(value);
            #self.optimize(doc_id, content);
            
            self.optimize(self.parse_document(value));
        self.finalize();
        '''
        
        gamma = {};
        phi_sufficient_statistics = nltk.probability.FreqDist();
        
        for key, value in data:
            document = value.strip().lower();
            contents = document.split("\t");

            assert len(contents)==2;
    
            content = [];
            for token in contents[-1].split():
                #print >> sys.stderr, "token:", token
                if token not in self._type_to_index:
                    continue;
                
                content.append(self._type_to_index[token]);
                
            assert len(content)>0, "content %d collapsed..." % int([contents[0]]);
        
            doc_id = int(content[0]);
        
            #phi_sufficient_statistics = nltk.probability.FreqDist();
            #phi_sufficient_statistics = {};
            
            topic_path_assignment = {};
            topic_sum = numpy.zeros((1, self._number_of_topics));
            for word_index in xrange(len(content)):
                topic_assignment = numpy.random.randint(0, self._number_of_topics);
                path_assignment = numpy.random.randint(0, len(self._word_index_to_path_indices[content[word_index]]));
                topic_path_assignment[word_index] = (topic_assignment, path_assignment);
                topic_sum[0, topic_assignment] += 1;
            del word_index, topic_assignment, path_assignment;
                
            # update path_phi and phi_sum until phi_sum converges
            for sample_index in xrange(self._number_of_samples):
                #document_phi = numpy.zeros((self._number_of_topics, self._number_of_paths));
                
                phi_entropy = 0;
                phi_E_log_beta = 0;
                
                for word_index in xrange(len(content)):
                    word_id = content[word_index];
                    topic_sum[0, topic_path_assignment[word_index][0]] -= 1;
                    
                    paths_lead_to_current_word = self._word_index_to_path_indices[word_id];
                    assert len(paths_lead_to_current_word)>0
    
                    #path_phi = numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]).T, (1, len(paths_lead_to_current_word)));
                    path_phi = numpy.tile((topic_sum + self._alpha_alpha).T, (1, len(paths_lead_to_current_word)));
                    assert path_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));
                    
                    for path_index in xrange(len(paths_lead_to_current_word)):
                        path_phi[:, path_index] *= numpy.exp(numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1));
                    del path_index
                    
                    assert path_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));
                    # normalize path_phi over all topics
                    path_phi /= numpy.sum(path_phi);
                    
                    random_number = numpy.random.random();
                    for topic_index in xrange(self._number_of_topics):
                        for path_index in xrange(len(paths_lead_to_current_word)):
                            random_number-=path_phi[topic_index, path_index];
                            if random_number<=0:
                                break;
                        if random_number<=0:
                            break;
                    topic_sum[0, topic_index] += 1;
                    topic_path_assignment[word_index] = (topic_index, path_index);
                    
                    if sample_index >= self._burn_in_samples: 
                        phi_sufficient_statistics.inc((topic_index, paths_lead_to_current_word[path_index]), 1)
                        
                        '''
                        if (topic_index, paths_lead_to_current_word[path_index]) not in phi_sufficient_statistics:
                            phi_sufficient_statistics[(topic_index, paths_lead_to_current_word[path_index])] = 1;
                        else:
                            phi_sufficient_statistics[(topic_index, paths_lead_to_current_word[path_index])] += 1;
                        '''
    
                    '''
                    phi_entropy += - numpy.sum(path_phi * numpy.log(path_phi)) * self._data[doc_id][word_id];
                    for path_index in xrange(len(paths_lead_to_current_word)):
                        phi_E_log_beta += self._data[doc_id][word_id] * numpy.sum(path_phi[:, [path_index]] * numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1)[:, numpy.newaxis])
                    del path_index
                    '''
    
                del word_index, paths_lead_to_current_word  
                
            gamma[doc_id] = self._alpha_alpha + topic_sum;
            #gamma[doc_id] = self._alpha_alpha + topic_sum;
    
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
                yield (-doc_id-1, -topic_index-1), "%f" %(gamma[doc_id][0, topic_index]);
        
        for (topic_index, path_index) in phi_sufficient_statistics:
            yield (topic_index+1, path_index+1), phi_sufficient_statistics[(topic_index, path_index)]/(self._number_of_samples - self._burn_in_samples);
