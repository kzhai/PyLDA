import os
import string

job_template_path = "../src/vb/prior/tree/dumbo/launch.sh";
input_directory = "../input/"
hadoop_directory="../hadoop/"

training_iterations=50;
number_of_mappers=50;
snapshot_interval=10;
update_hyper_parameter="true";
inference_mode="false";

'''
number_of_topics_list = [10, 20]
dataset_name_list = ["nist"];
dataset_format_list = ["sent", "sent-comb"];
dataset_language_list = ["zh", "zh-en"];
tree_name_list = ["dict-rsw", "empty", "train-aligned"]
test_file_list = ["mt02", "mt03", "mt05", "mt06", "mt08"]
'''
number_of_topics_list = [10, 20]
dataset_name_list = ["wiki"];
dataset_format_list = ["sent", "sent-comb", "doc-comb"];
dataset_language_list = ["zh", "zh-en"];
tree_name_list = ["dict-rsw", "empty", "train-aligned"]
test_file_list = ["mt02", "mt03", "mt05", "mt06", "mt08"]

for dataset_name in dataset_name_list:
    for dataset_format in dataset_format_list:
        for dataset_language in dataset_language_list:
            
            corpus_name=".".join([dataset_name, dataset_format, dataset_language]);
            
            if not os.path.exists(os.path.join(input_directory, corpus_name)):
                continue;
            
            for tree_name in tree_name_list:
                if not os.path.exists(os.path.join(input_directory, corpus_name, tree_name+".hyperparams")):
                    continue;
                if not os.path.exists(os.path.join(input_directory, corpus_name, tree_name+".wn.0")):
                    continue;
                if not os.path.exists(os.path.join(input_directory, corpus_name, tree_name+".wn.1")):
                    continue;
                
                for number_of_topics in number_of_topics_list:
                    number_of_reducers=number_of_topics;
                    
                    model_suffix = "K%d.I%d.M%d.R%d.%s.%s" % (number_of_topics, training_iterations, number_of_mappers, number_of_reducers, update_hyper_parameter, inference_mode)
                    print "generating script for %s.%s.%s" % (corpus_name, tree_name, model_suffix)
                    
                    input_stream = open(job_template_path, 'r');
                    output_stream = open(os.path.join(hadoop_directory, ".".join([corpus_name, tree_name, model_suffix, "sh"])), 'w');
        
                    for line in input_stream:
                        line = line.rstrip();
                    
                        if line.startswith("SET_PARAMETER"):
        
                            output_stream.write("DatasetName=%s\n" % dataset_name);
                            output_stream.write("DatasetFormat=%s\n" % dataset_format);
                            output_stream.write("DatasetLanguage=%s\n" % dataset_language);
                            output_stream.write("TreeName=%s\n" % tree_name);
        
                            output_stream.write("NumTopic=%d\n" % number_of_topics);
                            output_stream.write("Iterations=%d\n" % training_iterations);
                            output_stream.write("NumMapper=%d\n" % number_of_mappers);
                            output_stream.write("NumReducer=%d\n" % number_of_reducers);
                            output_stream.write("SnapshotInterval=%d\n" % snapshot_interval);
                            output_stream.write("UpdateHyperParameter=\"%s\"\n" % update_hyper_parameter);
                            output_stream.write("HybridMode=\"%s\"\n" % inference_mode);
                            #output_stream.write("Suffix=%s\n" %(parameterStrings));
                            output_stream.write("\n");
                            
                            continue;
                    
                        if line.startswith("SET_POST_PIPELINE"):
                            for test_file in test_file_list:
                                output_stream.write("$PYTHON_COMMAND -m experiments.test \\\n");
                                #output_stream.write("\t$ProjectDirectory/data/$CorpusName/doc.dat \\\n");
                                output_stream.write("\t$MTDevTestDirectory/%s.src.txt \\\n" % (test_file));
                                output_stream.write("\t$MTOutputSubDirectory/%s-test.topics \\\n" % (test_file));
                                output_stream.write("\t$LocalOutputDirectory/ \\\n");
                                output_stream.write("\t$LocalInputDirectory/voc.dat\n");
                                output_stream.write("\n");

                            output_stream.write("$PYTHON_COMMAND -m experiments.test \\\n");
                            #output_stream.write("\t$ProjectDirectory/data/$CorpusName/doc.dat \\\n");
                            output_stream.write("\t$ProjectDirectory/input/nist.sent.zh/doc.dat \\\n");
                            output_stream.write("\t$MTOutputSubDirectory/nist.sent.zh.topics \\\n");
                            output_stream.write("\t$LocalOutputDirectory/ \\\n");
                            output_stream.write("\t$LocalInputDirectory/voc.dat\n");
                            output_stream.write("\n");
                                
                            continue;
                    
                        output_stream.write(line + "\n");
