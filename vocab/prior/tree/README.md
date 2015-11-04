PyLDA/vocab/prior/tree
==========

PyLDA/vocab/prior/tree is a Latent Dirichlet Allocation topic modeling
package with tree prior on vocabulary, developed by the Cloud
Computing Research Team in [University of Maryland, College Park]
(http://www.umd.edu).

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyLDA/tree/master/vocab/prior/tree).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as
protocol-buffer, numpy, scipy and nltk.

Launch and Execute
----------

Assume the PyLDA/vocab/prior/tree package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyLDA/vocab/prior/tree

To prepare the example dataset,

	tar zxvf tree-synthetic.tar.gz

To launch PyLDA/vocab/prior/tree, first redirect to the directory of PyLDA/vocab/prior/tree source code,

	cd $PROJECT_SPACE/src/PyLDA/vocab/prior/tree

and run the following command on example dataset,

	python -m launch_train --input_directory=./tree-synthetic/
	--output_directory=./ --number_of_topics=5
	--training_iterations=50 --tree_name=tree1

The generic argument to run PyLDA.vocab.prior.tree is

	python -m launch_train
    --input_directory=$INPUT_DIRECTORY/$CORPUS_NAME
    --output_directory=$OUTPUT_DIRECTORY
    --number_of_topics=$NUMBER_OF_TOPICS
    --training_iterations=$NUMBER_OF_ITERATIONS
	--tree_name=$TREE_NAME

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help
