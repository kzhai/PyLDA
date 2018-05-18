PyLDA
==========

PyLDA is a Latent Dirichlet Allocation topic modeling package, developed by the Cloud Computing Research Team in [University of Maryland, College Park](http://www.umd.edu).

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyLDA).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk.

Launch and Execute
----------

Assume the PyLDA package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyLDA

To prepare the example dataset,

	tar zxvf associated-press.tar.gz

To launch PyLDA, first redirect to the directory of PyLDA source code,

	cd $PROJECT_SPACE/src/PyLDA

and run the following command on example dataset,

	python -m launch_train --input_directory=./associated-press --output_directory=./ --number_of_topics=10 --training_iterations=100

The generic argument to run PyLDA is

	python -m launch_train --input_directory=$INPUT_DIRECTORY/$CORPUS_NAME --output_directory=$OUTPUT_DIRECTORY --number_of_topics=$NUMBER_OF_TOPICS --training_iterations=$NUMBER_OF_ITERATIONS

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any circumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help
