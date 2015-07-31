PyLDA

-

PyLDA

	cd $WORKSPACE/PyLDA
	mkdir output
	
	cd $WORKSPACE/PyLDA/input
	tar zxvf ap.tar.gz
	
	cd $WORKSPACE/PyLDA/src/
	python -m lda.launch_train --input_directory=../input/ap --output_directory=../output/ --number_of_topics=10 --training_iterations=100 --inference_mode=0
