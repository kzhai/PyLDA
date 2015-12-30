#!/bin/bash

#if [ $# -ne 7 ]
#then
#    echo "USAGE: launch.sh corpus_name tree_name number_of_topics number_of_iterations number_of_mappers number_of_reducers update_hyper_parameters snapshot_interval"
#    exit
#fi

SET_PARAMETER

CorpusName=$DatasetName.$DatasetFormat.$DatasetLanguage

#Suffix=$(date +%y%b%d-%H%M%S)-$TreeName-K$NumTopic-I$Iterations-M$NumMapper-R$NumReducer
#Suffix=$TreeName.K$NumTopic.I$Iterations.M$NumMapper.R$NumReducer.$UpdateHyperParameter
Suffix=$TreeName.K$NumTopic.I$Iterations.M$NumMapper.R$NumReducer.$UpdateHyperParameter.$HybridMode

# set customizable pre-pipeline

#export HADOOP_HOME=/usr/lib/hadoop-0.20
#export HADOOP_HOME=/usr/lib/hadoop
export HADOOP_HOME=/fs/clip-lsbi/VirtualEnv/virenv/lib/hadoop
#export PYTHON_COMMAND=/opt/local/stow/python-2.7.2/bin/python
export PYTHON_COMMAND=/fs/clip-sw/rhel6/Scipy_Stack-1.0/bin/python

ProjectDirectory=/fs/clip-lsbi/Workspace/variational
ClusterDirectory=/user/zhaike/tree-prior

# generate local directories and files
LocalInputDirectory=$ProjectDirectory/input/$CorpusName
mkdir $ProjectDirectory/output/$CorpusName
LocalOutputDirectory=$ProjectDirectory/output/$CorpusName/$Suffix
mkdir $LocalOutputDirectory
LocalExchangeDirectory=$ProjectDirectory/output/$CorpusName/$Suffix/exchange
mkdir $LocalExchangeDirectory

cd $ProjectDirectory/src

if [ $UpdateHyperParameter = "false" ]; then
    if [ $HybridMode = "false" ]; then
	$PYTHON_COMMAND -m vb.prior.tree.dumbo.initializer \
	    --input_directory=$LocalInputDirectory \
	    --output_directory=$LocalOutputDirectory \
	    --tree_name=$TreeName \
	    --number_of_topics=$NumTopic
    else
	$PYTHON_COMMAND -m vb.prior.tree.dumbo.initializer \
	    --input_directory=$LocalInputDirectory \
	    --output_directory=$LocalOutputDirectory \
	    --tree_name=$TreeName \
	    --number_of_topics=$NumTopic \
	    --hybrid_mode
    fi
else
    if [ $HybridMode = "false" ]; then
	$PYTHON_COMMAND -m vb.prior.tree.dumbo.initializer \
	    --input_directory=$LocalInputDirectory \
	    --output_directory=$LocalOutputDirectory \
	    --tree_name=$TreeName \
	    --number_of_topics=$NumTopic \
	    --update_hyperparameter
    else
	$PYTHON_COMMAND -m vb.prior.tree.dumbo.initializer \
	    --input_directory=$LocalInputDirectory \
	    --output_directory=$LocalOutputDirectory \
	    --tree_name=$TreeName \
	    --number_of_topics=$NumTopic \
	    --update_hyperparameter \
	    --hybrid_mode
    fi
fi

# generate cluster directories and files
ClusterExchangeDirectory=$ClusterDirectory/$CorpusName/$Suffix
ClusterInputDirectory=$ClusterExchangeDirectory/input
ClusterOutputDirectory=$ClusterExchangeDirectory/output

hadoop fs -mkdir $ClusterInputDirectory
hadoop fs -mkdir $ClusterOutputDirectory
#hadoop fs -mkdir $ClusterExchangeDirectory

# update data to the cluster
hadoop fs -put $LocalInputDirectory/corpus.dat $ClusterInputDirectory/doc.dat

for i in $(seq 1 $Iterations);
do
    echo "Iteration $i"

    dumbo start $ProjectDirectory/src/vb/prior/tree/dumbo/inferencer.py \
	-hadoop $HADOOP_HOME \
	-name "inferencer-$i" \
	-input $ClusterInputDirectory/doc.dat \
	-output $ClusterOutputDirectory/ \
	-python $PYTHON_COMMAND \
	-hadoopconf mapred.child.java.opts=-Xmx4000m \
	-hadoopconf mapred.task.timeout=60000000 \
	-memlimit 2000000000 \
	-file $LocalOutputDirectory/current-tree \
	-file $LocalOutputDirectory/current-params \
	-file $LocalOutputDirectory/current-E-log-beta \
	-partitioner fm.last.feathers.partition.Prefix \
	-libjar $ProjectDirectory/lib/feathers.jar \
	-nummaptasks $NumMapper \
	-numreducetasks $NumReducer \
	-outputformat text \
	-overwrite yes
       	#> $LocalOutputDirectory/iter-$i.output

    wait

    rm -fv $LocalExchangeDirectory/*

    hadoop fs -get $ClusterOutputDirectory/* $LocalExchangeDirectory/

    if [ $[$i % $SnapshotInterval] -eq 0 ]; then
	$PYTHON_COMMAND -m vb.prior.tree.dumbo.formatter \
	    $LocalExchangeDirectory \
	    $LocalOutputDirectory \
	    $LocalOutputDirectory/word-beta-$i

	    cp -v $LocalOutputDirectory/current-E-log-beta $LocalOutputDirectory/E-log-beta-$i
	    cp -v $LocalOutputDirectory/current-params $LocalOutputDirectory/params-$i
    else
	$PYTHON_COMMAND -m vb.prior.tree.dumbo.formatter \
	    $LocalExchangeDirectory \
	    $LocalOutputDirectory
    fi

done

hadoop fs -rmr $ClusterExchangeDirectory
hadoop fs -rmr $LocalExchangeDirectory

# set customizable post-pipeline

if [ $UpdateHyperParameter = "false" ]; then
    MTOutputDirectory=$ProjectDirectory/output/mt-output-static
else
    MTOutputDirectory=$ProjectDirectory/output/mt-output-update
fi
MTOutputSubDirectory=$MTOutputDirectory/$CorpusName.$TreeName.K$NumTopic.I$Iterations.M$NumMapper.R$NumReducer.$UpdateHyperParameter.$HybridMode
mkdir $MTOutputSubDirectory

MTDevTestDirectory=/fs/clip-corpora/mt_corpora/chinese-english/dev_and_test

$PYTHON_COMMAND -m experiments.parse_gamma \
    $LocalOutputDirectory/gamma \
    $MTOutputSubDirectory/model.docs

SET_POST_PIPELINE

#$PYTHON_COMMAND -m experiments.test \
#    $ProjectDirectory/data/sent.fbis-zh/doc.dat \
#    $MTOutputSubDirectory/model.docs \
#    $LocalOutputDirectory/ \
#    $LocalInputDirectory/voc.dat