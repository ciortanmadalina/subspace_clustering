# Subspace clustering optimization algorithm for high dimensional data

## Environment Setup
We have created a docker container to facilitate reproducing the paper results.   

It can be launched by running the following:  

cd docker  
docker build -t discovers .  

The image has been created for GPU usage. In order to run it on CPU, in the Dockerfile, the line "FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter" should be replaced with "FROM tensorflow/tensorflow:1.15.2-py3-jupyter"

The command above created a docker container tagged as __discovers__ . Assuming the project has been cloned locally in a parent folder named __notebooks__, the image can be launched locally with:  


docker run -it --runtime=nvidia -v ~/notebooks:/tf/notebooks -p 8888:8888 discovers

This starts up a jupyter notebook server, which can be accessed at http://localhost:8888/tree/notebooks


## Project demo

The entry point for discovering the project is __Main.ipynb__, which simulates data and performs both unsupervised and semi-supervised analysis.

In order to perform the 2d feature ranking, the method expects a trained model at path models/gmm_arl.h5 (corresponding to GMM custering using Ratkowski Lance Score ) or models/hdbscan_as.h5 (HDBSCAN with Silhouette). We already provided pretrained models on up to 20 clusters (/models folder)

If a new model training is needed/desired, it can be performed by following the steps indicated below:

- Creation of training data (NN_generate_data_for_training.ipynb)
- Training the model corresponding to the desired clustering algorithm (NN_GMM.ipynb, NN_HDBSCAN.ipynb)

## Overview of the repository

- Main.ipynb: demos subspace clustering on simulated data
- BRCA.ipynb, KIRP.ipynb: analysis of RNA-seq data
- Microarray_GMM.ipynb, Microarray_HDBSCAN.ipynb: analysis of microarray data
- Patient_metadata.ipynb, Cell_lines.ipynb, AnnotatedGenesAnalysis.ipynb, Survival_analysis.ipynb, Gene_enrichment.ipynb explore various techniques to interpret subspaces found on biological data
- NN_execution_time.ipynb computes the speed of using the NN approximation instead of clustering + internal evaluation
- InternalScoresAnalysis.ipynb compares various internal evaluators

