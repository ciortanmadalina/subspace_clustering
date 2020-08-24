# Subspace clustering optimization algorithm for high dimensional data
This repository contains the implementation of the methodology article "Optimization algorithm for subspace clustering of high dimensional data".
The proposed method targets the subspace clustering problem by using an evolutionary algorithm that optimizes an internal clustering score.   
The method is generic but several applications on different types of transcriptomic data (microarray, bulk RNA-seq, single-cell RNA-seq) are presented.  
The typical high dimensionality has been addressed with a dedicated feature sampling strategy as well as by approximating the internal clustering score of feature pairs with a neural network. 
Finally, we proposed several strategies to interpret the discovered subspaces from a biological point of view.


## Overview of the repository

- __data__ folder contains all datasets used in our experiments. The content can be downloaded from this [Google Drive Link](https://drive.google.com/file/d/1BAuszp7VzCpRfHJbBE6AFH5IM1zQNiPg/view?usp=sharing)
- __scripts__ folder contains the implementation of all functionalities as python scripts
- __docker__ folder contains the Dockerfile
- __model__ contains the pretrained models
- __notebooks__ folder comprises all experiments presented in the article and detailed below. They have been coded as jupyter notebooks.


## Environment Setup
We have created a docker container to facilitate reproducing the paper results.   

It can be launched by running the following:  

cd docker  
docker build -t discovers .  

The image has been created for GPU usage. In order to run it on CPU, in the Dockerfile, the line "FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter" should be replaced with "FROM tensorflow/tensorflow:1.15.2-py3-jupyter"

The command above created a docker container tagged as __discovers__ . Assuming the project has been cloned locally in a parent folder named __notebooks__, the image can be launched locally with:  


docker run -it --runtime=nvidia -v ~/notebooks:/tf/notebooks -p 8888:8888 discovers

This starts up a jupyter notebook server, which can be accessed at http://localhost:8888/tree/notebooks


## Data
We have made available all datasets and intermediary results needed to run the notebooks. Due to the upload limit on git, we provide access to this folder at this 
 [Google Drive Link](https://drive.google.com/file/d/1BAuszp7VzCpRfHJbBE6AFH5IM1zQNiPg/view?usp=sharing). Download and unzip the file in the data folder.   

 We have made available the already processed datasets in pickle format.   
 However, the original datasets can be accessed as follows:

 - Ramhiser microarray data : https://github.com/ramhiser/datamicroarray
 - Bulk RNA-seq data: Broad institute GDAC FireBrowse Version 1.1.35 (https://gdac.broadinstitute.org/)
 - Single cell data published in:  

MouseES:  A. M. Klein et al., “Droplet barcoding for single-cell transcriptomics applied to embryonic stem cells,” Cell, vol. 161, no. 5, pp. 1187–1201, May 2015, doi: 10.1016/j.cell.2015.04.044.


and  
PBMC: G. X. Y. Zheng et al., “Massively parallel digital transcriptional profiling of single cells,” Nat. Commun., vol. 8, no. 1, pp. 1–12, Jan. 2017, doi: 10.1038/ncomms14049.

## Project demo

The entry point for discovering the project is __Main.ipynb__, which simulates data and performs both unsupervised and semi-supervised analysis.

In order to perform the 2d feature ranking, the method expects a trained model at path ../models/gmm_arl.h5 (corresponding to GMM custering using Ratkowski Lance Score ) or ../models/hdbscan_as.h5 (HDBSCAN with Silhouette). We already provided pretrained models on up to 20 clusters (/models folder)

If a new model training is needed/desired, it can be performed by following the steps indicated below:

- Creation of training data (NN_generate_data_for_training.ipynb)
- Training the model corresponding to the desired clustering algorithm (NN_GMM.ipynb, NN_HDBSCAN.ipynb)




## Notebooks

Find below the list of notebooks and a short description of the implemented functionality:
- Main.ipynb: demos subspace clustering on simulated data, both in unsupervised and semi-supervised modes
- BRCA.ipynb, KIRP.ipynb: analysis of RNA-seq data
- Microarray_GMM.ipynb, Microarray_HDBSCAN.ipynb: analysis of microarray data
- Patient_metadata.ipynb, Cell_lines.ipynb, AnnotatedGenesAnalysis.ipynb, Survival_analysis.ipynb, Gene_enrichment.ipynb explore various techniques to interpret subspaces found on biological data
- NN_execution_time.ipynb computes the speed of using the NN approximation instead of clustering + internal evaluation
- InternalScoresAnalysis.ipynb compares various internal evaluators

