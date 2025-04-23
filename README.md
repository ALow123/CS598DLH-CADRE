# DLH598 Project: Reproducing SADRE & CADRE
Team: Aaron Low (aclow2) & Kevin Zhou (kevinjz3)

## Introduction

In this DLH598 project, we aim to reproduce the work of "Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention" by Yifeng Tao, Shuangxia Ren, Michael Q. Ding, Russell Schwartz, and Xinghua Lu. The reproduction focuses on models SADRE and CADRE which are built on collaborative filtering, then adding attentnion mechanisms expanded with context.

## Code Execution

To execute the research paper on GCB (Google Collab), perform the following steps:
1. Download the repository as a ZIP file or clone the repository.
2. Upload the repository to your Google Drive. Make sure this is your own "Drive" rather than a shared drive as the path will be different.
3. Change the path variables in these files (bases.py, utils.py, run_cf.py) to your respective path in your Google Drive foder.
<br/>**Note**: The start of your path will contain "/content/drive/MyDrive/". Add the rest of the path that points to where your downloaded repository is.
4. Open the "Project_Code.ipynb" file and execute the instructions listed. 
<br/>**Note**: Be sure to change the runtime to the T4 GPU in order for resources not to be maxed. Consider paying for premium GPUs if you would like to test larger hyperparameters.
5. If you would like to change any parameters, open run_cf.py and change the hyperparameters listed in the args dictionary.

## Data Download Instructions

To acquire the data for the research reproduction, the researchers provided the data as part of their GitHub repository. While multiple files are provided, not all are used as part of the experiment and in their code. The following files listed in the “input” folder of the Github repository are used in the experiment: gdsc.csv, exp_gdsc.csv, drug_info_gdsc.csv, exp_emb_gdsc.csv, and rng.txt. Below are instructions to download the data for alternative analysis:
1. Visit the paper’s provided GitHub repository with this link: https://github.com/yifengtao/CADRE/tree/master 
2. Download the repository as a ZIP file or clone the repository.
3. Enter the `data/input` folder and access the files 


## Citation

Yifeng Tao<sup>＊</sup>, Shuangxia Ren<sup>＊</sup>, Michael Q. Ding, Russell Schwartz<sup>†</sup>, Xinghua Lu<sup>†</sup>. [**Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention**](http://proceedings.mlr.press/v126/tao20a.html). Proceedings of the Machine Learning for Healthcare Conference (***MLHC***). 2020.
