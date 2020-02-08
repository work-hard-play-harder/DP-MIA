# Differential Privacy Protection against Membership Inference Attack on Genomic Data

## Introduction
Motivation: Machine learning is powerful to model massive genomic data while genome privacy is a growing concern. Studies have shown that not only the raw data but also the trained model can potentially infringe genome privacy. An example is the membership inference attack (MIA), by which the adversary, who can only query a given model without knowing its internal parameters, can determine whether a specific record is in the model’s training dataset. Differential privacy (DP) has become a de facto privacy standard for publicly sharing information by describing the statistics of groups within the dataset while withholding information about individuals in the dataset. DP can be a solution for granting wider access to machine learning models while preventing risk of MIA.
Results: We investigate the vulnerability of machine learning on genomic data against MIA, and evaluate the effectiveness of using DP as a defense mechanism against MIA. We train two widely-used machine learning models, namely Lasso and convolutional neural network (CNN), with and without DP on real-world yeast genomic data. We study the effect of various DP budget levels on model accuracy as well as on its defense power against MIA. The results show that smaller privacy budget provides stronger privacy guarantee while leads to worse accuracy performance.We conduct MIA on the trained models with various privacy budget levels and show that DP hurts model accuracy as stronger DP leads to reduced accuracy. We investigate other factors including model over-fitting and sparsity, that affect model vulnerability against MIA. Our results demonstrate that, for any privacy budget, model sparsity can enhance model defense against MIA.


## Differential Privacy (DP)
DP is a state-of-the-art privacy protection standard. It requires that a mechanism outputting information about an underlying dataset is robust to any change of one sample.  It has been shown that DP can be an effective solution for granting wider access to machine learning models and results while keep them private. To investigate the impact of DP on the utility of machine learning models, we incorporate DP to two machine learning methods, namely Lasso and Convolutional neural network (CNN), that are commonly used for genomic data analysis. 

The corresponding code is in folder of differential-privacy.
```
Differential Privacy
|-- Lasso-dp.py
|-- CNN-dp.py
```

## Membership Inference Attack
Membership inference attack (MIA) is a privacy-leakage attack that predicts whether a given record was used in training a target model. It works under a setting where the target model is opaque but remotely accessible. It predicts whether a given record was in the model’s training dataset based on the output of the target model for the given record.

The corresponding code is in folder of membership-inference-attack.
```
Membership Inference Attack
|-- Lasso-MIA.py
|-- CNN-MIA.py
```

## Getting Started

### Prerequisites
```
Python >= 3.6 
virtualenv >= 16.4.3
```
### Setup
1. Create virtual environment
```
git clone https://github.com/shilab/DP-MIA.git
cd DP-MIA/
mkdir venv
python3 -m venv venv/
source venv/bin/activate
```
2. Install requirement dependents
```
pip install tensorflow==1.14 tensorflow_privacy sklearn pandas jupyter mia
```

## Citation
Junjie Chen, Hui Wang, Xinghua Shi, Differential Privacy Protection against Membership Inference Attack on Genomic Data, in submission.
