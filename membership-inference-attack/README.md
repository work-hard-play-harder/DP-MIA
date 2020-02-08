# Membership Inference Attack (MIA)
The goal of MIA is to infer whether a given data record is in the target dataset. The target model is a white/black access model trained on a labeled private target dataset. A given labeled record is fed into the target model and outputs a predicted probability vector. The shadow model mimics the target model. Shadow dataset and unused dataset are either simulated or used from publicly available datasets that have the same format and distribution as the target dataset. The attack dataset is composed of the probability vectors and true labels. The attack model performs binary classification (1/0) to determine whether a data record is included in the training dataset or not.

We use an [open-source library of MIA](https://github.com/spring-epfl/mia) to conduct MIA attack on the Lasso and CNN models running on yeast genomic data. The worst case of MIA is that the private training dataset for the target model is disjoint from the public dataset used to train the shadow model. The attack will perform better if the training datasets happen to overlap for the target and shadow models. We consider the worst case of MIA, and split the whole dataset into two disjoint subsets, one as the private target dataset and the other one as the public shadow dataset for training differentially
private machine learning models and performing MIA correspondingly. We randomly split the public shadow dataset into 80% for training the model and 20% for unused to generate the ground truth of the attack model. Each MIA attack is randomly repeated 5 times.

<div align="center">
<img src="overview-of-mia.png" width="700" />  
<p>Figure 1. An illustration of membership inference attack.</p>
</div>

## Target model with DP
To investigate whether DP is empirically effective against MIA on genomic data analysis, we constructe MIA on the best target models evaluated in differential privacy part for various DP budget levels. 

## Shadow model
In this study, we focus on a white-box model attack, where the targetmodel’s architecture and weights are accessible. Thus the shadow model has the same architecture as the target model. The shadow model is trained using the same hyperparameters as those used in the target model. We build one shadow model on the shadow dataset to mimic the target model, and generate the ground truth to train the attack model.

## Attack model
The attack dataset is constructed by concatenating probability vector output by the shadow model and true labels. If the sample is used to train the shadow model, the corresponding concatenated input for the attack dataset is labeled ‘in’, and ‘out’ for otherwise. For the attack model, we build a random forest of 10 estimators with max depth of 2.

