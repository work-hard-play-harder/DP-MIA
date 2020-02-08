# Membership Inference Attack (MIA)
The goal of MIA is to infer whether a given data record is in the target dataset. To construct the MIA model, a shadow training technique is applied to generate the ground truth for membership inference. Fig. 1 shows an overview of MIA.

1. Target model
To investigate whether DP is empirically effective against MIA on genomic data analysis, we construct MIA on the best target models evaluated in differential privacy part for various DP budget levels. 

2. Shadow model
In this study, we focus on a white-box model attack, where the target model’s architecture and weights are accessible. Thus the shadow model has the same architecture as the target model. The shadow model is trained using the same hyperparameters as those used in the target model. We build one shadow model on the shadow dataset to mimic the target model, and generate the ground truth to train the attack model.

3. Attack model
The attack dataset is constructed by concatenating probability vector output by the shadow model and true labels. If the sample is used to train the shadow model, the corresponding concatenated input for the attack dataset is labeled ‘in’, and ‘out’ for otherwise. We build a random forest of 10 estimators with max depth of 2 as the attack model.


<div align="center">  
<img src="overview-of-mia.png" width="600" />  
<p>Figure 1. An illustration of membership inference attack.</p>  
</div>  


We use an [open-source library of MIA](https://github.com/spring-epfl/mia) to conduct MIA attack on the Lasso and CNN models running on yeast genomic data. The worst case of MIA is that the private training dataset for the target model is disjoint from the public dataset used to train the shadow model. The attack will perform better if the training datasets happen to overlap for the target and shadow models. We consider the worst case of MIA, and split the whole dataset into two disjoint subsets, one as the private target dataset and the other one as the public shadow dataset for training differentially private machine learning models and performing MIA correspondingly. We randomly split the public shadow dataset into 80% for training the model and 20% for unused to generate the ground truth of the attack model. Each MIA attack is randomly repeated 5 times.  

## MIA on target models with DP
To evaluate whether DP is empirically effective against MIA on genomic data analysis, we conduct MIA on the target models with different DP budget levels. Fig. 2 shows the protection effect of different privacy budget levels against MIA. Generally speaking, attack accuracy is reduced as the privacy budget become smaller. For Lasso model, a smaller privacy budget (ε ≤ 10) rapidly reduces the attack accuracy. For CNN model, any privacy budget can significantly reduce attack accuracy comparing with that CNN model without DP. However, smaller privacy budget can’t provide more significant protection.

The results demonstrate that DP can defend against MIA effectively. The attack accuracy can be reduced significantly by reducing the DP budget.  

<div align="center">  
<img src="effect-MIA.png" width="600" />  
<p>Figure 2. The performance of attack model against target model with DP. The horizontal line represents the performance of attack model against target model without DP. Attack accuracy is measured by mean accuracy of 5-fold cross validation. </p>  
</div>  