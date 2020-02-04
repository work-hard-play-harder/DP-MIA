# Differential Privacy (DP)
DP measures privacy risk by a parameter ε, called privacy budget, that bounds the log-likelihood ratio of the output of an algorithm under two databases differing in a single individual’s data. Thus ε-differential privacy, a mathematical definition of DP, associates the privacy losswith any data release drawn froma statistical database. If the training model is differentially private with a small privacy budget, the probability of producing a given model from a training dataset that includes a particular record is close to the probability of producing the same model when this record is not included. DP has been used in many machine learning models, including classification, regression, clustering, and dimensionality reduction.

## Privacy budget
Privacy budget ε is a function of epoch, batch size, and noise multiplier. To investigate the effectiveness of different private budget levels, we utilize grid search with 5-fold cross validation to explore the space of hyperparameters: epoch ∈ {50, 100}, batch ∈ {8, 16} and noise multiplier ∈ {0.4, 0.6, 0.8, 1.0, 1.2}, under a fixed δ= 0.00066489 which is the inverse of training dataset size. Noise multiplier governs the amount of noise added during training. Generally, adding more noise results in better privacy and lower utility.

Table 1. The evaluated privacy budgets levels.

|  Epsilon  	| epoch 	| batch 	| noise_multiplier 	|
|:---------:  |:-----:	|:-----:	|:----------------:	|
| 2.160180  	| 50    	|   8   	|        1.2       	|
| 2.918704  	| 50    	|   8   	|        1.0       	|
| 3.110184  	| 100   	|   8   	|        1.2       	|
| 3.159608  	| 50.0  	|   16  	|        1.2       	|
| 4.211902  	| 100.0 	|   8   	|        1.0       	|
| 4.306786  	| 50.0  	|   16  	|        1.0       	|
| 4.577880  	| 100.0 	|   16  	|        1.2       	|
| 4.643344  	| 50.0  	|   8   	|        0.8       	|
| 6.259435  	| 100.0 	|   16  	|        1.0       	|
| 6.684719  	| 100.0 	|   8   	|        0.8       	|
| 6.895937  	| 50.0  	|   16  	|        0.8       	|
| 10.039137 	| 100.0 	|   16  	|        0.8       	|
| 10.620311 	| 50.0  	|   8   	|        0.6       	|
| 15.289721 	| 100.0 	|   8   	|        0.6       	|
| 15.332044 	| 50.0  	|   16  	|        0.6       	|
| 22.568996 	| 100.0 	|   16  	|        0.6       	|
| 47.658010 	| 50.0  	|   8   	|        0.4       	|
| 61.954299 	| 50.0  	|   16  	|        0.4       	|
| 72.945059 	| 100.0 	|   8   	|        0.4       	|
| 98.002301 	| 100.0 	|   16  	|        0.4       	|


## Hyperparameters setup
Besides above hyperparameters, following hyperparameters also can affect the final perfomance. We optimized them by using grid search with 5-fold cross validation. 
* The learning rate of SGD: The higher the learning rate, the more each update matters. In our study, we use two different learning rates, 0.01 and 0.001.
* Microbatch size: Microbatch size is the number of input data for each step. This number evenly divides input batch size. We use two microbatch sizes {0.5, 1.0} percentage of batch size.
* ℓ2 norm clipping: it is determines the maximum amounts of ℓ2 norm clipped to cumulative gradient across all network parameters from each microbatch. We use four unique ℓ2 norm clipping values, 0.6, 1.0, 1.4, 1.8.
* Kernel size, specifically for CNN: Larger kernel size represents larger receptive field. The kernel size is set as either 5 or 9. 
* Number of kernels, specifically for CNN: Larger number of kernels generates more feature map, which can capture more correlations oflocal genotypes. The number of kernels is in {8, 16}.
* ℓ1 norm for model sparsity: Larger values of the ℓ1 norm lead to more sparse weights while a value of 0 represents no sparsity. We set ℓ1 norm values as 0.001352, selected by using glmnet package in R programming language.
