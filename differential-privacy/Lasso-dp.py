import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
# depent on tensorflow 1.14
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# privacy package
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

# set random seed
np.random.seed(19122)

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer


class DataGenerator(object):
    """
    Load and preprocess data: filter NA, binarize phenotype, balance sample.
    """

    def __init__(self, genotype_file, phenotype_file, shuffle=True):
        super(DataGenerator, self).__init__()
        self.genotype_file = genotype_file
        self.phenotype_file = phenotype_file
        self.shuffle = shuffle

        # load data
        self._load_data()
        # preprocess
        self._filter_na_phenotype()
        self._binarize_phenotype()
        self._balance_sample()

    def _load_data(self):
        self.genotype = pd.read_csv(genotype_file, sep='\t', index_col=0)
        # print('genotype_file shape:', genotype.shape)

        self.multi_pheno = pd.read_csv(phenotype_file, sep=',', index_col=0)
        self.phenotype = self.multi_pheno.iloc[:, 2]
        # print('phenotype shape:', phenotype.shape)

    def _filter_na_phenotype(self):
        missing_mask = self.phenotype.isna()

        self.genotype = self.genotype[~missing_mask]
        self.phenotype = self.phenotype[~missing_mask]

    def _binarize_phenotype(self):
        self.phenotype[self.phenotype > 0] = 1
        self.phenotype[self.phenotype < 0] = 0

    def _balance_sample(self):
        # find majority and minority
        num_zeros = self.phenotype.value_counts()[0]
        num_ones = self.phenotype.value_counts()[1]
        if num_ones > num_zeros:
            majority = 1
            minority = 0
        else:
            majority = 0
            minority = 1

        # downsampling majority
        majority_index = self.phenotype == majority
        minority_index = self.phenotype == minority
        majority_downsampled = resample(
            self.genotype[majority_index],
            replace=True,
            n_samples=self.phenotype.value_counts()[minority],
        )

        self.genotype = pd.concat(
            [majority_downsampled, self.genotype[minority_index]], axis=0)
        if self.shuffle:
            self.genotype = self.genotype.sample(frac=1)
        self.phenotype = self.phenotype[self.genotype.index]


def split_to_be_divisible(X, y, shadow_perc, batch_size):
    """
    Split a dataframe into target dataset and shadow dataset, and make them divisible by batch size.

    :param X: genotype data
    :param y: phenotype data
    :param shadow_perc: specified percent for shadow dataset, target_perc = 1 - shadow_perc
    :param batch_size: batch_size for training process

    :return: target datasets, shadow datasets
    """

    # stop and output error, if X and y have different number of individuals.
    assert y.shape[0] == X.shape[0]

    # calculate sample size of target and shadow
    total_row = X.shape[0]
    num_shadow_row = int(total_row * shadow_perc) - int(total_row * shadow_perc) % batch_size
    num_target_row = (total_row - num_shadow_row) - (total_row - num_shadow_row) % batch_size

    # split train and valid
    random_row = np.random.permutation(total_row)
    shadow_row = random_row[:num_shadow_row]
    target_row = random_row[-num_target_row:]

    target_X = X.iloc[target_row]
    shadow_X = X.iloc[shadow_row]

    target_y = y.iloc[target_row]
    shadow_y = y.iloc[shadow_row]

    return target_X, target_y, shadow_X, shadow_y


def target_model(feature_size, kernel_regularization=0, batch_size=16, microbatches_perc=1.0, learning_rate=0.01,
                 l2_norm_clip=1.0, noise_multiplier=1.0
                 ):
    """
    build target model with dp if dpsgd is true

    :param feature_size: dimension of genotype
    :param kernel_regularization: value of lambda. There is no sparsity if 0.
    :param l2_norm_clip: to determines the maximum amounts of L2 norm clipped to cumulative gradient across all network parameters from each microbatch.
    :param noise_multiplier: Governs the amount of noise added during training. Generally, adding more noise results in better privacy and lower utility.
    :param batch_size: The number of input data for updating gradients
    :param microbatches_perc: the percent of batch size. Microbatch size is the number of input data for each step. This number should evenly divide input batch size.
    :param learning_rate: The higher the learning rate, the more each update matters.

    :return: target model
    """
    classifier = Sequential()
    classifier.add(Dense(1,
                         input_dim=feature_size,
                         kernel_regularizer=l1(kernel_regularization),
                         activation='sigmoid')
                   )

    if dpsgd:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=int(microbatches_perc * batch_size),
            learning_rate=learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.compat.v2.losses.Reduction.NONE)
    else:
        optimizer = GradientDescentOptimizer(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Compile model with Keras
    classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return classifier


def main():
    print("Loading data")
    yeast = DataGenerator(genotype_file, phenotype_file)
    # split yeast data to target and shadow
    target_X, target_y, shadow_X, shadow_y = split_to_be_divisible(yeast.genotype,
                                                                   yeast.phenotype,
                                                                   0.5,
                                                                   batch_size=80)
    print("Creating target model")
    keras_estimator = KerasClassifier(build_fn=target_model,
                                      feature_size=target_X.shape[1])
    print("Starting grid search")
    grid = GridSearchCV(estimator=keras_estimator,
                        param_grid=param_grid,
                        n_jobs=1,
                        cv=5,
                        verbose=10)
    grid_result = grid.fit(target_X, target_y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # save
    gridCV_result_zip = {
        'best_score': grid_result.best_score_,
        'best_params': grid_result.best_params_,
        'cv_results': zip(means, stds, params)
    }
    pickle.dump(gridCV_result_zip, open(os.path.join('cv_results', filename), "wb"))

if __name__ == '__main__':
    # genotype
    genotype_file = '../data/genotype_full.txt'
    # phenotype
    phenotype_file = '../data/phenotype.csv'

    # save results
    filename = "Lasso_grid_result_dp.pkl"

    # use differential privacy
    dpsgd = True

    # define the grid search parameters
    if dpsgd:
        param_grid = {
            'epochs': [50, 100],
            'batch_size': [8, 16],
            'microbatches_perc': [0.5, 1],
            'learning_rate': [0.01, 0.001],
            'kernel_regularization': [0, 0.001352],
            'noise_multiplier': [0.4, 0.6, 0.8, 1.0, 1.2],
            'l2_norm_clip': [0.6, 1.0, 1.4, 1.8],
            'verbose': [0]
        }
    else:
        # define the grid search parameters
        param_grid = {
            'epochs': [50, 100],
            'batch_size': [8, 16],
            'learning_rate': [0.01, 0.001],
            'kernel_regularization': [0, 0.001352],
            'verbose': [0]
        }

    main()
