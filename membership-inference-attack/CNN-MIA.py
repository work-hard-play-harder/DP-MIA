import numpy as np
import pandas as pd

from mia.estimators import ShadowModelBundle, prepare_attack_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# depent on tensorflow 1.14
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical

# privacy package
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

# set random seed
np.random.seed(19122)

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer


class DataGenerator(object):
    """
    Load and preprocess data: filter NA, binarize phenotype, balance sample, one hot encoding.
    """

    def __init__(self, genotype_file, phenotype_file, shuffle=True):
        super(DataGenerator, self).__init__()
        self.genotype_file = genotype_file
        self.phenotype_file = phenotype_file
        self.shuffle = shuffle

        # preprocess
        self._load_data()
        self._filter_na_phenotype()
        self._binarize_phenotype()
        self._balance_sample()
        self._one_hot_encode()

    def _load_data(self):
        self.genotype = pd.read_csv(genotype_file, sep='\t', index_col=0)
        # print('genotype_file shape:', genotype.shape)
        self.genotype[self.genotype == -1] = 0

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
            # random_state=27 # reproducible results
        )

        self.genotype = pd.concat(
            [majority_downsampled, self.genotype[minority_index]], axis=0)
        if self.shuffle:
            self.genotype = self.genotype.sample(frac=1)
        self.phenotype = self.phenotype[self.genotype.index]

    def _one_hot_encode(self):
        self.onehot = to_categorical(self.genotype)


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

    target_X = X[target_row]
    shadow_X = X[shadow_row]

    target_y = y.iloc[target_row]
    shadow_y = y.iloc[shadow_row]

    return target_X, target_y, shadow_X, shadow_y


def target_model():
    """The architecture of the target model.
    The attack is white-box, hence the attacker is assumed to know this architecture too.

    :return: target model
    """

    classifier = Sequential()
    classifier.add(Conv1D(num_kernels,
                          kernel_size,
                          padding='same',
                          activation='relu',
                          kernel_regularizer=l1(kernel_regularization),
                          input_shape=input_shape)
                   )
    classifier.add(AveragePooling1D(pool_size=2))
    classifier.add(Dropout(drop_prec))
    classifier.add(Flatten())
    classifier.add(Dense(1, activation='sigmoid'))

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


def shadow_model():
    """The architecture of the shadow model is same as target model, because the attack is white-box,
    hence the attacker is assumed to know this architecture too.

    :return: shadow model
    """

    classifier = Sequential()
    classifier.add(Conv1D(num_kernels,
                          kernel_size,
                          padding='same',
                          activation='relu',
                          kernel_regularizer=l1(kernel_regularization),
                          input_shape=input_shape)
                   )
    classifier.add(AveragePooling1D(pool_size=2))
    classifier.add(Dropout(drop_prec))
    classifier.add(Flatten())
    classifier.add(Dense(1, activation='sigmoid'))

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
    print("Training the target model...")
    # split target dataset to train and valid, and make them evenly divisible by batch size
    target_X_train, target_y_train, target_X_valid, target_y_valid = split_to_be_divisible(target_X,
                                                                                           target_y,
                                                                                           0.2,
                                                                                           batch_size)

    tm = target_model()
    tm.fit(target_X_train,
           target_y_train,
           batch_size=batch_size,
           epochs=epochs,
           validation_data=[target_X_valid, target_y_valid],
           verbose=1)

    print("Training the shadow models.")
    # train only one shadow model
    SHADOW_DATASET_SIZE = int(shadow_X.shape[0] / 2)
    smb = ShadowModelBundle(
        shadow_model,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=1,
    )
    # Training the shadow models with same parameter of target model, and generate attack data...
    attacker_X, attacker_y = smb.fit_transform(shadow_X, shadow_y.values,
                                               fit_kwargs=dict(epochs=epochs,
                                                               batch_size=batch_size,
                                                               verbose=1),
                                               )

    print("Training attack model...")
    clf = RandomForestClassifier(max_depth=2)
    clf.fit(attacker_X, attacker_y)

    # Test the success of the attack.
    ATTACK_TEST_DATASET_SIZE = unused_X.shape[0]
    # Prepare examples that were in the training, and out of the training.
    data_in = target_X_train[:ATTACK_TEST_DATASET_SIZE], target_y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = unused_X[:ATTACK_TEST_DATASET_SIZE], unused_y[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(tm, data_in, data_out)

    # Compute the attack accuracy.
    attack_guesses = clf.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print('attack accuracy: {}'.format(attack_accuracy))


if __name__ == '__main__':
    # genotype
    genotype_file = '../data/genotype_full.txt'
    # phenotype
    phenotype_file = '../data/phenotype.csv'

    # parameters
    dpsgd = True

    # target model hyper-parameters same as Lasso-dp
    epochs = 50
    batch_size = 16
    microbatches_perc = 1.0
    learning_rate = 0.01
    kernel_regularization = 0.001352
    drop_prec = 0.25
    num_kernels = 8
    kernel_size = 5
    noise_multiplier = 0.8
    l2_norm_clip = 1.0

    print("Loading and splitting dataset")
    yeast = DataGenerator(genotype_file, phenotype_file)
    target_X, target_y, shadow_X, shadow_y = split_to_be_divisible(yeast.onehot,
                                                                   yeast.phenotype,
                                                                   0.5,
                                                                   batch_size=80)
    shadow_X, shadow_y, unused_X, unused_y = split_to_be_divisible(shadow_X,
                                                                   shadow_y,
                                                                   0.2,
                                                                   batch_size)

    input_shape = (target_X.shape[1], target_X.shape[2])

    # # define the grid search parameters
    # if dpsgd:
    #     param_grid = {
    #         'epochs': [50, 100],
    #         'batch_size': [8, 16],
    #         'microbatches_perc': [0.5, 1.0],
    #         'learning_rate': [0.01, 0.001],
    #         'kernel_regularization': [0, 0.001352],
    #         'drop_prec': [0.25, 0.5],
    #         'num_kernels': [8, 16],
    #         'kernel_size': [5, 9],
    #         'noise_multiplier': [0.4, 0.6, 0.8, 1.0, 1.2],
    #         'l2_norm_clip': [0.6, 1.0, 1.4, 1.8],
    #         'verbose': [0]
    #     }
    # else:
    #     # define the grid search parameters
    #     param_grid = {
    #         'epochs': [50, 100],
    #         'batch_size': [8, 16],
    #         'learning_rate': [0.01, 0.001],
    #         'kernel_regularization': [0, 0.001352],
    #         'drop_prec': [0.25, 0.5],
    #         'num_kernels': [8, 16],
    #         'kernel_size': [5, 9],
    #         'verbose': [0]
    #     }

    main()
