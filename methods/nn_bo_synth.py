import random
import numpy as np
import time
import tensorflow as tf
from random import randint
from scipy.stats import norm
from scipy.special import ndtr
from scipy.spatial.distance import cdist

class NeuralNetworkBOSynth():

    def __init__(self, config=None):

        # the surrogate models
        self.surrogate_model = None

        # the hyper-hyperparameter configurations of the method
        if config is None:
            self.config = {'is_rank_version': True, 'eta': 0.01, 'optim_iters': 300, 'train_batch_size': 30,
                           'acquisition_batch_size': 1000, 'log_iters': 300, 'hidden_layers_units': [16, 16, 16, 16],
                           'use_batch_norm': False, 'use_dropout': False, 'dropout_rate': 0.0, 'beta': 7.0}
        else:
            self.config = config

        # should the method be the rank version, or a standard surrogate
        self.is_rank_version = self.config['is_rank_version']

        # the optimizer
        self.optimizer = None
        # the metrics for storing the training performance
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        # the loss
        if self.is_rank_version:
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        else:
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()

        # the acquisition function
        if self.is_rank_version:
            self.acq_fun = self.UCB
        else:
            self.acq_fun = self.EI

        # store the number of hyperparameters in the current task
        self.num_hyperparameters = -1

        self.beta = self.config['beta']

    # some acquisition choices
    # probability of improvement
    def PI(self, max_val, mean, std, tradeoff=0.0):
        return ndtr((mean - max_val - tradeoff) / std)

    # expected improvement
    def EI(self, max_val, mean, std, tradeoff=0.0):
        z = (mean - max_val - tradeoff) / std
        return (mean - max_val - tradeoff) * ndtr(z) + std * norm.pdf(z)

    # upper confidence bound
    def UCB(self, max_val, mean, std):
        return mean + self.beta * std

    # create the surrogate model
    def create_surrogate_model(self, num_hyperparameters):

        # set the input and output dimensions of the surrogate, as well as the final activation
        self.num_hyperparameters = num_hyperparameters

        if self.is_rank_version:
            input_dimensionality = num_hyperparameters*2
        else:
            input_dimensionality = num_hyperparameters

        output_dimensionality = 1

        # define the input layer of the surrogate model
        x_input = tf.keras.Input(shape=(input_dimensionality, ), name='xInput')

        # define the hidden layers
        h = x_input
        for n in self.config['hidden_layers_units']:
            h = tf.keras.layers.Dense(units=n, activation='selu')(h)

            if self.config['use_batch_norm']:
                h = tf.keras.layers.BatchNormalization()(h)

            if self.config['use_dropout']:
                h = tf.keras.layers.Dropout(rate=self.config['dropout_rate'])(h)

        # define the output layer
        y_hats = tf.keras.layers.Dense(units=output_dimensionality, activation=None)(h)

        # create a surrogate model and print the summary
        self.surrogate_model = tf.keras.Model(inputs=x_input, outputs=y_hats)
        #self.surrogate_model.summary()

    # an update step on the surrogate, given the minibatch x,y
    @tf.function
    def update_surrogate(self, x, y):
        # forward pass and the loss computation
        with tf.GradientTape() as tape:
            y_hats = self.surrogate_model(x, training=True)
            loss = self.loss_fn(y_true=y, y_pred=y_hats)

        # compute gradients
        gradients = tape.gradient(loss, self.surrogate_model.trainable_variables)
        # apply back propagation
        self.optimizer.apply_gradients(zip(gradients, self.surrogate_model.trainable_variables))
        # store the training loss
        self.train_loss(loss)

    def draw_batch(self, X_obs, y_obs):

        # the number of observations as the length of the list of observations
        num_observations = len(X_obs)
        batch_size = self.config['train_batch_size']

        X_batch_list = []
        y_batch_list = []

        for _ in range(batch_size):

            x_batch, y_batch = None, None

            if self.is_rank_version:
                idx_1, idx_2 = randint(0, num_observations-1), randint(0, num_observations-1)
                x_batch = np.concatenate((X_obs[idx_1], X_obs[idx_2])).flatten()
                y_batch = (y_obs[idx_1] - y_obs[idx_2]).flatten()

            else:
                idx = randint(0, num_observations - 1)
                x_batch = X_obs[idx].flatten()
                y_batch = y_obs[idx].flatten()

            # add instance to the batch
            X_batch_list.append(x_batch)
            y_batch_list.append(y_batch)

        # create a numpy array of the batch features and targets
        X_batch = np.stack(X_batch_list)
        y_batch = np.stack(y_batch_list)

        return X_batch, y_batch

    # train the surrogate from the list of observations
    def train_surrogate(self, X_obs, y_obs):

        # keep track of the time
        start_time = time.time()

        # the number of observations as the length of the list of observations
        num_observations = len(X_obs)

        # re-set the optimizer and the loss rolling metric
        self.optimizer = tf.optimizers.Adam(learning_rate=self.config['eta'])

        self.train_loss.reset_state()

        # train the surrogate for a series of iterations
        for iter in range(self.config['optim_iters']):

            # draw a batch
            X_batch, y_batch = self.draw_batch(X_obs, y_obs)

            # update the surrogate model on the batch
            self.update_surrogate(X_batch, y_batch)

            if (iter+1) % self.config['log_iters'] == 0:
                template = 'Surrogate fitting: Num observations {}, Iter {}, Time: {:4.4f}, Loss: {:4.6f}'
                print(template.format(num_observations, iter+1, time.time() - start_time, self.train_loss.result()))


    # infer the test mean and std
    def infer(self, X, X_obs):

        y_hats = self.surrogate_model(X, training=False)

        # call the acquisition function
        y_mean = np.squeeze(np.array(y_hats)).astype(float)

        # compute the synthetic uncertainty

        # get the first element of the tuple of the batch
        if self.is_rank_version:
            X_first = X[:, :self.num_hyperparameters]
        else:
            X_first = X
        # convert the observed data points to a tensor
        X_obs_tensor = np.stack(X_obs)

        # compute pairwisse distances
        D = cdist(X_first, X_obs_tensor, lambda u, v: np.abs(u-v).mean())
        # compute weighted RBF
        #D = self.alpha*(np.tanh(self.gamma * D))
        # compute synthetic uncertainty
        y_std = np.min(D, axis=1).astype(float)

        return y_mean, y_std

    def inference_batch(self, batch_feasible_configs, X_obs, y_obs):

        x_inf_batch = None

        x_best_so_far = X_obs[np.argmax(y_obs)]

        if self.is_rank_version:
            x_raw_batch = batch_feasible_configs
            x_best_so_far_batch = np.tile(x_best_so_far, (len(x_raw_batch), 1))
            x_inf_batch = np.concatenate((x_raw_batch, x_best_so_far_batch), axis=1)
        else:
            x_inf_batch = batch_feasible_configs

        return x_inf_batch

    # recommend the next configuration from a list of feasible configurations
    def recommend_next_config(self, X_obs, y_obs, feasible_configs):
        # the index in the feasible configurations for the incumbent
        incumbent_idx = 0
        incumbent_acquisition_value = -1
        y_max = np.max(y_obs).astype(float)

        # create batches
        inference_batches = []
        for i in range(0, len(feasible_configs), self.config['acquisition_batch_size']):
            x_inf_batch = self.inference_batch(batch_feasible_configs=feasible_configs[i:i + self.config['acquisition_batch_size']],
                                               X_obs=X_obs, y_obs=y_obs)
            inference_batches.append(x_inf_batch)

        # iterate through the feasible configurations and return the index of the config with the highest acquisition
        i = 0
        for idx, x_batch in enumerate(inference_batches):
            # infer the batch's posterior predictions
            y_hats_mean, y_hats_std = self.infer(X=x_batch, X_obs=X_obs)
            # convert numpy versions of the posteriors to lists
            y_hats_mean, y_hats_std = y_hats_mean.tolist(), y_hats_std.tolist()
            # check if the configuration is the incumbent
            for idx_batch, x in enumerate(x_batch):
                acquisition_val = self.acq_fun(y_max, y_hats_mean[idx_batch], y_hats_std[idx_batch])

                if acquisition_val >= incumbent_acquisition_value:
                    incumbent_idx = i
                    incumbent_acquisition_value = acquisition_val
                    #print(i, acquisition_val, y_max, y_hats_mean[idx_batch], y_hats_std[idx_batch])
                i += 1

        print('Acquisition: Utility', incumbent_acquisition_value, 'over incumbent', y_max)

        return incumbent_idx

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):

        if X_pen is not None:

            # check if this the first ever call for a task, if yes initialize the surrogate model
            if self.surrogate_model is None:
                # create the surrogate model
                current_num_hyperparameters = len(X_obs[0])
                self.create_surrogate_model(current_num_hyperparameters)

            # train the surrogate using the history
            self.train_surrogate(X_obs, y_obs)

            return self.recommend_next_config(X_obs, y_obs, X_pen)

        else:
            raise Exception('The continuous HPO variant is not yet implemented')
