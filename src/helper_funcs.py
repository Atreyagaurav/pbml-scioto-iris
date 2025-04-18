import os
from typing import List, Union
import pandas as pd
import numpy as np
import seaborn as sns
import setuptools.dist
from tensorflow.keras import layers, activations, callbacks, optimizers, models
from tensorflow.keras import Model
from tensorflow.keras.activations import relu
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def probabilistic_layer(y_pred: tf.Tensor) -> tfp.distributions.Distribution:
    distribution = tfp.distributions.Normal(loc=y_pred[...,0], scale=y_pred[...,1])
    return distribution

def nse_loss(y_true, y_pred):

    y_pred_mean = y_pred

    if len(y_pred_mean.shape) > 2:
        y_pred_mean = y_pred_mean[...,0,0]

    return K.sum((y_pred_mean-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)

def mse_loss(y_true, y_pred):

    y_pred_mean = y_pred

    if len(y_pred_mean.shape) > 2:
        y_pred_mean = y_pred_mean[...,0,0]

    return K.mean(K.square(y_pred_mean - y_true))


def monotonicity_loss(y_true, y_pred):
    """
    Monotonicity loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    # B,C,F
    mse_loss_mon_loss = 0.0
    y_pred_mean = y_pred

    if len(y_pred_mean.shape) > 2 and y_pred.shape[1] > 1:
        y_pred_mean = y_pred_mean[...,0]
        y_pred_permute = tf.transpose(y_pred_mean, perm=[1, 0])
        mse_loss_mon_loss = activations.relu(
            y_pred_permute[0, :] - y_pred_permute[1:, :]
        )

        mse_loss_mon_loss = tf.reduce_mean(K.square(mse_loss_mon_loss))

    return mse_loss_mon_loss

def negative_log_likelihood(y_true, y_pred):
    """
    Negative log likelihood loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    "B,P,F"

        # check shape of y_pred scale
    # tf.print(f"ypred", y_pred)

    if len(y_pred.shape) > 2:
        normal: tfp.distributions.Distribution = probabilistic_layer(
            y_pred[..., 0 , :]
        )
        log_likelihood = normal.log_prob(y_true)
    else:
        normal: tfp.distributions.Distribution = probabilistic_layer(
            y_pred
        )
        log_likelihood = normal.log_prob(y_true)

    neg_ll_value = -tf.reduce_mean(log_likelihood)

    # tf.print(f"neg_ll_value", neg_ll_value)

    return neg_ll_value


def combined_loss(y_true, y_pred):
    """
    Combined loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    mse = mse_loss(y_true, y_pred)
    monotonicity = monotonicity_loss(y_true, y_pred)
    neg_ll = negative_log_likelihood(y_true, y_pred)

    return mse + monotonicity + neg_ll


def errors(true_vals, calc_vals):
    pearsonr_err = np.corrcoef(true_vals, calc_vals)[0,1]
    rmse_err = np.sqrt(((true_vals - calc_vals)**2).sum())
    norm_rmse_err = np.sqrt(
        (((true_vals - calc_vals)/true_vals)**2
         ).sum())
    nse_err = 1 - (
        ((true_vals - calc_vals)**2).sum()
        / ((true_vals - true_vals.mean())**2).sum()
        )
    r = np.corrcoef(true_vals, calc_vals)[1,0]
    α = np.var(calc_vals) / np.var(true_vals)
    β = np.mean(calc_vals) / np.mean(true_vals)
    kge_err = 1 - np.sqrt((1 - r)**2 + (1 - α)**2 + (1 - β)**2)
    return dict(
        pearsonr = pearsonr_err,
        r_square = pearsonr_err**2,
        rmse = rmse_err,
        norm_rmse = norm_rmse_err,
        nse = nse_err,
        kge = kge_err,
    )
