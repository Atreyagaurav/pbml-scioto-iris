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
from datetime import datetime
import matplotlib

# this library is not upto date and couldn't use monotonicity using this
# from airt.keras.layers import MonoDense

import matplotlib.pyplot as plt
from helper_funcs import *

HERE = os.path.dirname(os.path.abspath(__file__))

precip_fields = ['group_0','group_1','group_2','group_3']
endogenous_fields = ["streamflow"]

df_org = pd.read_csv(os.path.join(HERE, "../data/combined-all.csv"), index_col="datetime", parse_dates=True)
df = df_org.copy(deep=True)

trend = df[endogenous_fields].rolling(pd.Timedelta(hours=24), center=True).mean()

new_columns = {}

for field in precip_fields:
    new_columns[f'{field}_3h'] = df[field].rolling(pd.Timedelta(hours=3)).sum()
    new_columns[f'{field}_6h'] = df[field].rolling(pd.Timedelta(hours=6)).sum()
    new_columns[f'{field}_9h'] = df[field].rolling(pd.Timedelta(hours=9)).sum()
    new_columns[f'{field}_12h'] = df[field].rolling(pd.Timedelta(hours=12)).sum()
    new_columns[f'{field}_24h'] = df[field].rolling(pd.Timedelta(hours=24)).sum()
    new_columns[f'{field}_48h'] = df[field].rolling(pd.Timedelta(hours=48)).sum()
    new_columns[f'{field}_72h'] = df[field].rolling(pd.Timedelta(hours=72)).sum()
    new_columns[f'{field}_14d'] = df[field].rolling(pd.Timedelta(days=14)).sum()
    new_columns[f'{field}_30d'] = df[field].rolling(pd.Timedelta(days=30)).sum()
    new_columns[f'{field}_60d'] = df[field].rolling(pd.Timedelta(days=60)).sum()

# Add all new columns to the DataFrame at once
df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

# Update precip_fields
precip_fields.extend(new_columns.keys())
exogenous_fields = [*precip_fields]

def add_seasonality_signals(
        data: pd.DataFrame,
        seasonality_signals: List[str],
        add_sine: bool = True,
        add_cosine: bool = True
):
    """
    Add seasonality signals to the data
    :return:
    """
    d = data.copy()

    for signal in seasonality_signals:
        if signal == 'hour_of_day':
            hour_of_day = d.index.hour + d.index.minute / 60 + d.index.second / 3600
            d['hour_of_day'] = hour_of_day
            if add_sine:
                d['hour_of_day_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
            if add_cosine:
                d['hour_of_day_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
        elif signal == 'day_of_week':
            hour_of_day = d.index.hour + d.index.minute / 60 + d.index.second / 3600
            d['day_of_week'] = d.index.dayofweek + hour_of_day / 24
            if add_sine:
                d['day_of_week_sin'] = np.sin(2 * np.pi * d['day_of_week'] / 7)
            if add_cosine:
                d['day_of_week_cos'] = np.cos(2 * np.pi * d['day_of_week'] / 7)
        elif signal == 'day_of_year':
            hour_of_day = d.index.hour + d.index.minute / 60 + d.index.second / 3600
            day_of_year = d.index.dayofyear + hour_of_day / 24
            d['day_of_year'] = day_of_year
            if add_sine:
                d['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            if add_cosine:
                d['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    return d

df = add_seasonality_signals(df, seasonality_signals =['day_of_year'])
df.dropna(inplace=True)

exogenous_fields.append('day_of_year_sin')
exogenous_fields.append('day_of_year_cos')

x = df[exogenous_fields]
y = df[endogenous_fields]

from sklearn import set_config
set_config(transform_output="pandas")

scalar_x = MinMaxScaler().fit(x)
scalar_y = MinMaxScaler().fit(y)

x_scaled = scalar_x.transform(x)
y_scaled = scalar_y.transform(y)

# X = df.to_numpy()
# y = s.loc[df.index].values
# N = X.shape[1]
#
# X.shape, y.shape
dropout = 0.3

# random split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.3, shuffle=False)
x_train_a, x_valid, y_train_a, y_valid = train_test_split(x_train, y_train, test_size=0.3, shuffle=False)

precip_perturbations = np.arange(1.0, 2.0+0.05, 0.05)
x_train_a= np.repeat(x_train_a.values[:, np.newaxis, :], len(precip_perturbations), axis=1)




# perturb the rainfall fields
for i, perturbation in enumerate(precip_perturbations):
    x_train_a[:, i, :len(precip_fields)] *= perturbation

model = models.Sequential([
    layers.Input(shape=(None, len(exogenous_fields))),
    layers.Dense(64),
    layers.PReLU(shared_axes=[-1]),
    layers.Dropout(dropout),
    layers.Dense(128),
    layers.PReLU(shared_axes=[-1]),
    layers.Dropout(dropout),
    layers.Dense(32),
    layers.PReLU(shared_axes=[-1]),
    layers.Dropout(dropout),
    layers.Dense(2, activation="softplus"),
    ]
)



model.compile(
    optimizer="adam",
    metrics=[nse_loss, mse_loss, monotonicity_loss, negative_log_likelihood],
    loss=combined_loss,
)

# model._train_counter = 0
# model._test_counter = 0
# model._is_graph_network = False
model.summary()


batch_size = 32
epochs = 1000

hist = model.fit(
    x_train_a, y_train_a,
    batch_size=batch_size,
    epochs=epochs,
    validation_data = (x_valid.values[:, np.newaxis, ...], y_valid),
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
        callbacks.CSVLogger(
            os.path.join(HERE, "training.log"), append=True
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(HERE, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")),
            update_freq="batch"
        ),
        callbacks.ModelCheckpoint(
            os.path.join(HERE, "model.h5"), save_best_only=True
        ),
        callbacks.TerminateOnNaN(),
    ],

)

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

y_train_pred = model.predict(x_train.values[:, np.newaxis, ...])
y_test_pred = model.predict(x_test.values[:, np.newaxis, ...])

y_train_pred = tfp.distributions.Normal(loc=y_train_pred[..., 0, 0], scale=y_train_pred[..., 0 , 1])
y_test_pred = tfp.distributions.Normal(loc=y_test_pred[..., 0, 0], scale=y_test_pred[..., 0 , 1])

y_train_pred_scaled = scalar_y.inverse_transform(y_train_pred.mean().numpy()[..., np.newaxis])
y_test_pred_scaled = scalar_y.inverse_transform(y_test_pred.mean().numpy()[..., np.newaxis])


concat_all_y = np.concatenate([y_train, y_train_pred_scaled, y_test_pred_scaled, y_test])

min_y = np.min(concat_all_y)
max_y = np.max(concat_all_y)
plt.scatter(y_train, y_train_pred_scaled, label="train", s=0.2)
plt.scatter(y_test, y_test_pred_scaled, label="test", s=0.2)
plt.plot([min_y, max_y], [min_y, max_y], label="1:1", c="red")
plt.legend()
plt.xlabel("Observed Flow (cfs)")
plt.ylabel("Simulated Flow (cfs)")
matplotlib.use('TkAgg')
plt.show()

obs = pd.concat([
    pd.Series(y_train_val, index=traindt),
    pd.Series(y_test_val, index=testdt)
])
sim = pd.concat([
    pd.Series(y_train_pred_val, index=traindt),
    pd.Series(y_test_pred_val, index=testdt)
])

# obs = obs + trend[obs.index]
# sim = sim + trend[obs.index]

AGG_DAILY = False
# if we want to see aggregates
if AGG_DAILY:
    obs = obs.resample("1d").mean().dropna()
    sim = sim.resample("1d").mean().dropna()

# result = pd.DataFrame(dict(observed=obs, simulated=sim))


# def category(dt):
#     if dt < split2dt:
#         return "train"
#     elif dt < splitdt:
#         return "test"
#     return "validation"


# result.loc[:, "category"] = result.index.map(category)
# result.loc[:, "trend"] = trend
# result.to_csv(f"pbml-results{'-daily' if AGG_DAILY else ''}.csv")

# sns.lineplot(sf)
# sns.scatterplot(result, x = "datetime", y="observed", hue="category")
fig, ax1 = plt.subplots() # initializes figure and plots
ax2 = ax1.twinx()

sns.lineplot(result, x="datetime", y="observed", hue="category", ax=ax1)
sns.lineplot(trend.reset_index(), x="datetime", y="streamflow", ax=ax1, dashes=[2, 2], label="trend", color="green")
sns.scatterplot(result, x="datetime", y="simulated", color="red", ax=ax1)
mean_precip = df_org.iloc[:, :-1].mean(axis=1)
mean_precip.name = "precip"
# sns.barplot(pd.DataFrame(mean_precip), x="datetime", y="precip")
plt.show()


sea = sns.FacetGrid(result, col="category")
sea.map(sns.scatterplot, "observed", "simulated", alpha=.8, color=None)
plt.show()

trainx = result.observed.loc[result.category == "train"]
trainy = result.simulated.loc[result.category == "train"]
testx = result.observed.loc[result.category == "test"]
testy = result.simulated.loc[result.category == "test"]
validx = result.observed.loc[result.category == "validation"]
validy = result.simulated.loc[result.category == "validation"]

high_threshold = result.observed.quantile(0.99)

# pd.DataFrame(dict(
#     train=errors(trainx, trainy),
#     test=errors(testx, testy),
#     valid=errors(validx, validy),
#     train_high=errors(trainx.loc[trainx > high_threshold], trainy.loc[trainx > high_threshold]),
#     test_high=errors(testx.loc[testx > high_threshold], testy.loc[testx > high_threshold]),
#     valid_high=errors(validx.loc[validx > high_threshold], validy.loc[validx > high_threshold]),
# )).T
