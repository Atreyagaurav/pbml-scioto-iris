import pandas as pd
import numpy as np
import seaborn as sns
import setuptools.dist
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K
import tensorflow as tf

# this library is not upto date and couldn't use monotonicity using this
# from airt.keras.layers import MonoDense

import matplotlib.pyplot as plt


df_org = pd.read_csv("data/combined-all.csv", index_col="datetime")
df_org.index = pd.to_datetime(df_org.index)

df = df_org.copy(deep=True)

sf_org = df.pop("streamflow")
sf = sf_org.copy(deep=True)

temp = pd.read_csv("data/temp.csv", index_col="date").iloc[:, 0]
temp.index = pd.to_datetime(temp.index)

trend = sf_org.rolling(24, center=False).mean()
# s2 = (sf - trend)
s2 = sf

temp_mean = np.mean(temp)
temp_std = np.std(temp)
temp_norm = lambda f: (f - temp_mean)/temp_std

y_mean = np.mean(s2)
y_std = np.std(s2)
s2_norm = lambda f: (f - y_mean)/y_std
s = s2.map(s2_norm)

rol03 = df.rolling(3).mean()
rol06 = df.rolling(6).mean()
rol12 = df.rolling(12).mean()
rol24 = df.rolling(24).mean()
rol24007 = df.rolling(24*7).mean()

lag06 = rol06.shift(6)
lag0612 = rol06.shift(12)
lag12 = rol12.shift(12)

df = df.join(rol03, rsuffix="r03")
df = df.join(rol06, rsuffix="r06")
df = df.join(rol12, rsuffix="r12")
df = df.join(lag06, rsuffix="l06")
df = df.join(lag0612, rsuffix="l0612")
df = df.join(lag12, rsuffix="l12")
df = df.join(rol24, rsuffix="r24")
df = df.join(rol24007, rsuffix="r24007")

# scale precipitation
# df = df.apply(np.sqrt)
Prec_mean = np.mean(df)
Prec_std = np.std(df.dropna().values)
df = (df - Prec_mean)/Prec_std

sf_lag_24 = sf.shift(24).map(sf_norm)
sf_lag_24_06 = sf.shift(24).rolling(6).mean().map(sf_norm)
sf_lag_24_24 = sf.shift(24).rolling(24).mean().map(sf_norm)

df = df.join(sf_lag_24, rsuffix="sf_l24")
df = df.join(sf_lag_24_06, rsuffix="sf_l24_06")
df = df.join(sf_lag_24_24, rsuffix="sf_l24_24")

temp_rol_06 = temp.rolling(6).mean().map(temp_norm)
temp_rol_24 = temp.rolling(24).mean().map(temp_norm)
temp_rol_48_04 = temp.rolling(48*4).mean().map(temp_norm)

df = df.join(temp_rol_06, rsuffix="temp_r06")
df = df.join(temp_rol_24, rsuffix="temp_r24")
df = df.join(temp_rol_48_04, rsuffix="temp_r48_04")

# seasonality and day of time proxy
numdays = pd.Series(df.index.is_leap_year).map({True: 366, False: 365})
seasonality = (df.index.day_of_year / numdays) * np.pi * 2
timeofday = (df.index.hour + df.index.minute / 60) / 24 * np.pi * 2
proxies = pd.DataFrame(dict(
    season_y=seasonality.map(np.sin),
    season_x=seasonality.map(np.cos),
    time_y=timeofday.map(np.sin),
    time_x=timeofday.map(np.cos),
))
proxies.index = df.index

df = df.join(proxies)

nonaind = df.join(s2, rsuffix="sth").dropna().index
df = df.loc[nonaind, :]
sf = s2.loc[nonaind]

X = df.to_numpy()
y = s.loc[df.index].values
N = X.shape[1]

X.shape, y.shape

# random split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
split = X.shape[0] * 70 // 100
splitdt = nonaind[split]
# temporal split
X_train = X[:split,]
X_test = X[split:,]
y_train = y[:split]
y_test = y[split:]
traindt = nonaind[:split,]
testdt = nonaind[split:,]
# X_train = X[-split:,]
# X_test = X[:-split,]
# y_train = y[-split:]
# y_test = y[:-split]
# traindt = sel.index[sel][-split:,]
# testdt = sel.index[sel][:-split]


inp = layers.Input(shape=(N, ))
x = layers.Dense(
    128,
    activation='sigmoid')(inp)
#    monotonicity_indicator=[1 if col.endswith("r06") else 0 for col in df.columns]
x = layers.Dropout(0.4)(x)
x = layers.Dense(16, activation='linear')(x)
x = layers.Dropout(0.4)(x)

# using relu with negative slope to avoid problem with backpropagation
output = layers.Dense(1, activation=lambda x: relu(x, threshold=0.0, negative_slope=0.1))(x)

model = Model(inp, output)


def nseloss(y_true, y_pred):
  return K.sum((y_pred-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)


def mse_w_penalty(y_true, y_pred):
  mse = K.sum((y_pred-y_true)**2)
  true_rank = tf.argsort(y_true)
  pred_rank = tf.argsort(y_pred)
  # add penalty when there is large difference in rank/quantile of the
  # values predicted and observed; hope it helps with monotonicity;
  # doesn't seem to do much
  mse += 0.01 * tf.cast(K.sum((pred_rank-true_rank)**2), tf.float32)
  return mse


model.compile(optimizer="adam", metrics=['mse', nseloss, mse_w_penalty], loss=mse_w_penalty)
model.summary()


batch_size = 64
epochs = 10

split2 = X_train.shape[0] * 70 // 100
split2dt = nonaind[split2]
# temporal split
X_train_t = X_train[:split2,]
X_train_v = X_train[split2:,]
y_train_t = y_train[:split2]
y_train_v = y_train[split2:]

hist = model.fit(X_train_t, y_train_t, batch_size=batch_size, epochs=epochs, validation_data = (X_train_v, y_train_v))

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


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

y_train_val = (y_train * y_std + y_mean)
y_train_pred_val = (y_train_pred[:,0] * y_std + y_mean)
y_test_val = (y_test * y_std + y_mean)
y_test_pred_val = (y_test_pred[:,0] * y_std + y_mean)

y_train_pred_val[y_train_pred_val < 0] = 0
y_test_pred_val[y_test_pred_val < 0] = 0

y1 = y_train_val
y2 = y_train_pred_val
y3 = y_test_val
y4 = y_test_pred_val

pd.DataFrame(dict(
    train = errors(y1, y2),
    test = errors(y3, y4),
)).T

# y_train_pred[y_train_pred < 0] = 0
# y_test_pred[y_test_pred < 0] = 0
min_y = np.min(np.concatenate([y1, y2, y3, y4]))
max_y = np.max(np.concatenate([y1, y2, y3, y4]))
plt.scatter(y1, y2, label="train", s=0.2)
plt.scatter(y3, y4, label="test", s=0.2)
plt.plot([min_y, max_y], [min_y, max_y], label="1:1", c="red")
plt.legend()
plt.xlabel("Observed Flow (cfs)")
plt.ylabel("Simulated Flow (cfs)")

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

result = pd.DataFrame(dict(observed=obs, simulated=sim))


def category(dt):
    if dt < split2dt:
        return "train"
    elif dt < splitdt:
        return "test"
    return "validation"


result.loc[:, "category"] = result.index.map(category)
result.loc[:, "trend"] = trend
result.to_csv(f"pbml-results{'-daily' if AGG_DAILY else ''}.csv")

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

pd.DataFrame(dict(
    train=errors(trainx, trainy),
    test=errors(testx, testy),
    valid=errors(validx, validy),
    train_high=errors(trainx.loc[trainx > high_threshold], trainy.loc[trainx > high_threshold]),
    test_high=errors(testx.loc[testx > high_threshold], testy.loc[testx > high_threshold]),
    valid_high=errors(validx.loc[validx > high_threshold], validy.loc[validx > high_threshold]),
)).T
