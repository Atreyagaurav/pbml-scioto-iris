import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator

from plotly.subplots import make_subplots
import plotly.graph_objects as go


df = pd.read_csv("data/precip-and-sf.csv", index_col="datetime")
df.index = pd.to_datetime(df.index)

# some stations correlated with each other
sns.heatmap(df.corr())
plt.title("Hourly Correlation")
plt.show()

sns.heatmap(df.groupby(df.index.year).mean().corr())
plt.title("Annual Aggregated Correlation")
plt.show()
# anything negative here is useless, Alton-Darby should be removed.


df.groupby(df.index.year).mean().corr().loc["streamflow"]
df.pop("Alton-Darby")
df.pop('Headley_Park')

# take out streamflow
sf = df.pop("streamflow")

# PCA + Kmeans
pca = PCA(n_components=5)

pca.fit(df.dropna().values)
comps = pca.components_[0:10, :]

results = []
for i in range(1, 11):
    km = KMeans(i, max_iter=1000, n_init=100)
    km.fit(np.transpose(comps))
    results.append(km.inertia_)
kn = KneeLocator(
    range(1, 11),
    results,
    curve='convex',
    direction='decreasing', interp_method='interp1d')

kn.plot_knee()
plt.show()
kn.knee # 5

km = KMeans(kn.knee, max_iter=1000, n_init=100)
km.fit(np.transpose(comps))
km.labels_

# plt.scatter(comps[0], comps[2], c=km.labels_)
# plt.show()

gages = gpd.read_file("data/gis/gages.gpkg", layer="storms2")

gages.index = gages.name.map(lambda n: n.replace(".", "").replace(" ", "_"))
gages.index.name = "ind"

selected = list(df.columns)
gages.loc[selected, "cluster"] = km.labels_
gages.to_file("data/gis/clusters.gpkg")

names = pd.Series(gages.index).groupby(gages.cluster.to_list()).agg(list)
names.index = names.index.astype(int)
for c, n in names.items():
    print(c, n)

names.old = {
    0: ['Camp_Lazarus', 'Delaware', 'Mt_Gilead_SP', 'Pharisburg'],
    1: ['Blendon_Woods_Park', 'Highbanks_Metro_Park', 'Walnut_St', 'Westerville_Water', 'Woodward_Park'],
    2: ['Dexter_Falls_Park', 'Frazell_Road', 'Johnathan_Alder', 'Kenlawn_Park', 'Ohio_EMA', 'Raymond_Mem_GC'],
    3: ['Big_Walnut_Park', 'Jackson_Twp_Admin', 'Jefferson_Twp_Sewer', 'Pataskala', 'Three_Creeks_Park'],
    4: ['ASCS', 'Killdeer_Plains_SP', 'Larue'],
}


dfnew = pd.DataFrame(index=df.index, columns=[f"group_{i}" for i in names.index])

for col, g in zip(dfnew.columns, names):
    dfnew.loc[:, col] = (df.loc[:, g] * 25.4).mean(axis=1)  # inch -> mm


# how much gaps there are
len(dfnew.index)
(dfnew.isna().sum(axis=1) == 0).sum()
(dfnew.isna().sum(axis=1) == 0).sum() / len(dfnew.index)
dfnew.isna().sum()

# group4 has too many gaps, and are at the edge of the basin, so let's remove them
dfnew.pop("group_4")
# with this we went from 85% data to 97% of data not NA.

# sns.heatmap(dfnew.corr())
# plt.title("Hourly Correlation")
# plt.show()

fillval = dfnew.mean(axis=1)
for col in dfnew.columns:
    ind = dfnew.loc[:, col].isna()
    dfnew.loc[ind, col] = fillval.loc[ind]

fig = make_subplots(rows=5, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)
for i, col in enumerate(dfnew.columns):
    fig.add_trace(go.Scatter(x=dfnew.index, y=dfnew.loc[:, col], name=col), row=i+1, col=1)
fig.add_trace(go.Scatter(x=sf.index, y=sf, name="streamflow"), row=5, col=1)
fig.write_html("plots/precip-groups.html")

# There seems to be a gap before the 2005 flood, which is not
# good. All stations are missing precip data at that time.


dfnew.isna().sum()

# blocks analysis
na = dfnew.group_1.isna()
blocks = (na != na.shift(1)).cumsum()
blocks.loc[dfnew.group_1.isna()]

gaps = blocks.value_counts()[na.groupby(blocks).first()]
gaps.index.name = "block"
dates = pd.Series(blocks.index).groupby(blocks.to_list()).first()

plt.scatter(dates.loc[gaps.index], gaps)
plt.show()

dm_temp = pd.read_csv("data/temperature.csv")

temp = dm_temp.Temperature
temp.index = pd.to_datetime(dm_temp.date)

dfnew.join(sf, how="outer").join(temp, how="left").to_csv("data/combined-all.csv")
