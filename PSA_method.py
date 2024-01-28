# |%%--%%| <xNSC4t310i|PeaOCcJE2L>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# |%%--%%| <PeaOCcJE2L|Y4noTI0TuQ>


df = pd.read_csv(
    "D:\\testfolder\\test_exercise\\output_data\\sold_flats_cat_num.csv", delimiter=","
)


# |%%--%%| <Y4noTI0TuQ|uSsEhd7DBQ>

df


# |%%--%%| <uSsEhd7DBQ|3LvuD7FPL8>


cat_columns = [
    "city_id",
    "district_id",
    "street_id",
    "date_sold",
    "metro_station_id",
    "flat_on_floor",
    "builder_id",
    "type",
    "bathroom",
    "plate",
    "windows",
    "keep",
    "series_id",
    "wall_id",
    "balcon",
    "closed_yard",
    "date_sold",
]
num_columns = [
    "price",
    "sold_price",
    "floor_num",
    "floors_cnt",
    "rooms_cnt",
    "bedrooms_cnt",
    "building_year",
    "area_total",
    "area_live",
    "area_kitchen",
    "area_balcony",
    "levels_count",
    "bathrooms_cnt",
    "ceiling_height",
]


# |%%--%%| <3LvuD7FPL8|eu3155tSMu>


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.values = None
        self.mean = None

    # --------------------------------
    def fit(self, X):
        self.mean = np.mean(X, axis=0)

        # расчет матрицы ковариации
        cov_matrix = np.cov(X - self.mean, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[
            ::-1
        ]  # индексы сортировки по значениям собственных векторов

        # сортируем собственные вектора и значения
        self.components = eigenvectors[:, idx][:, : self.n_components]
        self.values = eigenvalues[idx]

        return self

    # --------------------------------
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)

    # --------------------------------
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # --------------------------------
    def inverse_transform(self, X_new):
        return np.dot(X_new, self.components.T) + self.mean

    # --------------------------------
    def score(self, X):
        SStot = np.sum(np.square(X - np.mean(X)))
        SSres = np.sum(np.square(X - self.inverse_transform(self.fit_transform(X))))
        return 1 - SSres / SStot

    # --------------------------------
    def plot_eigvalues(self, figsize=(12, 4)):
        plt.figure(figsize=figsize)
        plt.plot(self.values, "-o", label="all eigvalues")
        plt.plot(self.values[: self.n_components], "-o", label="eigen subspace")
        plt.title("eigenvalues")
        plt.legend()
        plt.show()


# |%%--%%| <eu3155tSMu|FT0wb594Mr>

df_num = df[num_columns].copy()
scaler = MinMaxScaler()
X = scaler.fit_transform(df_num)

labels = df.sold_price.values

pca = PCA(n_components=2)
pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=labels, alpha=0.7)
plt.show()

print("eigenvalues", pca.values)
pca.plot_eigvalues()

print("score:", pca.score(X))

X2 = pca.inverse_transform(X_new)
plt.figure(figsize=(12, 4))
plt.scatter(X[:, 1], X2[:, 1], alpha=0.7)
plt.plot([X[:, 1].min(), X[:, 1].max()], [X[:, 1].min(), X[:, 1].max()], "r")
plt.xlabel("original")
plt.ylabel("restored")

# |%%--%%| <FT0wb594Mr|rjOY3rA6A3>

fig = plt.figure(figsize=(10, 10))
W = pca.components.T
pca_names = ["pca-" + str(x + 1) for x in range(2)]
plt.matshow(W.astype(float), cmap="bwr", vmin=-1, vmax=1, fignum=1)
for (i, j), z in np.ndenumerate(W):
    plt.text(
        j,
        i,
        "{:0.2f}".format(z),
        ha="center",
        va="center",
        color="k",
        fontsize="xx-large",
    )
plt.xticks(np.arange(0, W.shape[1]), df[num_columns].columns, rotation=45)
plt.yticks(np.arange(0, W.shape[0]), pca_names)
plt.colorbar()
