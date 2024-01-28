# |%%--%%| <vncgIZ2R8t|VH9sm5cQh6>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# |%%--%%| <VH9sm5cQh6|jQU6Hvj1a8>

df = pd.read_csv(
    "D:\\testfolder\\test_exercise\\output_data\\sold_flats_cat_num.csv", delimiter=","
)

# |%%--%%| <jQU6Hvj1a8|eAwUQ6Y18x>

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

# |%%--%%| <eAwUQ6Y18x|kxKPvU7fbM>

df_num = df[num_columns].copy()

df_num.info()

X, y = (
    df_num.drop(columns=["price", "floors_cnt", "levels_count"]).values,
    df_num["sold_price"].values,
)

features_names = df_num.drop(columns=["sold_price"]).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = MinMaxScaler()
scaler.fit_transform(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pd.DataFrame(X_train).tail()

# |%%--%%| <kxKPvU7fbM|7k7m9HsiwW>


def forward(weights, inputs):
    return inputs.dot(weights.T)


def loss_func(predicts, labels):
    return np.square(predicts - labels)


# |%%--%%| <7k7m9HsiwW|kvASLMdfLv>

weights = np.random.randn(X.shape[1])
print(weights)
yhat = forward(weights, X_train[0])
loss = np.sqrt(loss_func(yhat, y[0]))
print(yhat, y[0], loss)

# |%%--%%| <kvASLMdfLv|td93smrzfI>


decline = np.linspace(start=0.5, stop=1.5, num=11)
yhat = decline * y[0]
loss = loss_func(yhat, y[0])
plt.plot(decline, loss, "-o")
# |%%--%%| <td93smrzfI|Z3AHHxO6vy>


def grad_loss(predicts, labels, inputs):
    return 2 * (predicts - labels) * inputs / inputs.size


# |%%--%%| <Z3AHHxO6vy|UHLbBn7chm>

weights = np.random.randn(X.shape[1])
yhat = forward(weights, X_train[0])
print(weights)
grad = grad_loss(yhat, y[0], X[0])
print(grad)

# |%%--%%| <UHLbBn7chm|im4dQ8W9Jr>


def update_weights(grad, weights, lerning_rate):
    return weights - lerning_rate * grad


# |%%--%%| <im4dQ8W9Jr|8mK0592NnA>

lerning_rate = 0.01
weights = update_weights(grad, weights, lerning_rate)
print(weights)

# |%%--%%| <8mK0592NnA|cKJ0Ae8O52>


def weights_init(weights, random_state=42):
    if np.ndim(weights) < 1:
        weights = np.zeros(weights)

    np.random.seed(random_state)
    return np.random.randn(*weights.shape) / np.sqrt(weights.size)


# |%%--%%| <cKJ0Ae8O52|xdlnOIrnyx>
weights = weights_init(X_train.shape[1], random_state=42)
weights


# |%%--%%| <xdlnOIrnyx|efFZkNZ9Qz>


def fit(X, y, weights, lr, epochs=30):
    cost = np.zeros(epochs)
    for i in range(epochs):
        grad = np.zeros(weights.shape)
        loss = 0

        for m in range(X.shape[0]):
            yhat = forward(weights, X[m, :])
            grad += grad_loss(yhat, y[m], X[m, :])
            loss += loss_func(yhat, y[m])

        weights = update_weights(grad / X.shape[0], weights, lr)
        cost[i] = loss / X.shape[0]

    return weights, cost


# |%%--%%| <efFZkNZ9Qz|co6XnyWOuJ>

weights = weights_init(X_train.shape[1], random_state=42)

weights, cost = fit(X_train, y_train, weights, lr=0.9, epochs=100)

plt.plot(cost, "-*")

# |%%--%%| <co6XnyWOuJ|ELdXJiIMsv>


def predict(weights, inputs):
    yhat = np.zeros(inputs.shape[0])

    for m in range(inputs.shape[0]):
        yhat[m] = inputs[m, :].dot(weights.T)

    return yhat


# |%%--%%| <ELdXJiIMsv|alxEIVEuz5>

yhat = predict(weights, X_test)
plt.plot(y_test, label="original")
plt.plot(yhat, label="predicted")

# |%%--%%| <alxEIVEuz5|3fxexagQNy>

plt.scatter(y_test, yhat, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r")
plt.xlabel("original")
plt.ylabel("predicted")

# |%%--%%| <3fxexagQNy|3brGHs1Yab>


def r2_score(weights, inputs, labels):
    predicts = predict(weights, inputs)
    return 1 - np.sum(np.square(labels - predicts)) / np.sum(
        np.square(labels - np.mean(labels))
    )


# |%%--%%| <3brGHs1Yab|ld1GTMeYqa>

r2_score(weights, X_test, y_test)

# |%%--%%| <ld1GTMeYqa|C7Ef4GCzJ8>

BATCH = 5000


def fit_SGD(X, y, weights, lr, epochs=30, batch_size=BATCH, random_state=42):
    np.random.seed(random_state)

    cost = np.zeros(epochs)
    for i in range(epochs):
        grad = np.zeros(weights.shape)
        loss = 0

        idx_batch = np.random.randint(0, X.shape[0], batch_size)
        x_batch = np.take(X, idx_batch, axis=0)
        y_batch = np.take(y, idx_batch)

        for m in range(batch_size):
            yhat = forward(weights, x_batch[m, :])
            grad += grad_loss(yhat, y_batch[m], x_batch[m, :])
            loss += loss_func(yhat, y_batch[m])

        weights = update_weights(grad / batch_size, weights, lr)
        cost[i] = loss / batch_size

    return weights, cost


# |%%--%%| <C7Ef4GCzJ8|tJ0Q4sOZFt>

weights = weights_init(X_train.shape[1], random_state=42)

weights, cost = fit_SGD(X_train, y_train, weights, lr=0.7, epochs=300)

plt.plot(cost, "-*")

print(r2_score(weights, X_test, y_test))

# |%%--%%| <tJ0Q4sOZFt|UtRK0qWU9V>


class LinearRegression:
    def __init__(
        self,
        learning_rate=0.5,
        epochs=100,
        weights=None,
        bias=None,
        batch_size=1000,
        random_state=42,
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = weights
        self.bias = bias
        self.seed = random_state
        self.batch_size = batch_size
        self.cost = np.zeros(epochs)

        # если веса и смещения заданы
        if not (self.weights is None) and (self.bias):
            if self.weights.size == X.shape[1]:
                # совмещаем в один массив, если мы этого не сделали
                self.weights = np.append(self.bias, self.weights)

    # ---------------------------------
    def forward(self, X):
        return self.weights.dot(X.T)

    # ---------------------------------
    def loss(self, yhat, y):
        return np.square(yhat - y).sum() / y.size

    # ---------------------------------
    def grad_step(self, yhat, y, X):
        return 2 * np.dot(X.T, (yhat - y)) / y.size

    # ---------------------------------
    def update(self):
        return self.weights - self.lr * self.grad

    # ---------------------------------
    def init(self, weights_size):
        np.random.seed(self.seed)
        return np.random.randn(weights_size) / np.sqrt(weights_size)

    # ---------------------------------
    def add_bias(self, X):
        return np.column_stack((np.ones(X.shape[0]), X))

    # ---------------------------------
    def predict(self, X):
        yhat = self.forward(self.add_bias(X))
        return yhat

    # ---------------------------------
    def score(self, X, y):
        yhat = self.predict(X)
        return 1 - np.sum(np.square(y - yhat)) / np.sum(np.square(y - np.mean(y)))

    # ---------------------------------
    def load_batch(self, X, y):
        idx_batch = np.random.randint(0, X.shape[0], self.batch_size)
        x_batch = np.take(X, idx_batch, axis=0)
        x_batch = self.add_bias(x_batch)
        y_batch = np.take(y, idx_batch)
        return x_batch, y_batch

    # ---------------------------------
    def fit(self, X, y):
        np.random.seed(self.seed)

        if self.weights is None:
            self.weights = self.init(X.shape[1])

        if self.bias is None:
            self.bias = self.init(1)

        if self.weights.size == X.shape[1]:
            # совмещаем в один массив, если мы этого не сделали
            self.weights = np.append(self.bias, self.weights)

        self.grad = np.zeros(self.weights.shape)
        self.cost = np.zeros(self.epochs)

        if self.batch_size is None:
            x_batch = self.add_bias(X)
            y_batch = y

        for i in range(self.epochs):
            if self.batch_size:
                x_batch, y_batch = self.load_batch(X, y)

            yhat = self.forward(x_batch)
            self.grad = self.grad_step(yhat, y_batch, x_batch)
            self.weights = self.update()
            self.cost[i] = self.loss(yhat, y_batch)

        self.bias = self.weights[0]

    # ---------------------------------
    def plot_cost(self, figsize=(12, 6)):
        plt.figure(figsize=figsize)
        plt.plot(self.cost, "-*")
        plt.show()

    # ---------------------------------
    def get_w_and_b(self):
        return (self.weights[1:], self.bias)


# |%%--%%| <UtRK0qWU9V|AlqzjFZk51>
r"""°°°
#Test if we overdid model
°°°"""
# |%%--%%| <AlqzjFZk51|HfKiSbG8fA>

regr = LinearRegression(learning_rate=0.5, epochs=300, batch_size=3000)
regr.fit(X_train, y_train)

regr.plot_cost()

print(
    "train R2: %.4f; test R2: %.4f"
    % (regr.score(X_train, y_train), regr.score(X_test, y_test))
)

# |%%--%%| <HfKiSbG8fA|BNVImKgPLl>

w, b = regr.get_w_and_b()
plt.bar(x=range(w.size), height=w)

plt.xticks(range(13), features_names, rotation=45)
