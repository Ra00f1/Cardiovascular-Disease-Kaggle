import sklearn.linear_model as lm
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
import tensorflow as tf

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


class MyModel:
    def __init__(self, input_shape):
        super().__init__()
        # Create the model layers
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-07,
                                             amsgrad=False)

        self.model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

    def train(self, X_train, y_train, epochs=1000, batch_size=32):

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return history

    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

    def predict(self, X_new):
        return self.model.predict(X_new)

    def load_best_model(self):
        # Loads the best model saved during training based on lowest validation MAE.
        self.model = tf.keras.models.load_model('best_model.h5')


def load_data(file_path):
    # columns are divided by';' in the dataset
    data = pd.read_csv(file_path, index_col=0, low_memory=False, sep=';')
    return data


# Changing the data type of the columns to category and then to numerical values for the model
def data_categories(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

    return data


def Process_Data(data):

    scaler = StandardScaler()

    # Scaling height, weight, and age columns, ap_hi, ap_lo
    data['height'] = scaler.fit_transform(data['height'].values.reshape(-1, 1))
    data['weight'] = scaler.fit_transform(data['weight'].values.reshape(-1, 1))
    data['age'] = scaler.fit_transform(data['age'].values.reshape(-1, 1))
    data['ap_hi'] = scaler.fit_transform(data['ap_hi'].values.reshape(-1, 1))
    data['ap_lo'] = scaler.fit_transform(data['ap_lo'].values.reshape(-1, 1))

    # Splitting the data into features and labels
    X_train = data.iloc[:, :-1]
    # Select the last column as the label
    y_train = data.iloc[:, -1]

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    return X_train, y_train


def Visualize_Data(data):
    plt.figure(figsize=(7, 6))
    sns.distplot(data['SalePrice'], color='g', bins=100)
    plt.show()


# Data Summary Function
def Data_Summary(data):
    print("Data Summary")
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    print(data.iloc[:, -1].describe())
    # Visualize the last column
    temp_data = data.drop(data.index[0])
    plt.figure(figsize=(7, 6))
    plt.bar(temp_data.iloc[:, -1].unique(), temp_data.iloc[:, -1].value_counts())
    plt.show(block=True)

    print("---------------------------------------------------------------------------------")


def Machine_Learning(X_train, y_train):
    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        )

    log_reg = lm.LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_score = log_reg.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: ", log_reg_score)

    lin_reg = lm.LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)
    print("Linear Regression Testing Accuracy: ", lin_reg_score)

    Sgd_reg = lm.SGDRegressor()
    Sgd_reg.fit(X_train, y_train)
    Sgd_reg_score = Sgd_reg.score(X_test, y_test)
    print("SGD Testing Accuracy: ", Sgd_reg_score)

    Ridge_reg = lm.Ridge()
    Ridge_reg.fit(X_train, y_train)
    Ridge_reg_score = Ridge_reg.score(X_test, y_test)
    print("Ridge Testing Accuracy: ", Ridge_reg_score)

    Lasso_reg = lm.Lasso()
    Lasso_reg.fit(X_train, y_train)
    Lasso_reg_score = Lasso_reg.score(X_test, y_test)
    print("Lasso Testing Accuracy: ", Lasso_reg_score)

    Elastic_reg = lm.ElasticNet()
    Elastic_reg.fit(X_train, y_train)
    Elastic_reg_score = Elastic_reg.score(X_test, y_test)
    print("Elastic Testing Accuracy: ", Elastic_reg_score)

    Huber_reg = lm.HuberRegressor()
    Huber_reg.fit(X_train, y_train)
    Huber_reg_score = Huber_reg.score(X_test, y_test)
    print("Huber Testing Accuracy: ", Huber_reg_score)

    Ransac_reg = lm.RANSACRegressor()
    Ransac_reg.fit(X_train, y_train)
    Ransac_reg_score = Ransac_reg.score(X_test, y_test)
    print("Ransac Testing Accuracy: ", Ransac_reg_score)

    Theil_reg = lm.TheilSenRegressor()
    Theil_reg.fit(X_train, y_train)
    Theil_reg_score = Theil_reg.score(X_test, y_test)
    print("Theil Testing Accuracy: ", Theil_reg_score)

    models = [log_reg, lin_reg, Sgd_reg, Ridge_reg, Lasso_reg, Elastic_reg, Huber_reg, Ransac_reg, Theil_reg]
    scores = [log_reg_score, lin_reg_score, Sgd_reg_score, Ridge_reg_score, Lasso_reg_score, Elastic_reg_score, Huber_reg_score, Ransac_reg_score, Theil_reg_score]

    return models, scores

#
# return models, scores


if __name__ == '__main__':
    data = load_data("Data/cardio_train.csv")
    # Data_Summary(data)

    X_train, y_train = Process_Data(data)

    models, scores = Machine_Learning(X_train, y_train)
#
    # print("Models: ", models)
    # print("Scores: ", scores)
    # best_model = models[scores.index(max(scores))]
    # print("Best Model: ", best_model)
#
    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        )
    model = MyModel(input_shape=(X_train.shape[1],))  # Pass the input shape
    model.train(X_train, y_train, epochs=100, batch_size=32)
    test_predictions = model.predict(X_test)
    print("Test Predictions: ", test_predictions)

    # compare the predictions with the actual values
    print("Mean Squared Error: ", mean_squared_error(y_test, test_predictions))