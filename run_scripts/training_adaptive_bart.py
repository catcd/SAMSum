import logging
import os
import pickle

from keras.layers import Dense
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from utils.data_loader import DataLoader

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)


def prepare_data():
    print('Preparing data')
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    X, y = [], []
    for question_id, data in data_loader.docs.items():
        lengths, sentences = [], []
        for answer_id, answer in data['answers'].items():
            tokens_count = 0
            for sentence_id, sentence in answer['sentences'].items():
                tokens_count += len(sentence['tokens'])
            sentences.append(len(answer['sentences']))
            lengths.append(tokens_count)
        X.append([sum(lengths), sum(lengths) / len(lengths), min(lengths), max(lengths),
                  sum(sentences), sum(sentences) / len(sentences), min(sentences), max(sentences),
                  len(data['answers'])])
        y.append(len(data['abstractive_summary'].split(' ')))
    return X, y


def test_linear_regression(X, y):
    print('Test linear regression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=58)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print('Predict:', y_pred)
    print('Actual:', y_test)
    print('Diff:', y_test - y_pred)
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))


def test_simple_mlp(X, y):
    print('Test MLP')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=58)

    model = Sequential()
    model.add(Dense(16, input_shape=(9,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Configure the model and start training
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=1)
    y_pred = model.predict(X_test)
    y_pred = [val for v in y_pred for val in v]

    print('Predict:', y_pred)
    print('Actual:', y_test)
    # print('Diff:', y_test - y_pred)
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))


def test_mlp_regressor(X, y):
    print('Test MLPRegression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=58)
    reg = MLPRegressor(random_state=58, max_iter=20000, verbose=1, n_iter_no_change=100000).fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print('Predict:', y_pred)
    print('Actual:', y_test)
    # print('Diff:', y_test - y_pred)
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))

    with open('./data/models/MLPRegressor.sav', 'wb') as file:
        pickle.dump(reg, file)


if __name__ == '__main__':
    X, y = prepare_data()
    # test_linear_regression(X, y)
    # test_simple_mlp(X, y)
    test_mlp_regressor(X, y)
