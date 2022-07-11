from sensor_analysis import read_data, manipulate_y, manipulate_x
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def splitting_and_shape(data_x, data_y):
    train_x = data_x[0:120000].values
    train_y = data_y[0:120000].values

    val_x = data_x[140000:].values
    val_y = data_y[140000:].values

    test_x = data_x[120000:140000].values
    test_y = data_y[120000:140000].values

    # train_x.astype('float32')
    # val_x.astype('float32')
    # test_x.astype('float32')
    return train_x, train_y, val_x, val_y, test_x, test_y


def scaling(data):
    scaler = MinMaxScaler().fit(data)
    scaled_features = scaler.transform(data)
    return scaled_features


class TimeSeries:
    def __init__(self, data, n_in=1, n_out=1):
        self.data = data
        self.n_in = n_in
        self.n_out = n_out

    def series_to_supervised(self, drop_nan=True):
        n_vars = 1 if type(self.data) is list else self.data.shape[1]
        df = pd.DataFrame(self.data)
        cols, names = list(), list()
        # input sequence (t-n, ... , t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('sensor%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        for i in range(0, self.n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('sensor%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('sensor%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if drop_nan:
            agg.dropna(inplace=True)

        self.data = self.clean_series_to_supervised(agg)
        return self.data

    def clean_series_to_supervised(self, shifted_data):
        to_remove_list = ['sensor' + str(n) + '(t)' for n in range(1, len(values.columns) + 1)]
        data_y = data_window.iloc[:, -1]
        data_x = data_window.drop(to_remove_list, axis=1)
        data_x.drop(data_x.columns[len(data_x.columns) - 1], axis=1, inplace=True)
        data = pd.concat([data_x, data_y], axis=1)
        data.columns = [*data.columns[-1], 'machine_status']
        return data


if __name__ == '__main__':
    # Load data
    data, sensor_names = read_data("pump_sensor.csv", 1)

    # preprocess data
    values = manipulate_x(data, print_plot=False)
    # remove timestamps column
    sensor_names = data.keys()[2:-1]
    # create a dataframe with sensors and target only!
    data = data[sensor_names.insert(len(sensor_names), 'machine_status')]

    # create windowed data


    # map targets to values
    encoded_y = manipulate_y(data)
    values = pd.concat([data[sensor_names], encoded_y], axis=1)

    # preprocess data


    # create windowed data
    future = 1
    data_window = series_to_supervised(values, n_in=future, n_out=1)

    # create train/test/val sets
    train_x, train_y, val_x, val_y, test_x, test_y = splitting_and_shape(data_x, data_y)

    scaled_train_x = scaling(train_x[1, :])
    pass
