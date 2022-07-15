from sensor_analysis import read_data, manipulate_x
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd


class DataAssistant:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.encoded_y = []
        self.one_hot = []

    def manipulate_y(self):
        '''
        Convert the classes from strings to values by using the sckit-learn mapper
        '''
        le = LabelEncoder()
        le.fit(self.data_y)
        self.data_y = le.transform(self.data_y)
        self.data_y = pd.DataFrame(self.data_y, columns=['target'])

        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)

    def make_float(self):
        self.data_x.astype('float32')

    def split(self, start, end):
        split_x = self.data_x[start:end].values
        split_y = self.data_y[start:end].values
        return split_x, split_y

    def onehot(self):
        one_hot = OneHotEncoder()
        one_hot.fit(self.data_y.reshape(-1, 1))
        self.data_y = one_hot.transform(self.data_y.reshape(-1, 1)).toarray()
        pass

    def scaling(self):
        scaler = MinMaxScaler().fit(self.data_x)
        self.data_x = scaler.transform(self.data_x)

    def reshape_for_LSTM(self):
        timestemps = 1
        samples = int(np.floor(self.data_x.shape[0]/timestemps))
        self.data_x = self.data_x.reshape((samples, timestemps, self.data_x.shape[1]))  #samples, timesteps, sensors


class TimeseriesAssistant:
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
        agg.iloc[0, 0:54] = agg.iloc[1, 0:54]
        if drop_nan:
            agg.dropna(inplace=True, axis=0)

        self.data = self.clean_series_to_supervised(agg)
        return self.data

    def clean_series_to_supervised(self, shifted_data):
        to_remove_list = ['sensor' + str(n) + '(t)' for n in range(1, len(self.data.columns) + 1)]
        # keep only the last column which is the target
        data_y = shifted_data.iloc[:, -1]
        # remove all the sensor values at t including the target at t
        data_x = shifted_data.drop(to_remove_list, axis=1)
        # drop the last column which is the target but named as sensor46
        data_x.drop(data_x.columns[len(data_x.columns) - 1], axis=1, inplace=True)
        data = pd.concat([data_x, data_y], axis=1)
        # asterisk means unpacking the list of data.columns except the last item which will be renamed to 'machine_status'
        data.columns = [*data.columns[:-1], 'machine_status']
        return data


if __name__ == '__main__':
    # Load data
    data, sensor_names = read_data("pump_sensor.csv", 1)

    # preprocess data
    data = manipulate_x(data, print_plot=False)
    # remove timestamps column
    sensor_names = data.keys()[2:-1]
    # create a dataframe with sensors and target only!
    data = data[sensor_names.insert(len(sensor_names), 'machine_status')]

    # create windowed data
    FUTURE = 1
    time_series = TimeseriesAssistant(data, FUTURE).series_to_supervised()
    # these are the sensor names including the time shift, in our case the t-1 since future is 1
    sensor_names_shift = time_series.keys()[:-1]

    # preprocess dataset
    the_data = DataAssistant(time_series[sensor_names_shift], time_series['machine_status'])
    # map target's text to indexes
    the_data.manipulate_y()
    the_data.make_float()

    # split the data to three sets [train, val and test]
    train_x, train_y = the_data.split(0, 120000)
    test_x, test_y = the_data.split(120000, 140000)
    val_x, val_y = the_data.split(140000, len(the_data.data_x))

    # let's create three objects, one for each set
    training_data = DataAssistant(train_x, train_y)
    validation_data = DataAssistant(val_x, val_y)
    testing_data = DataAssistant(test_x, test_y)

    training_data.onehot()
    training_data.scaling()
    training_data.reshape_for_LSTM()

    validation_data.onehot()
    validation_data.scaling()
    validation_data.reshape_for_LSTM()

    testing_data.onehot()
    testing_data.scaling()
    testing_data.reshape_for_LSTM()





