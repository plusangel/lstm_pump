from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from models import model_setup_Fapi, plot_training, plot_signal_hat
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class DataPreparation:
    def __init__(self):
        self.sensors_names = None
        self.data = None

    def read_data(self, path) -> pd.DataFrame:
        '''
        Reads the data from a csv file
        The sensor data come in columns.
        :param path: The path of the csv file
        '''
        self.data = pd.read_csv(path)
        self.update_sensors_names()

    def update_sensors_names(self):
        self.sensors_names = self.data.keys()

    def manipulate_x(self):
        # Sensor_15 is completely empty
        self.data = self.data.drop(labels=['sensor_15'], axis=1)

        # lets merge sensor_50 and sensor_51 to come up with a good one
        # data['sensor_51'][110000:140000] = data['sensor_50'][110000:140000]
        self.data.iloc[110000:140000, self.data.columns.get_loc('sensor_51')] = self.data.iloc[110000:140000,
                                                                      self.data.columns.get_loc('sensor_50')]
        self.data = self.data.drop(labels=['sensor_50'], axis=1)

        self.plot_nans()
        # the sensors between 06–09 show most NaNs
        self.data = self.data.drop(labels=['sensor_06', 'sensor_07', 'sensor_08', 'sensor_09'], axis=1)
        self.data = self.data.fillna(method='pad', limit=30)
        self.data = self.data.dropna()


    def plot_nans(self, save=False):
        print((self.data.isna().sum()))
        self.plotting_stuff((self.data.isna().sum()[2:-1]), 'bar', 'fill_nan', saving=save)

    def plotting_stuff(self, data, plot_type, title, saving=False):
        fig = plt.figure()
        data.plot(kind=plot_type)
        plt.title(title)
        if saving == True:
            plt.savefig(title + '.png', format='png', dpi=300, transparent=True)
        fig.show()

    def remove_timestamps(self):
        # remove timestamps column
        self.update_sensors_names()
        self.sensors_names = experiment.sensors_names[1:-1]
        # create a dataframe with sensors and target only!
        self.data = self.data[self.sensors_names.insert(len(self.sensors_names), 'machine_status')]

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


class DataAssistant:

    def __init__(self):
        self.data_x = []
        self.data_y = []
        self.encoded_y = []
        self.one_hot = []

    def make_float(self):
        self.data_x.astype('float32')

    def split(self, start, end):
        split_x = self.data_x[start:end].values
        split_y = self.data_y[start:end].values
        return split_x, split_y

    def onehot(self):
        one_hot = OneHotEncoder()
        one_hot.fit(self.data_y.reshape(-1, 1))
        self.one_hot_y = one_hot.transform(self.data_y.reshape(-1, 1)).toarray()

    def scaling(self):
        scaler = MinMaxScaler().fit(self.data_x)
        self.data_x = scaler.transform(self.data_x)

    def reshape_for_LSTM(self):
        timestemps = 1
        samples = int(np.floor(self.data_x.shape[0] / timestemps))
        self.data_x = self.data_x.reshape((samples, timestemps, self.data_x.shape[1]))  # samples, timesteps, sensors


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
    experiment = DataPreparation()
    experiment.read_data('pump_sensor.csv')
    # print(experiment.get_sensors_names())

    # there is a column called 'Unnamed 0' including indexes and we need to drop it
    experiment.data.drop(experiment.data.filter(regex="Unnamed"), axis=1, inplace=True)
    # print(experiment.get_sensors_names())

    # preprocess data
    experiment.manipulate_x()

    # drop timestamps
    experiment.remove_timestamps()

    # create windowed data
    FUTURE = 1
    time_series = TimeseriesAssistant(experiment.data, FUTURE).series_to_supervised()

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

    EPOCH = 10
    BATCHSIZE = 32

    input_shape_x = training_data.data_x.shape
    print(input_shape_x)

    TRAIN = False

    # Train the model
    if TRAIN:
        model = model_setup_Fapi(input_shape_x)
        history = model.fit(training_data.data_x, [training_data.data_y, training_data.one_hot_y], epochs=EPOCH,
                            batch_size=32,
                            validation_data=(
                                validation_data.data_x, [validation_data.data_y, validation_data.one_hot_y]),
                            shuffle=False)

        plot_training([history.history['class_out_loss'], history.history['val_class_out_loss']], what='loss',
                      save=True,
                      name=('training_' + str(FUTURE)))
        plot_training([history.history['class_out_acc'], history.history['val_class_out_acc']], what='acc', save=True,
                      name=('training_' + str(FUTURE)))
        model.save('pump_LSTM_' + str(FUTURE))
    else:
        model = load_model('./pump_LSTM_1')

    # inference
    [yhat, yclass] = model.predict(testing_data.data_x)
    y_class = [np.argmax(yclass[i], axis=0) for i in range(len(yclass))]

    plot_signal_hat(y_class, testing_data.data_y, save=True, name='prediction_' + str(FUTURE))

    # calculate how much time in advance the model predicts a problem with pump
    # find the index of the first time that the pump broken
    first_broken_tested = np.where(testing_data.data_y == 0)[0]
    # find the index of the first time that the pump recovered (there is no broken signal)
    first_recover_predicted = y_class.index(2)
    difference_in_minutes = (first_broken_tested[0] - first_recover_predicted) / 60

    print(f"The model predicted that the pump will brake {difference_in_minutes}min in advanced using the testing data")
