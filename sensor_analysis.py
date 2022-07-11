import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple


def read_data(path: str, start_index: int, end_index: int = -1) -> Tuple[pd.DataFrame, pd.array]:
    '''
    Reads the data from a csv file
    The sensor data come in columns.
    The target data is the last column
    We select the region using the indexes
    It returns the data and the sensor names
    :param path: The path of the csv file
    :param start_index: The column where the sensors' data start
    :param end_index: The column where the data ends, it is usually the target
    :return data: the dataframe containing the x and y
    :return sensor_names: the array containing the labels with the sensors names
    '''
    data = pd.read_csv(path)
    sensor_names = data.keys()[start_index:end_index]
    return data, sensor_names


def explore(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''

    :param data:
    :return:
    '''
    print()
    print('Data overview: ')
    print(data.shape)
    print("keys :")
    print(data.keys())
    print()
    print('status options: ')
    print(data['machine_status'].unique())
    print(data['machine_status'].value_counts())
    print()
    info = data.describe()
    variance = pd.DataFrame([data.var(numeric_only=True)], columns=['var'])
    info = pd.concat([info, variance.transpose()])
    return data.head(), data.tail(), info


def plotting_stuff(data, plot_type, title, saving=False):
    fig = plt.figure()
    data.plot(kind=plot_type)
    plt.title(title)
    if saving == True:
        plt.savefig(title + '.png', format='png', dpi=300, transparent=True)
    fig.show()


def manipulate_x(data, print_plot=False):
    data = data.drop(labels=['sensor_15'], axis=1)
    data = data.drop(labels=['sensor_00'], axis=1)

    # data['sensor_51'][110000:140000] = data['sensor_50'][110000:140000]
    data.iloc[110000:140000, data.columns.get_loc('sensor_51')] = data.iloc[110000:140000,
                                                                  data.columns.get_loc('sensor_50')]
    data = data.drop(labels=['sensor_50'], axis=1)

    data = data.drop(labels=['sensor_06', 'sensor_07', 'sensor_08', 'sensor_09'], axis=1)
    data = data.fillna(method='pad', limit=30)
    data = data.dropna()

    if print_plot:
        print((data.isna().sum()))
        plotting_stuff((data.isna().sum()[2:-1]), 'bar', 'fill_nan', saving=True)
    return data


def manipulate_y(data):
    '''
    Convert the classes from strings to values by using the sckit-learn mapper
    '''
    le = LabelEncoder()
    le.fit(data['machine_status'])
    encoded_y = pd.DataFrame(le.transform(data['machine_status']), columns=['target'])
    print(f"encoded_y shape {encoded_y.shape}")

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    return pd.DataFrame(encoded_y, columns=['target'])


def plot_target(data, col='target', saving=False, name='target'):
    '''
    Plot target data
    '''
    y = data[col]
    x = np.linspace(1, len(y), len(y))
    print(f"x shape {x.shape}")
    plt.plot(x, y)
    plt.ylabel('class')
    plt.xlabel('target')
    labels = ['Normal', 'Broken', 'Recovering']
    if col == 'target':
        plt.yticks([1, 0, 2], labels, rotation='vertical')
    elif col == 'machine_status':
        plt.yticks([0, 1, 2], labels, rotation='vertical')
    if saving == True:
        plt.savefig(name + '.png', format='png', dpi=300, transparent=True)
    plt.show()


def plotting_merged(data_x, encoded_y, sensor_names, saving=False):
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)

    for i in sensor_names:
        fig = plt.figure()
        ax = scaled_data[i].plot.line()
        encoded_y.plot(ax=ax)
        plt.title('together_' + str(i))
        plt.legend(['sensor', 'target'])
        if saving == True:
            fig.savefig('Sensor_' + str(i) + '.png', format='png', dpi=300, transparent=True)
        plt.show()


def plotting_together(values):
    plt.figure()
    values.plot(subplots=True, sharex=True, figsize=(30, 55))
    plt.show()


if __name__ == '__main__':
    data, sensor_names = read_data("pump_sensor.csv", 1)

    # plot NaNs in the sensors' inputs
    # plotting_stuff((data.isna().sum())[2:-1], 'bar', 'Raw-NaN')

    # explore data
    head, tail, info = explore(data)
    # plot the variance
    # plotting_stuff(info.iloc[8][1:-1], 'bar', 'variance')

    '''
    removing NaNs
    removing faulty sensors
    removing low variance sensors
    '''
    manipulate_x(data, False)

    # plot the one sensor
    plotting_stuff(data['sensor_50'], 'line', 'sensor50')

    encoded_y = manipulate_y(data)
    plot_target(data=encoded_y, col='target', saving=False)

    values = pd.concat([data[sensor_names], encoded_y], axis=1)
    # plotting_merged(data=data[sensor_names], encoded_y=encoded_y, sensor_names=sensor_names, saving=False)
    # plotting_together(values)
