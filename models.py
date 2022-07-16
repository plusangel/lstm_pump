from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras import Model
import matplotlib.pyplot as plt

def model_setup_Fapi(in_shope):
    inputs = Input(shape=(in_shope[1], in_shope[2]))
    x = LSTM(units=42, activation='relu', input_shape=(in_shope[1], in_shope[2]), return_sequences=True)(inputs)
    x = LSTM(units=42, activation='relu')(x)
    out_signal = Dense(units=1, name='signal_out')(x)
    out_class = Dense(units=3, activation='softmax', name='class_out')(x)

    model = Model(inputs=inputs, outputs=[out_signal, out_class])

    model.compile(loss={'signal_out': 'mean_squared_error',
                        'class_out': 'categorical_crossentropy'},
                  optimizer='adam',
                  metrics={'class_out': 'acc'})

    print(model.summary())
    return model

def plot_training(history, what='loss', save=False, name='training'):
    fig = plt.figure()
    plt.plot(history[0])
    plt.plot(history[1])
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    if what=='loss':
        plt.title('model loss')
        plt.ylabel('loss')
    elif what=='acc':
        plt.title('model acc')
        plt.ylabel('accuracy')

    if save:
        fig.savefig(name + '_' + what + '.png', format='png', dpi=300, transparent=True)

    plt.show()

def plot_signal_hat(y_test, y_hat, save=False, name='result_signal'):
    fig = plt.figure()
    plt.plot(y_hat)
    plt.plot(y_test)
    plt.legend(['target', 'target_predicted'])
    plt.ylabel('State')
    plt.title('Prediction on test data')
    if save == True:
        fig.savefig(name + '.png', format='png', dpi=300, transparent=True)
    plt.show()