import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM, TimeDistributed, Flatten, AveragePooling1D, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import CSVLogger
import seaborn as sns
from keras import regularizers
from keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping


class EpilepsyClassifier:
    def __init__(self, path, seed=0, timesteps=128):
        self.data_dim = 0
        self.df = pd.read_csv(path, index_col=0)
        self.df_best = pd.DataFrame(columns=['seed', 'Timestep', 'Acc', 'Val_acc', 'Loss', 'Val_loss'])
        self.timesteps = timesteps
        self.seed = seed
        self.categorize_y()
        self.shuffle_rows()
        self.shape_data()
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.split_()
        self.model_()
        self.return_results()


    def categorize_y(self):
        '''Depending on the type of classification - the values need to be changed in the categorization of y_train,
        y_test (line 84 & 86) and in the output (Dense) layer of the model (line 105)

        Classes:
        Set A - Class 5: EEG recording of a non-epileptic awake patient with eyes open eyes open.
        Set B - Class 4: EEG recording of a non-epileptic awake patient with eyes open eyes closed.
        Set C - Class 3: EEG recording of an epileptic patient during seizure free period using electrodes
                        implanted in the brain epileptogenic zone.
        Set D - Class 2: EEG recording of an epileptic patient during seizure free period from the
                        hippocampal formation of the opposite hemisphere of the brain from C.
        Set E - Class 1: EEG recording of a patient experiencing an active epileptic stroke.

        There are three possible level of classification:
        - 2 class classification: seizure = 1 non-seizure = 0;
        - 3 class classification: seizure = 2 inter-actal seizure = 1 non-seizure = 0;
        - 5 class classification: all claases described above
        '''

        #for 2 class: seizure = 1 non-seizure =0
        #self.df['y'] = self.df['y'].replace({1: 1, 2: 0, 3: 0, 4: 0, 5: 0})

        #for 3 class: #seizure = 2 inter-actal seizure = 1 non-seizure = 0
        #self.df['y'] = self.df['y'].replace({1: 2, 2: 1, 3: 1, 4: 0, 5: 0})

        # for 5 class:
        self.df['y'] = self.df['y'].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})

    def shuffle_rows(self):
        self.df = self.df.sample(frac=1)

    def shape_data(self):
        '''shape the data to fit the LSTM model'''
        data_length = 4096
        timesteps = self.timesteps
        self.data_dim = data_length // timesteps
        print('data dimension: ', self.data_dim)
        print('timesteps: ', self.timesteps)

    def split_(self):
        X = self.df.drop(['y'], axis=1)
        y = self.df['y']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.seed)
        self.x_train = np.reshape(self.x_train.values, (self.x_train.shape[0], self.timesteps, self.data_dim))
        self.x_test = np.reshape(self.x_test.values, (self.x_test.shape[0], self.timesteps, self.data_dim))
        self.y_train = np_utils.to_categorical(self.y_train, num_classes=5)
        self.y_test = np_utils.to_categorical(self.y_test, num_classes=5)


    def model_(self):
        csv_logger = CSVLogger('LSTM_Model_Logger.log')

        tf.random.set_seed(self.seed)
        model = Sequential()
        ##without regularizer
        model.add(LSTM(15, input_shape=(self.timesteps, self.data_dim), return_sequences=True))
        ##with regularizer
        #model.add(LSTM(100, input_shape=(self.timesteps, self.data_dim), return_sequences=True, recurrent_regularizer=regularizers.l2(0.1)))
      #  model.add(Dropout(0.1))
        model.add(TimeDistributed(Dense(50)))
        model.add(GlobalAveragePooling1D())
      #  model.add(LSTM(50, return_sequences=True, recurrent_regularizer=regularizers.l2(0.1)))
        # model.add(Flatten())
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                            callbacks=[csv_logger], batch_size=64, epochs=40)


        best_val = history.history['val_accuracy'][-1]
        best_acc = history.history['accuracy'][-1]
        best_loss = history.history['loss'][-1]
        best_val_loss = history.history['val_loss'][-1]

        df_ = pd.DataFrame()
        df_.loc[self.seed, 'seed'] = self.seed
        df_.loc[self.seed, 'Timestep'] = self.timesteps
        df_.loc[self.seed, 'Acc'] = best_acc
        df_.loc[self.seed, 'Val_acc'] = best_val
        df_.loc[self.seed, 'Loss'] = best_loss
        df_.loc[self.seed, 'Val_loss'] = best_val_loss

        self.df_best = pd.concat([self.df_best, df_])
        
        #PLOTS
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')

        plt.title('Training and validation accuracy')
        plt.legend()
        plt.show()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')

        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        #CONFUSION MATRIX

        y_pred = model.predict(self.x_test)
        y_test_class = np.argmax(self.y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)

        # target_names = ['1', '2', '3', '4', '5']

        #two class
        #target_names = ['0', '1']

        #three class
        #target_names = ['0', '1', '2']

        #five class
        target_names = ['0', '1', '2', '3', '4']

        # Accuracy of the predicted values
        print(classification_report(y_test_class, y_pred_class, target_names=target_names))
        cm = confusion_matrix(y_test_class, y_pred_class)
        print(cm)

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g');

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        ax.set_title('Confusion Matrix Epileptic Seizures')
        ax.xaxis.set_ticklabels(['0','1', '2', '3', '4'])
        ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4'])

        plt.show()

    def return_results(self):
        return self.df_best
