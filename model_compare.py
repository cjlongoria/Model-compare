from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from args import get_args
from math import floor, ceil
from sklearn import model_selection
from timeit import default_timer as timer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras.models import load_model

def Load_classifier_Data():
    df_classify = pd.read_csv('~/Downloads/covtype.csv', header=None)
    df_classify = df_classify.loc[df_classify[54].isin([3,6,7])]
    df = normalize(df_classify, 10)
    df_x = df.iloc[:,0:-1]
    df_y = df.iloc[:,-1:]
    X = df_x.to_numpy()
    Y = df_y.to_numpy()
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)
    return (X_train,Y_train, X_test, Y_test)

def Load_regression_Data():
    df_regression = pd.read_csv('~/Downloads/energydata_complete.csv')
    df_regression.drop(columns=['date', 'lights', 'rv1','rv2'], inplace=True)
    df_regression['label'] = df_regression['Appliances']
    df_regression.drop(columns=['Appliances'], inplace=True)
    df = normalize(df_regression, -1)
    df_x = df.iloc[:,0:-1]
    df_y = df.iloc[:,-1:]
    X = df_x.to_numpy()
    Y = df_y.to_numpy()
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=123)
    return (X_train,Y_train, X_test, Y_test)

def normalize(df, i):
    column_maxes = df.max()
    df_max = column_maxes.max()
    column_mins = df.min()
    df_min = column_mins.min()
    normalized_df = (df - df_min) / (df_max - df_min)
    df.iloc[:,0:i] = normalized_df.iloc[:,0:i].values
    return df

def summarize_diagnostics(optimizers, history, epoch):
    # plot loss
    fig = plt.figure(figsize=(10,20))
    ax1 = fig.add_subplot(311)
    plt.title('Cross Entropy Loss')
    for i in range(len(history)):
        plt.plot(history[i].history['loss'], label=optimizers[i], linewidth=1)
        # plt.plot(history[0].history['val_loss'], color='orange', label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if epoch >= 10:
        plt.xticks(np.arange(0, epoch, step=(epoch//10)), np.arange(epoch,step=(epoch//10))+1)
    else:
        plt.xticks(np.arange(0, epoch), np.arange(epoch)+1)
    plt.legend()

    # plot divergence
    ax1 = fig.add_subplot(312)
    plt.title('Loss Divergence')
    for i in range(len(history)):
        diverge = []
        zip_object = zip(history[i].history['val_loss'], history[i].history['loss'], )
        for x, y in zip_object:
            diverge.append(x-y)
        plt.plot(diverge, label=optimizers[i], linewidth=1)
        plt.axhline(0, linewidth=1, linestyle='dotted', color='black')
        print(f'{optimizers[i]} - divergence: {diverge[-1]}')
        # plt.plot(history[i].history['val_accuracy'], color='orange', label='Validation')
    plt.ylabel('Divergence')
    plt.xlabel('Epoch')
    if epoch >= 10:
        plt.xticks(np.arange(0, epoch, step=(epoch//10)), np.arange(epoch,step=(epoch//10))+1)
    else:
        plt.xticks(np.arange(0, epoch), np.arange(epoch)+1)
    plt.legend()

    # plot accuracy
    ax1 = fig.add_subplot(313)
    plt.title('Accuracy')
    for i in range(len(history)):
        plt.plot(history[i].history['accuracy'], label=optimizers[i], linewidth=1)
        # plt.plot(history[0].history['val_loss'], color='orange', label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if epoch >= 10:
        plt.xticks(np.arange(0, epoch, step=(epoch//10)), np.arange(epoch,step=(epoch//10))+1)
    else:
        plt.xticks(np.arange(0, epoch), np.arange(epoch)+1)
    plt.legend()
    plt.subplots_adjust(hspace=0.4)

    # plt.show()
    plt.savefig('test1')

def train_val_plot(optimizers, history, epoch):
    # plot loss
    fig = plt.figure(figsize=(16,16))
    # ax1 = fig.add_subplot(221)
    plt.title(f'Cross Entropy Loss - {optimizers[0].upper()}')
    plt.plot(history[0].history['loss'], label='Train', linewidth=1)
    plt.plot(history[0].history['val_loss'], color='orange', label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if epoch >= 10:
        plt.xticks(np.arange(0, epoch, step=(epoch//10)), np.arange(epoch,step=(epoch//10))+1)
    else:
        plt.xticks(np.arange(0, epoch), np.arange(epoch)+1)
    plt.legend()


    # ax1 = fig.add_subplot(222)
    # plt.title(f'Cross Entropy Loss - {optimizers[1].upper()}')
    # plt.plot(history[1].history['loss'], label='Train', linewidth=1)
    # plt.plot(history[1].history['val_loss'], color='orange', label='Validation')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # if epoch >= 10:
    #     plt.xticks(np.arange(0, epoch, step=(epoch//10)), np.arange(epoch,step=(epoch//10))+1)
    # else:
    #     plt.xticks(np.arange(0, epoch), np.arange(epoch)+1)
    # plt.legend()

    # ax1 = fig.add_subplot(223)
    # plt.title(f'Cross Entropy Loss - {optimizers[2].upper()}')
    # plt.plot(history[2].history['loss'], label='Train', linewidth=1)
    # plt.plot(history[2].history['val_loss'], color='orange', label='Validation')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # if epoch >= 10:
    #     plt.xticks(np.arange(0, epoch, step=(epoch//10)), np.arange(epoch,step=(epoch//10))+1)
    # else:
    #     plt.xticks(np.arange(0, epoch), np.arange(epoch)+1)
    # plt.legend()

  
    # ax1 = fig.add_subplot(224)
    # plt.title(f'Cross Entropy Loss - {optimizers[3].upper()}')
    # plt.plot(history[3].history['loss'], label='Train', linewidth=1)
    # plt.plot(history[3].history['val_loss'], color='orange', label='Validation')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # if epoch >= 10:
    #     plt.xticks(np.arange(0, epoch, step=(epoch//10)), np.arange(epoch,step=(epoch//10))+1)
    # else:
    #     plt.xticks(np.arange(0, epoch), np.arange(epoch)+1)
    # plt.legend()
    # plt.subplots_adjust(hspace=0.4)
    plt.savefig('test')


def define_model(opt):
    model = Sequential() 
    dropout_percent = 0.2
    model.add(Dense(128, activation='sigmoid', input_shape=(54,)))
    # model.add(Dropout(dropout_percent))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))


    model.add(Dense(8, activation='softmax'))
    # opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    args_ = get_args()
    print('----------Load Data----------')
    (X_train,Y_train, X_test, Y_test) = Load_classifier_Data()
    # (X_train,Y_train, X_test, Y_test) = Load_regression_Data()


    # optimizers = ['adam', 'rmsprop', 'sgd', 'adagrad']
    optimizers = ['adam']
    results = []
    for opt in optimizers:
        print('----------Build Model----------')
        model = ''
        model = define_model(opt)
        print('----------Train Model----------')
        start = timer()
        history = model.fit(X_train, Y_train, epochs=args_.epochs, batch_size=4 , validation_split= 0.2, verbose=1)
        end = timer()
        print(f'Optimizer: {opt}')
        print(f'Training time: {end - start}')
        print(f'Training Size: {len(X_train)}')
        results.append(history)


    print('----------Plotting----------')
    # summarize_diagnostics(history, args_.epochs)
    # summarize_diagnostics(optimizers, results, args_.epochs)
    train_val_plot(optimizers, results, args_.epochs)
    
    print('----------Evaluate Model----------')
    _, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f'Testing Size: {len(X_test)}')
    print('> %.3f' % (acc * 100.0))
    print()


    print('Complete')
