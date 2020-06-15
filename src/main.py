import json
import os
import pickle
from datetime import datetime
from os import path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint

import evaluation
import lines
import preprocessing
from CustomCSVLogger import CustomCSVLogger
from HTRModel import HTRModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

X_pad = []
y_pad = []
dictionary = {}


def load_json(filename):
    with open(filename, encoding='utf8') as f:
        return json.load(f)


def save_json(filename, data):
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)


def load_pickle(pickle_path):
    f = open(pickle_path, 'rb')
    content = pickle.load(f)
    f.close()
    return content


def prepare_model(load_checkpoint, checkpoint_file=None, fold_index=""):
    model = HTRModel(X_pad.shape[2], X_pad.shape[1], X_pad.shape[3], len(dictionary))
    model.compile(learning_rate=0.001)

    if load_checkpoint:
        if path.exists(checkpoint_file + ".hdf5"):
            model.load_checkpoint(checkpoint_file + ".hdf5")
            print("Loading saved weights from file...")

    checkpoint = ModelCheckpoint(checkpoint_file + ".hdf5", monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True, period=1)
    timestamp = str(datetime.now()).replace(":", "").split(".")[0]
    csv_logger = CustomCSVLogger(additional_columns={'seconds': None},
                                 filename='./logs/training-' + str(fold_index) + '-' + timestamp + '.log')
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return model, [checkpoint, csv_logger, tensorboard_callback]


def train_and_predict(checkpoint_file, num_epochs=300, load_checkpoint=False, enable_kfold=False, num_splits=0,
                      resume_from_fold=None):
    global x_train, y_train, x_test, y_test, X_pad, y_pad
    fold_index = 0

    if enable_kfold:
        kfold = KFold(n_splits=num_splits, shuffle=False, random_state=None)

        for train, val in kfold.split(x_train, y_train):
            print("Starting fold #" + str(fold_index))

            if resume_from_fold is not None and fold_index < resume_from_fold:
                print("Skipping fold #" + str(fold_index))
            else:
                model, callbacks = prepare_model(load_checkpoint, checkpoint_file, str(fold_index))
                model.fit(x=x_train[train], y=y_train[train], validation_data=(x_train[val], y_train[val]),
                          epochs=num_epochs, callbacks=callbacks, verbose=0)
                model.load_checkpoint(checkpoint_file)
                predict, _ = model.predict(x=x_test)
                evaluation.decode_predicted_output(predict, y_test, X_pad, y_pad, dictionary, str(fold_index))

            fold_index += 1
    else:
        model, callbacks = prepare_model(load_checkpoint, checkpoint_file)
        model.fit(x=x_train, y=y_train, validation_split=0.1, epochs=num_epochs, callbacks=callbacks, verbose=0)
        model.load_checkpoint(checkpoint_file)
        predict, _ = model.predict(x=x_test)
        evaluation.decode_predicted_output(predict, y_test, X_pad, y_pad, dictionary)


if __name__ == '__main__':
    datasets = ["genesis", "exodus"]

    try:
        X_pad = load_pickle("X_pad")
        y_pad = load_pickle("y_pad")
        dictionary = load_json("dictionary.json")
    except IOError:
        preprocessing.prepare_images(datasets)
        X_pad, y_pad = lines.prepare_datasets(datasets, extract=False, greyscale=True)

    try:
        x_train = load_pickle("x_train")
        y_train = load_pickle("y_train")
        x_test = load_pickle("x_test")
        y_test = load_pickle("y_test")
    except IOError:
        x_train, x_test, y_train, y_test = train_test_split(X_pad, y_pad, test_size=0.05)

        lines.save_pickle(x_train, "x_train")
        lines.save_pickle(x_test, "x_test")
        lines.save_pickle(y_train, "y_train")
        lines.save_pickle(y_test, "y_test")

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    evaluation.load_latin_words(datasets)

    print(str(len(X_pad)) + " samples")

    with tf.device('/device:GPU:0'):
        train_and_predict("./checkpoint/6C2R-KFold", num_epochs=300, enable_kfold=True, num_splits=10,
                          load_checkpoint=False)
