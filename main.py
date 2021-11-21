import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate

import matplotlib.pyplot as plt
import keras_tuner as kt
from keras_tuner import HyperParameters

header = ['user_id', 'item_id', 'rating', 'timestamp']
dataset: pd.DataFrame = pd.read_csv('data/ml-100k/u.data', sep='\t', names=header)

number_users = 943
number_movies = 1682

min_rating = dataset['rating'].min()
max_rating = dataset['rating'].max()
normalize = lambda rating: (rating - min_rating) / (max_rating - min_rating)

dataset['rating'] = dataset['rating'].map(normalize)
# X_train, X_test, y_train, y_test = train_test_split(dataset, test_size=0.2, random_state=42)
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

X_train = train[['user_id', 'item_id']]
y_train = train['rating']

X_test = test[['user_id', 'item_id']]
y_test = test['rating']


# create a model

class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = Embedding(num_users, 1)
        self.movie_embedding = Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = Embedding(num_movies, 1)

    def call(self, inputs, training=None, masks=None):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


def build_model(hp: HyperParameters):
    model = RecommenderNet(number_users + 1, number_movies + 1,
                           embedding_size=hp.Int(name='Embedding', min_value=50, max_value=125))
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    )
    return model


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=15,
    directory='tuning_results',
    project_name='movie_recommendation'

)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, validation_data=(X_test, y_test), callbacks=[stop_early], epochs=10)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# history = model.fit(x=X_train,
#                     y=y_train,
#                     batch_size=64,
#                     epochs=5, verbose=1,
#                     validation_data=(X_test, y_test))
#
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title("model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper left")
# plt.show()
