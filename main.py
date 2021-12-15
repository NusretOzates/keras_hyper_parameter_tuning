import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dot, Input

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
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

test, validation = train_test_split(test, test_size=0.5, random_state=42)

train_user_ids = train['user_id'].values
train_movie_ids = train['item_id'].values
y_train = train['rating']

test_user_ids = test['user_id'].values
test_movie_ids = test['item_id'].values
y_test = test['rating']

val_user_ids = validation['user_id'].values
val_movie_ids = validation['item_id'].values
y_val = validation['rating']

train_dataset_size = len(train_user_ids)
train = tf.data.Dataset.from_tensor_slices(((train_user_ids, train_movie_ids), y_train)) \
    .shuffle(train_dataset_size) \
    .batch(64) \
    .prefetch(tf.data.AUTOTUNE) \
    .cache()

test_dataset_size = len(test_user_ids)
test = tf.data.Dataset.from_tensor_slices(((test_user_ids, test_movie_ids), y_test)) \
    .shuffle(test_dataset_size) \
    .batch(64) \
    .prefetch(tf.data.AUTOTUNE) \
    .cache()

val_dataset_size = len(val_user_ids)
validation = tf.data.Dataset.from_tensor_slices(((val_user_ids, val_movie_ids), y_val)) \
    .shuffle(val_dataset_size) \
    .batch(64) \
    .prefetch(tf.data.AUTOTUNE) \
    .cache()


def build_without_hyperparameters():
    user_layer = Input((1,))
    movie_layer = Input((1,))

    embedding_size = 50

    user_embedding = Embedding(number_users + 1, embedding_size)(user_layer)

    movie_embedding = Embedding(number_movies + 1, embedding_size)(movie_layer)

    movie_user_dot = Dot(2)([user_embedding, movie_embedding])

    output = tf.nn.sigmoid(movie_user_dot)

    model = Model([user_layer, movie_layer], output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam'
    )

    return model


# create a model
def build_model(hp: HyperParameters):
    user_layer = Input((1,))
    movie_layer = Input((1,))

    embedding_size = hp.Int('embedding_size', 30, 75)
    # "log" will assign equal probabilities to each order of magnitude range
    learning_rate = hp.Float('learning_rate', 1e-3, 1e-2, sampling='log')
    user_embedding = Embedding(number_users + 1, embedding_size)(user_layer)

    movie_embedding = Embedding(number_movies + 1, embedding_size)(movie_layer)

    movie_user_dot = Dot(2)([user_embedding, movie_embedding])

    output = tf.nn.sigmoid(movie_user_dot)

    model = Model([user_layer, movie_layer], output)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        run_eagerly=True
    )

    return model


tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_loss', 'min'),
    max_epochs=15,
    hyperband_iterations=6,
    directory='tuning_results',
    project_name='movie_recommendation'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

"""
You should use:

Validation data: To find hyper-parameters
Test data: Test the model in the end

"""
tuner.search(train, callbacks=[stop_early], validation_data=validation)

# Get the optimal hyperparameters
best_hps: HyperParameters = tuner.get_best_hyperparameters(num_trials=5)[0]
print(best_hps.values)

model_hp: Model = build_model(best_hps)
model_hp.fit(train, validation_data=test, callbacks=[stop_early], epochs=best_hps.values['tuner/epochs'])

print('=====================================================')

model = build_without_hyperparameters()
model.fit(train, validation_data=test,  callbacks=[stop_early], epochs=15)
