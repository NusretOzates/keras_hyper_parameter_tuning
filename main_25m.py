import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dot

import keras_tuner as kt
from keras_tuner import HyperParameters

dataset: pd.DataFrame = pd.read_csv('data/ml-25m/ratings.csv', sep=',')

number_users = dataset['userId'].max()
number_movies = dataset['movieId'].max()

min_rating = dataset['rating'].min()
max_rating = dataset['rating'].max()
normalize = lambda rating: (rating - min_rating) / (max_rating - min_rating)

dataset['rating'] = dataset['rating'].map(normalize)
dataset = dataset.sample(frac=1).reset_index(drop=True)

train, test = train_test_split(dataset, test_size=0.2, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)

train_user_ids = train['userId'].values.reshape((-1, 1))
train_movie_ids = train['movieId'].values.reshape((-1, 1))
y_train = train['rating']

test_user_ids = test['userId'].values.reshape((-1, 1))
test_movie_ids = test['movieId'].values.reshape((-1, 1))
y_test = test['rating']

validation_user_ids = validation['userId'].values.reshape((-1, 1))
validation_movie_ids = validation['movieId'].values.reshape((-1, 1))
y_validation = validation['rating']

train_dataset_size = len(train_user_ids)
train = tf.data.Dataset.from_tensor_slices(((train_user_ids, train_movie_ids), y_train)) \
    .shuffle(train_dataset_size) \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE) \
    .cache()

test_dataset_size = len(test_user_ids)
test = tf.data.Dataset.from_tensor_slices(((test_user_ids, test_movie_ids), y_test)) \
    .shuffle(test_dataset_size) \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE) \
    .cache()

val_dataset_size = len(validation_user_ids)
validation = tf.data.Dataset.from_tensor_slices(((validation_user_ids, validation_movie_ids), y_validation)) \
    .shuffle(val_dataset_size) \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE) \
    .cache()


def build_wo_hp():
    user_layer = tf.keras.layers.Input((1,))
    movie_layer = tf.keras.layers.Input((1,))

    embedding_size = 50

    user_embedding = tf.keras.layers.Embedding(number_users + 1,
                                               embedding_size)(user_layer)

    movie_embedding = tf.keras.layers.Embedding(number_movies + 1,
                                                embedding_size)(movie_layer)

    movie_user_dot = tf.keras.layers.Dot(2)([user_embedding, movie_embedding])

    output = tf.nn.sigmoid(movie_user_dot)

    model = tf.keras.Model([user_layer, movie_layer], output)

    # "log" will assign equal probabilities to each order of magnitude range
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer='adam'
    )

    return model


# create a model
def build_model(hp: HyperParameters):
    user_layer = Input((1,))
    movie_layer = Input((1,))

    embedding_size = hp.Int(name='Embedding', min_value=128, max_value=512)
    # "log" will assign equal probabilities to each order of magnitude range
    learning_rate = hp.Float('learning_rate', 1e-6, 1e-2, sampling='log')

    user_embedding = Embedding(number_users + 1, embedding_size)(user_layer)

    movie_embedding = Embedding(number_movies + 1, embedding_size)(movie_layer)

    movie_user_dot = Dot(2)([user_embedding, movie_embedding])

    output = tf.nn.sigmoid(movie_user_dot)

    model = Model([user_layer, movie_layer], output)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    return model


tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_loss', 'min'),
    max_epochs=5,
    directory='tuning_results',
    project_name='movie_recommendation_25m'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

tuner.search(train, validation_data=validation, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps)
model_hp: Model = tuner._build_model(best_hps)
model_hp.fit(train, validation_data=test, callbacks=[stop_early], epochs=5, )

print('=====================================================')
model = build_wo_hp()
model.fit(train, validation_data=test, callbacks=[stop_early], epochs=5)
