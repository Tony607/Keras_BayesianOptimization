#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install bayesian-optimization


# In[2]:


import numpy as np
import keras
from tensorflow.keras import backend as K
import tensorflow as tf


# In[3]:


NUM_CLASSES = 10

def get_input_datasets(use_bfloat16=False):
    """Downloads the MNIST dataset and creates train and eval dataset objects.

    Args:
      use_bfloat16: Boolean to determine if input should be cast to bfloat16

    Returns:
      Train dataset, eval dataset and input shape.

    """
    # input image dimensions
    img_rows, img_cols = 28, 28
    cast_dtype = tf.bfloat16 if use_bfloat16 else tf.float32

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # train dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    train_ds = train_ds.batch(64, drop_remainder=True)

    # eval dataset
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_ds = eval_ds.repeat()
    eval_ds = eval_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    eval_ds = eval_ds.batch(64, drop_remainder=True)

    return train_ds, eval_ds, input_shape


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation
from tensorflow.python.keras.optimizer_v2 import rmsprop


def get_model(input_shape, dropout2_rate=0.5, dense_1_neurons=128):
    """Builds a Sequential CNN model to recognize MNIST.

    Args:
      input_shape: Shape of the input depending on the `image_data_format`.
      dropout2_rate: float between 0 and 1. Fraction of the input units to drop for `dropout_2` layer.
      dense_1_neurons: Number of neurons for `dense1` layer.

    Returns:
      a Keras model

    """
    # Reset the tensorflow backend session.
    # tf.keras.backend.clear_session()
    # Define a CNN model to recognize MNIST.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,
                     name="conv2d_1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
    model.add(Dropout(0.25, name="dropout_1"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(dense_1_neurons, activation='relu', name="dense_1"))
    model.add(Dropout(dropout2_rate, name="dropout_2"))
    model.add(Dense(NUM_CLASSES, activation='softmax', name="dense_2"))
    return model


# In[6]:


train_ds, eval_ds, input_shape = get_input_datasets()


# In[12]:


def fit_with(input_shape, verbose, dropout2_rate, dense_1_neurons_x128, lr):

    # Create the model using a specified hyperparameters.
    dense_1_neurons = max(int(dense_1_neurons_x128 * 128), 128)
    model = get_model(input_shape, dropout2_rate, dense_1_neurons)

    # Train the model for a specified number of epochs.
    optimizer = rmsprop.RMSProp(learning_rate=lr)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Train the model with the train dataset.
    model.fit(x=train_ds, epochs=1, steps_per_epoch=468,
              batch_size=64, verbose=verbose)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(eval_ds, steps=10, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Return the accuracy.

    return score[1]


# In[13]:


from functools import partial

verbose = 1
fit_with_partial = partial(fit_with, input_shape, verbose)


# In[14]:


fit_with_partial(dropout2_rate=0.5, lr=0.001, dense_1_neurons_x128=1)


# The BayesianOptimization object will work out of the box without much tuning needed. The main method you should be aware of is `maximize`, which does exactly what you think it does.
# 
# There are many parameters you can pass to maximize, nonetheless, the most important ones are:
# - `n_iter`: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
# - `init_points`: How many steps of **random** exploration you want to perform. Random exploration can help by diversifying the exploration space.

# In[11]:


from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'dropout2_rate': (0.1, 0.5), 'lr': (1e-4, 1e-2), "dense_1_neurons_x128": (0.9, 3.1)}

optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(init_points=10, n_iter=10,)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)


# In[15]:


print(optimizer.max)


# In[ ]:




