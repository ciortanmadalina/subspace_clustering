import keras
import numpy as np
import pandas as pd
from keras.utils import to_categorical


def digitize_feature_importance(data, nb_bins=20):
    """[summary]

    Arguments:
        data {[type]} -- [description]

    Keyword Arguments:
        nb_bins {int} -- [description] (default: {20})

    Returns:
        [type] -- [description]
    """
    bins = np.linspace(0, 1, nb_bins)
    digitized_data = np.zeros_like(data)
    for i in range(digitized_data.shape[1]):
        digitized_data[:, i] = np.digitize(data[:, i], bins)
    digitized_data = digitized_data.astype(int)
    digitized_data = digitized_data.T

    img = np.zeros((digitized_data.shape[0], nb_bins + 1))
    for k in range(digitized_data.shape[0]):
        for l in range(digitized_data.shape[1]):
            img[k][digitized_data[k][l]] += 1
    img = img / np.max(img, axis=1).reshape(-1, 1)
    return img


def digitize(data, nb_bins=20):
    """[summary]

    Arguments:
        data {[type]} -- [description]

    Keyword Arguments:
        nb_bins {int} -- [description] (default: {20})

    Returns:
        [type] -- [description]
    """
    bins = np.linspace(0, 1, nb_bins)
    digitized_data = np.zeros_like(data)
    for i in range(digitized_data.shape[1]):
        digitized_data[:, i] = np.digitize(data[:, i], bins)
    digitized_data = digitized_data.astype(int)
    digitized_data = digitized_data.T

    return digitized_data


def digitized_subspace_to_img(digitized_subspace, nb_bins):
    img = np.zeros((nb_bins + 1, nb_bins + 1))

    for i in range(digitized_subspace.shape[1]):
        ii, jj = digitized_subspace[:, i]
        img[ii][jj] += 1
    img = img / np.max(img)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return img


def predict(model,
            population,
            digitized_subspaces,
            n_clusters,
            num_classes=18):
    """[summary]
    Uses the given model to preict the score of feature pair subspaces passed 
    as population. Uses the digitized representation of the input dataset
    Arguments:
        model {[type]} -- [description]
        population {[type]} -- [description]
        digitized_subspaces {[type]} -- [description]
        n_clusters {[type]} -- [description]

    Keyword Arguments:
        num_classes {int} -- [description] (default: {18})

    Returns:
        [type] -- [description]
    """
    inp_data = []
    for i in range(len(population)):
        inp_data.append(
            digitized_subspace_to_img(digitized_subspaces[population[i]], 20))

    if n_clusters is not None:
        inp_k = to_categorical([n_clusters - 2] * len(inp_data),
                               num_classes=num_classes)
        preds = model.predict([inp_data, inp_k])
    else:
        inp_data = np.array(inp_data)
        preds = model.predict(inp_data)

    res = pd.DataFrame()
    res["f1"] = population[:, 0]
    res["f2"] = population[:, 1]
    res["pred"] = np.ravel(preds)

    res = res.sort_values(by="pred", ascending=False).reset_index(drop=True)

    return res


def img_based_model(
    input_shape,
    filter_size=16,
    dropout=0.2,
    firstKernelSize=5,
    secondKernelSize=3,
):

    input_layer = keras.layers.Input(input_shape)
    x = input_layer
    x = keras.layers.Conv2D(filters=filter_size,
                            kernel_size=(firstKernelSize, firstKernelSize),
                            padding='same')(x)
    x = keras.layers.Conv2D(filters=filter_size,
                            kernel_size=(firstKernelSize, firstKernelSize),
                            padding='same')(x)
    x = keras.layers.Activation(activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Conv2D(filters=filter_size * 2,
                            kernel_size=(secondKernelSize, secondKernelSize),
                            padding='same')(x)
    x = keras.layers.Conv2D(filters=filter_size * 2,
                            kernel_size=(secondKernelSize, secondKernelSize),
                            padding='same')(x)

    x = keras.layers.Activation(activation="relu")(x)
    #     x = keras.layers.normalization.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)
    #     x = keras.layers.AveragePooling2D()(x)

    x = keras.layers.Conv2D(filters=filter_size * 2,
                            kernel_size=(secondKernelSize, secondKernelSize),
                            padding='same')(x)
    x = keras.layers.Conv2D(filters=filter_size * 2,
                            kernel_size=(secondKernelSize, secondKernelSize),
                            padding='same')(x)

    x = keras.layers.Activation(activation="relu")(x)
    #     x = keras.layers.normalization.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(1)(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)

    model.compile(loss='mean_squared_error',
                  optimizer="adam",
                  metrics=['mean_absolute_percentage_error', 'mse'])

    return model


def mean_absolute_percentage_error(y_true, y_pred):
    """[summary]

    Arguments:
        y_true {[type]} -- [description]
        y_pred {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
