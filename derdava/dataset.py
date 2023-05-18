import cv2
import numpy as np
import os
import pandas as pd
from sklearn import datasets

PATH_TO_CPU = "derdava/cpu.csv"
PATH_TO_CREDITCARD = "derdava/datasets/creditcard.csv"
PATH_TO_DIABETES = "derdava/datasets/diabetes.csv"
PATH_TO_FLOWER_DAISY = "derdava/datasets/flowers/daisy"
PATH_TO_FLOWER_SUNFLOWER = "derdava/datasets/flowers/sunflower"
PATH_TO_PHONEME = "derdava/datasets/phoneme.csv"
PATH_TO_POL = "derdava/datasets/pol.csv"
PATH_TO_WIND = "derdava/datasets/wind.csv"


def load_dataset(name: str):
    """Loads a built-in dataset.

    :param name: One of ``'cpu'``, ``'credit card'``, ``'diabetes'``, ``'flower'``, ``'mnist'``, ``'phoneme'``, ``'pol'``, ``'wind'``.
    :return: A tuple ``(X, y)`` containing the features and labels of the loaded dataset. ``X`` and ``y`` are Numpy arrays.
    :raise ValueError: If ``name`` is not one of the above names.
    """
    if name == "cpu":
        return load_cpu()
    elif name == "credit card":
        return load_creditcard()
    elif name == "diabetes":
        return load_diabetes()
    elif name == "flower":
        return load_flower()
    elif name == "mnist":
        return load_mnist()
    elif name == "phoneme":
        return load_phoneme()
    elif name == "pol":
        return load_pol()
    elif name == "wind":
        return load_wind()
    else:
        raise ValueError("No such dataset found.")


def load_cpu():
    """:meta private:"""
    df = pd.read_csv(PATH_TO_CPU)
    FEATURE_NAMES = list(df.columns)
    LABEL_NAME = "binaryClass"
    FEATURE_NAMES.remove(LABEL_NAME)
    X = df[FEATURE_NAMES].to_numpy()
    y = df[LABEL_NAME].to_numpy()
    return X, y


def load_creditcard():
    """:meta private:"""
    df = pd.read_csv(PATH_TO_CREDITCARD)
    FEATURE_NAMES = []
    for i in range(1, 24):
        feature_name = "x" + str(i)
        FEATURE_NAMES.append(feature_name)
    LABEL_NAME = "y"
    X = df[FEATURE_NAMES][:1000].to_numpy()
    y = df[LABEL_NAME][:1000].to_numpy()
    return X, y


def load_diabetes():
    """:meta private:"""
    df = pd.read_csv(PATH_TO_DIABETES)
    FEATURE_NAMES = list(df.columns)
    LABEL_NAME = "Outcome"
    FEATURE_NAMES.remove(LABEL_NAME)
    X = df[FEATURE_NAMES].to_numpy()
    y = df[LABEL_NAME].to_numpy()
    return X, y


def load_flower():
    """:meta private:"""
    X = []
    X1 = []
    y = []
    y1 = []
    _make_train_data(PATH_TO_FLOWER_DAISY, 0, X, y, cnt=120)
    _make_train_data(PATH_TO_FLOWER_SUNFLOWER, 1, X1, y1, cnt=120)
    X.extend(X1)
    y.extend(y1)
    X = np.array(X)
    y = np.array(y)
    return X, y


def load_mnist():
    """:meta private:"""
    digits = datasets.load_digits()
    df = pd.DataFrame(digits['data'])
    df['label'] = digits['target']
    FEATURE_NAMES = list(df.columns)
    LABEL_NAME = "label"
    FEATURE_NAMES.remove(LABEL_NAME)
    X = df[FEATURE_NAMES].to_numpy()
    y = df[LABEL_NAME].to_numpy()
    return X, y


def load_phoneme():
    """:meta private:"""
    df = pd.read_csv(PATH_TO_PHONEME)
    FEATURE_NAMES = ["V1", "V2", "V3", "V4", "V5"]
    LABEL_NAME = "Class"
    X = df[FEATURE_NAMES].to_numpy()
    y = df[LABEL_NAME].to_numpy()
    return X, y


def load_pol():
    """:meta private:"""
    df = pd.read_csv(PATH_TO_POL)
    FEATURE_NAMES = list(df.columns)
    LABEL_NAME = "binaryClass"
    FEATURE_NAMES.remove(LABEL_NAME)
    X = df[FEATURE_NAMES].to_numpy()
    y = df[LABEL_NAME].to_numpy()
    return X, y


def load_wind():
    """:meta private:"""
    df = pd.read_csv(PATH_TO_WIND)
    FEATURE_NAMES = list(df.columns)
    LABEL_NAME = "binaryClass"
    FEATURE_NAMES.remove(LABEL_NAME)
    FEATURE_NAMES.remove("year")
    FEATURE_NAMES.remove("month")
    FEATURE_NAMES.remove("day")
    X = df[FEATURE_NAMES].to_numpy()
    y = df[LABEL_NAME].to_numpy()
    return X, y


def _make_train_data(root, label, X, y, width=50, height=50, cnt=100):
    """:meta private:"""
    t = 0
    for img in os.listdir(root):
        if t >= cnt:
            break

        path = os.path.join(root, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (width, height))

        X.append(np.array(img))
        y.append(label)

        t += 1
