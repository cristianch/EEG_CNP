import numpy as np
from sklearn.decomposition import PCA


def test_setup(test_index, n_pp):
    """
    Returns a pair consisting of boolean (True is test patient is PP) and test label
    Labels are 1 for pain, 0 for no pain
    Note that PP must be first in the data set, followed by PNP
    """

    test_is_pp = test_index < n_pp
    test_label = 1 if test_is_pp else 0
    return test_is_pp, test_label


def get_train_test(data, test_index):
    """
    Splits into train and test sets based on the index of the test patient
    Returns pair of test and train
    """

    return data[test_index], np.delete(data, test_index)


def get_pp_pnp_length(pp_count, pnp_count, test_count, test_is_pp):
    """ Returns pair of the lengths of PP train data and respectively PNP train data """

    pp_train_len = pp_count if not test_is_pp else pp_count - test_count
    pnp_train_len = pnp_count if test_is_pp else pnp_count - test_count
    return pp_train_len, pnp_train_len


def ravel_all_trials(data, channels):
    """
    Ravel first dimension so that trials from all patients are treated separately; select channels
    """
    return np.array(list(map(np.ravel, data[:, channels, :])))


def pca(array, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(array)
