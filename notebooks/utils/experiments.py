import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm


def test_setup(test_index, pp_count):
    """
    Returns a pair consisting of boolean (True is test patient is PP) and test label
    Labels are 1 for pain, 0 for no pain
    Note that PP must be first in the data set, followed by PNP
    """

    test_is_pp = test_index < pp_count
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


def classify_nusvm_with_xvalid(data_pp, data_pnp, nu, selected_channels, test_index, mul=1, pca_components=None, verbose=True):
    """
    Leaves out patient at test_index and trains a linear SVM classifier on all other patients
    Validates classifier on patient at test_index
    Returns accuracy over all repetitions for the test patient
    Input data must be array of EEG recordings with shape [n_repetitions, n_channels, n_features]
    :param data_pp: positive (e.g. pain) input data
    :param data_pnp: negative (e.g. no pain) input data
    :param nu: SVM hyper-parameter
    :param selected_channels: list of channels to be used for classification
    :param test_index: data of patient at this index position will be used as validation sample
    :param mul: multiply all input data by this factor (defaults to 1, i.e. no change in input values)
    :param pca_components: defaults to None (don't apply PCA); set to an integer to apply PCA with the given number of components
    :param verbose: if true, print out train and test accuracy
    """

    pp_count = np.vstack(data_pp).shape[0]
    pnp_count = np.vstack(data_pnp).shape[0]

    data_bp = np.concatenate((data_pp, data_pnp))

    test_is_pp, test_label = test_setup(test_index, pp_count)
    test_p, train_p = get_train_test(data_bp, test_index)
    train_p_separated = np.vstack(train_p)
    pp_train_len, pnp_train_len = get_pp_pnp_length(pp_count, pnp_count, len(test_p), test_is_pp)

    if pca_components:
        train = pca(ravel_all_trials(train_p_separated, selected_channels) * mul, n_components=pca_components)
        test = pca(ravel_all_trials(test_p, selected_channels) * mul, n_components=pca_components)
    else:
        train = ravel_all_trials(train_p_separated, selected_channels) * mul
        test = ravel_all_trials(test_p, selected_channels) * mul


    labels = [1] * pp_train_len + [0] * pnp_train_len
    test_labels = [test_label] * len(test)

    if verbose:
        print('Test index', test_index, 'Preparing to classify set of', pp_train_len, 'PP and', pnp_train_len, 'PNP.')

    clas = svm.NuSVC(nu=nu, kernel='linear')
    clas.fit(train, labels)
    train_acc = clas.score(train, labels)
    test_acc = clas.score(test, test_labels)

    if verbose:
        print('Train score:', train_acc, '  Test score:', test_acc)

    return test_acc
