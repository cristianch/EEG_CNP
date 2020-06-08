import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

# TODO look into imports, should work in both notebooks and test
from .logger import log_result
from .experiments_setup import test_setup, get_train_test, get_pp_pnp_length, pca, ravel_all_trials


def get_tp_tn_fp_fn(true_labels, pred_labels):
    c = confusion_matrix(true_labels, pred_labels, [0, 1])
    return c[1, 1], c[0, 0], c[0, 1], c[1, 0]


def classify_nusvm_with_valid(data_pp, data_pnp, nu, selected_channels, test_index, pca_components=None,
                              verbose=True):
    """
    Leaves out patient at test_index and trains a linear SVM classifier on all other patients
    Validates classifier on patient at test_index
    :returns accuracy, true positives, true negatives, false positives, false negatives
    Input data must be array of EEG recordings with shape [n_repetitions, n_channels, n_features]
    :param data_pp: positive (e.g. pain) input data
    :param data_pnp: negative (e.g. no pain) input data
    :param nu: SVM hyper-parameter
    :param selected_channels: list of channels to be used for classification
    :param test_index: data of patient at this index position will be used as validation sample
    :param pca_components: defaults to None (don't apply PCA); set to an integer to apply PCA with the given number of components
    :param verbose: if true, print out train and test accuracy
    """

    pp_count = np.vstack(data_pp).shape[0]
    pnp_count = np.vstack(data_pnp).shape[0]

    data_bp = np.concatenate((data_pp, data_pnp))

    test_is_pp, test_label = test_setup(test_index, len(data_pp))
    test_p, train_p = get_train_test(data_bp, test_index)
    train_p_separated = np.vstack(train_p)
    pp_train_len, pnp_train_len = get_pp_pnp_length(pp_count, pnp_count, len(test_p), test_is_pp)

    if pca_components:
        train = pca(ravel_all_trials(train_p_separated, selected_channels), n_components=pca_components)
        test = pca(ravel_all_trials(test_p, selected_channels), n_components=pca_components)
    else:
        train = ravel_all_trials(train_p_separated, selected_channels)
        test = ravel_all_trials(test_p, selected_channels)

    labels = [1] * pp_train_len + [0] * pnp_train_len
    test_labels = [test_label] * len(test)

    if verbose:
        print('Test index', test_index, 'Preparing to classify set of', pp_train_len, 'PP and', pnp_train_len, 'PNP.')

    clas = svm.NuSVC(nu=nu, kernel='linear')
    clas.fit(train, labels)
    train_acc = clas.score(train, labels)
    test_acc = clas.score(test, test_labels)
    tp, tn, fp, fn = get_tp_tn_fp_fn(test_labels, clas.predict(test))

    if verbose:
        print('Train score:', train_acc, '  Test score:', test_acc)

    return test_acc, tp, tn, fp, fn


def classify_nusvm_cross_valid(data_pp, data_pnp, nu, selected_channels, pca_components=None,
                               verbose=True, logging=True):
    """
    :param data_pp:
    :param data_pnp:
    :param nu:
    :param selected_channels:
    :param pca_components:
    :param verbose:
    :param logging:
    :returns overall accuracy, sensitivity, specificity, average accuracy
    """
    total_score, total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0, 0
    patients_correct = 0
    n_patients = len(data_pp) + len(data_pnp)
    for i in range(n_patients):
        score, tp, tn, fp, fn = classify_nusvm_with_valid(data_pp, data_pnp, nu,
                                                       selected_channels, i,
                                                       pca_components=pca_components, verbose=verbose)
        total_score += score
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        if score > 0.5:
            patients_correct += 1

    avg_accuracy = total_score / n_patients
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sensitivity = total_tp / (total_tp + total_fn)
    specificity = total_tn / (total_tn + total_fp)

    # TODO better logging
    log_title = 'all_results/test_'
    with open(log_title, 'a') as file:
        log_result(file, log_title, accuracy, patients_correct, n_patients, "TODO SET NAME", selected_channels, "TODO NOTES")

    if verbose:

        print('Correctly labeled', patients_correct, 'out of', n_patients, 'accuracy', accuracy)

    return accuracy, sensitivity, specificity, avg_accuracy

