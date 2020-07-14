import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import json
import os

# TODO look into imports, should work in both notebooks and test
from .logger import log_result, log_column_titles
from .experiments_setup import test_setup, get_train_test, get_pp_pnp_length, pca, ravel_all_trials
from .database import new_log_database, log_to_database

CSV_EXTENSION = '.csv'
DETAILED_LOG_SUFFIX = '_detailed'


def get_tp_tn_fp_fn(true_labels, pred_labels):
    """
    Calculate true positives, true negatives, false positives, false negatives based on true labels, predicted labels
    :param true_labels: array
    :param pred_labels: array
    :return: true positives, true negatives, false positives, false negatives
    """
    c = confusion_matrix(true_labels, pred_labels, [0, 1])
    return c[1, 1], c[0, 0], c[0, 1], c[1, 0]


def classify_linear_nusvm_with_valid(data_pp, data_pnp, nu, selected_channels, test_index, pca_components=None,
                                     verbose=True, probability=False):
    """
    Leaves out patient at test_index and trains a linear SVM classifier on all other patients
    Validates classifier on patient at test_index
    Input data must be array of EEG recordings with shape [n_repetitions, n_channels, n_features]
    :returns accuracy, true positives, true negatives, false positives, false negatives
    :param data_pp: positive (e.g. pain) input data
    :param data_pnp: negative (e.g. no pain) input data
    :param nu: SVM hyper-parameter
    :param selected_channels: list of channels to be used for classification
    :param test_index: data of patient at this index position will be used as validation sample
    :param pca_components: defaults to None (don't apply PCA); set to an integer to apply PCA with the given number of components
    :param verbose: if true, print out train and test accuracy
    :param probability: use probabilities in classifier
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

    clas = svm.NuSVC(nu=nu, kernel='linear', probability=probability)
    clas.fit(train, labels)
    train_acc = clas.score(train, labels)
    test_acc = clas.score(test, test_labels)
    tp, tn, fp, fn = get_tp_tn_fp_fn(test_labels, clas.predict(test))

    pred_labels = np.around(clas.predict_proba(test), 3) if probability else clas.predict(test)

    if verbose:
        print('Train score:', train_acc, '  Test score:', test_acc)

    return test_acc, tp, tn, fp, fn, test_label, pred_labels


def classify_nusvm_cross_valid(data_pp, data_pnp, nu, selected_channels, channel_names, pca_components=None,
                               verbose=True, log_db_name=None, log_txt=True, log_proc_method=None,
                               log_dataset=None, log_notes=None, log_location='./results/', log_details=False):
    """
    Cross-validates over entire set; each validation fold will consist of the recordings from 1 patient.
    Input data must be array of EEG recordings with shape [n_repetitions, n_channels, n_features]
    :param data_pp:
    :param data_pnp:
    :param nu:
    :param selected_channels:
    :param channel_names:
    :param pca_components: defaults to None (don't apply PCA); set to an integer to apply PCA with the given number of components
    :param verbose:
    :param log_db_name: if set, will be used to indicate name of database log file
    :param log_txt: if true, results will be logged to text file as csv
    :param log_proc_method: processing method name for logging
    :param log_dataset: dataset name for logging
    :param log_notes: should be passed as a dictionary; will be recorded as a json string
    :param log_location: location of log file and database
    :param log_details: if true, predict probabilities and true labels for each partition are logged
    :returns overall accuracy, sensitivity, specificity, average accuracy (mean of accuracy in feach fold),
        true label, predicted labels (per-label probabilities if log_details is True, else just the label)
    """

    total_score, total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0, 0
    patients_correct = 0
    n_patients = len(data_pp) + len(data_pnp)
    details = []
    probability = log_details

    for i in range(n_patients):
        score, tp, tn, fp, fn, tl, pl = classify_linear_nusvm_with_valid(data_pp, data_pnp, nu,
                                                                         selected_channels, i,
                                                                         pca_components=pca_components, verbose=verbose,
                                                                         probability=probability)
        total_score += score
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        if score > 0.5:
            patients_correct += 1

        if log_details:
            details.append({'true label': tl, 'predicted probabilities': pl.tolist()})

    avg_accuracy = total_score / n_patients
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sensitivity = total_tp / (total_tp + total_fn)
    specificity = total_tn / (total_tn + total_fp)
    patients_correct_ratio = patients_correct / n_patients

    classifier = 'Linear SVM'
    notes_json = json.dumps(log_notes) if log_notes else ''

    details_json = json.dumps(details) if log_details else ''

    selected_channel_names = (np.array(channel_names)[selected_channels]).tolist()

    if log_db_name:
        log_to_database(log_location + log_db_name, log_proc_method, classifier, log_dataset, accuracy, sensitivity,
                        specificity,
                        avg_accuracy, float(patients_correct_ratio), str(selected_channel_names), nu, notes_json,
                        details_json)

    if log_txt:
        log_title = (log_proc_method + '_' + classifier).replace(' ', '_')
        if log_details:
            log_title += DETAILED_LOG_SUFFIX
        log_title += CSV_EXTENSION
        log_file_name = log_location + log_title

        log_file_exists = os.path.exists(log_file_name)

        with open(log_location + log_title, 'a', newline='') as file:
            if not log_file_exists:
                log_column_titles(file)
            log_result(file, log_proc_method, classifier, log_dataset, accuracy, sensitivity, specificity,
                       avg_accuracy, float(patients_correct_ratio), str(selected_channel_names), nu, notes_json,
                       details_json)

    if verbose:
        print('Correctly labeled', patients_correct, 'out of', n_patients, 'accuracy', accuracy)

    return accuracy, sensitivity, specificity, avg_accuracy


def classify_nusvm_param_seach(data_pp, data_pnp, nu_lowest, nu_highest, nu_step_size, pca_components=None,
                               verbose=False, log_db_name=None, log_txt=True, log_proc_method=None,
                               log_dataset=None, log_notes=None, log_location='./results/', log_details=False):

    max_acc_overall = {'channels': [], 'value': 0, 'nu': 0}

    for param_nu in np.arange(nu_lowest, nu_highest, nu_step_size):
        print('nu:', param_nu)
        previous_channels = []
        prev_max_acc = 0

        #TODO don't hardcode channel no, extract it from data
        #TODO use channel names instead of numbers
        n_channels = 61

        max_acc = {'index': 0, 'value': 0}
        while max_acc['value'] >= prev_max_acc:
            max_acc = {'index': 0, 'value': 0}
            for channel in range(n_channels):
                if channel in previous_channels:
                    continue

                selected_channels = previous_channels + [channel]

                accuracy, sensitivity, specificity, avg_accuracy = classify_nusvm_cross_valid(data_pp, data_pnp,
                                                                                              param_nu,
                                                                                              selected_channels,
                                                                                              pca_components=pca_components,
                                                                                              verbose=verbose,
                                                                                              log_db_name=log_db_name,
                                                                                              log_txt=log_txt,
                                                                                              log_proc_method=log_proc_method,
                                                                                              log_dataset=log_dataset,
                                                                                              log_notes=log_notes,
                                                                                              log_location=log_location)

                if accuracy > max_acc['value']:
                    max_acc['index'] = channel
                    max_acc['value'] = accuracy
                    max_acc['nu'] = param_nu

            if max_acc['value'] >= prev_max_acc:
                prev_max_acc = max_acc['value']
                previous_channels = previous_channels + [max_acc['index']]
            print(previous_channels, '{:.2f}'.format(prev_max_acc))

        if prev_max_acc > max_acc_overall['value']:
            max_acc_overall['channels'] = list(previous_channels)
            max_acc_overall['value'] = prev_max_acc
            max_acc_overall['nu'] = param_nu
        print('Mac Accuracy:', max_acc_overall)

    print('Final Max Accuracy:', max_acc_overall)
    return max_acc_overall