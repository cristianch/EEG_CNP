import csv


def log_result(file, processing_method, classifier, dataset, accuracy, sensitivity, specificity, avg_accuracy,
               patients_correct, channels, hyperparameters, notes, details=None):
    """
    Log fields to a csv file.
    :param file: Reference to an opened file, ready for writing
    :param processing_method:
    :param classifier:
    :param dataset:
    :param accuracy:
    :param sensitivity:
    :param specificity:
    :param avg_accuracy:
    :param patients_correct:
    :param channels:
    :param hyperparameters:
    :param notes:
    :param details:
    :return:
    """
    writer = csv.writer(file)
    writer.writerow(
        [processing_method, classifier, dataset, accuracy, sensitivity, specificity, avg_accuracy, patients_correct,
         channels, hyperparameters, notes, details])
