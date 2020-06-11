import csv


def log_result(file, name, accuracy, patients_correct, patients_total, set_name, channels, notes):
    """
    Log the result of an experiment in a csv file with the following information:
        experiment name, accuracy (over all repetitions), number of correct patients/total number of patients, 
        dataset name, selected channels, notes
    """
    writer = csv.writer(file)
    writer.writerow(
        [name, str(accuracy * 100) + '%', ' ' + str(patients_correct) + '/' + str(patients_total), set_name, channels,
         notes])
