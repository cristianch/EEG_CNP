import sqlite3
import os


def new_log_database(name):
    """
    Create a new database for logging
    :param name: name of database file
    """
    conn = sqlite3.connect(name)
    c = conn.cursor()
    c.execute('''CREATE TABLE logs
             (ProcessingMethod TEXT, Classifier TEXT, Dataset TEXT, Accuracy REAL, Sensitivity REAL, Specificity REAL, AvgAccuracy REAL, 
             PatientsCorrect REAL, Channels TEXT, Hyperparameters TEXT, Notes TEXT, DetailedPredictions TEXT)''')
    conn.commit()
    conn.close()


def log_to_database(db, processing_method, classifier, dataset, accuracy, sensitivity, specificity, avg_accuracy,
                    patients_correct, channels, hyperparameters, notes, details=None):
    """
    Enter a log entry to already existing database; table must be named 'logs' and follow the structure of 'new_log_database' function.
    :param db: name of database file
    :param processing_method: text
    :param classifier: text
    :param dataset: text
    :param accuracy: real
    :param sensitivity: real
    :param specificity: real
    :param avg_accuracy: real
    :param patients_correct: real
    :param channels: text
    :param hyperparameters: text
    :param notes: text (preferably json)
    :param details: text (preferable json)
    :return: n/a
    """
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('INSERT INTO logs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
              (processing_method, classifier, dataset, accuracy, sensitivity, specificity, avg_accuracy,
               patients_correct, channels, hyperparameters, notes, details))
    conn.commit()
    conn.close()
