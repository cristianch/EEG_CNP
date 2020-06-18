import pandas as pd


def read_log_and_display(file_name):
    log = pd.read_csv(file_name)
    print(log)

