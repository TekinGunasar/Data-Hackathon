import pandas as pd

class csv():

    def __init__(self,csv_file_path):
        self._csv = pd.read_csv(csv_file_path)

csv_file_path = 'advanced_trainset.csv'
reader = csv(csv_file_path)
print(reader.csv)