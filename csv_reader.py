import pandas as pd
from numpy import array

class csv():

    def __init__(self,csv_file_path):
        self._csv = pd.read_csv(csv_file_path)

    def get_proportion_label(self,label,value):
        return sum(array(self._csv[label].values) == value)/len(self._csv[label].values)


csv_file_path = 'advanced_trainset.csv'
reader = csv(csv_file_path)

label = 'Sentiment'
value = 'negative'
test = reader.get_proportion_label(label,value)
print(test)
