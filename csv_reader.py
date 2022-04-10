import pandas as pd
from numpy import array

class csv():

    SENTENCE_INDEX = 'Sentence'


    def __init__(self,csv_file_path):
        self._csv = pd.read_csv(csv_file_path)


    '''
        Returns an individual entry from the csv
    '''
    def get_individual_entry(self,index):

        sentences = self._csv[self.SENTENCE_INDEX].values
        if index > len(sentences):
            raise ValueError('Index entered exceeds maximum number of entries in csv')

        return self._csv.iloc[index].values


    '''
        Removes the stop words from an individual sentence
    '''
    def remove_stop_word(self,sentence):
        

    def get_proportion_label(self,label,value):
        return sum(array(self._csv[label].values) == value)/len(self._csv[label].values)


csv_file_path = 'advanced_trainset.csv'
reader = csv(csv_file_path)

label = 'Sentiment'
value = 'negative'

index = 0
sentence = reader.get_individual_entry(index)
print(sentence)
