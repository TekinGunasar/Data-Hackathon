import pandas as pd
from numpy import array
from nltk import word_tokenize
from nltk.corpus import stopwords
from IPython.display import display

class csv():

    SENTENCE_INDEX = 'Sentence'
    STOP_WORDS = (' '.join(stopwords.words('english'))).lower().split(' ')
    PUNKT = ['.',',','-','(',')']

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
        Removes the stop words from an individual sentence in the dataset
    '''
    def remove_stop_words(self,sentence):
        sentence = sentence.split(' ')
        corrected_sentence = ' '.join(list(filter(lambda word: word.lower() not in self.STOP_WORDS and word not in self.PUNKT,
                                                  sentence)))
        return corrected_sentence

    '''
        removes all stop words from the entire data set, replacing the sentences with their corrected version
    '''
    def remove_all_stop_words(self):
        self._csv[self.SENTENCE_INDEX] = self._csv[self.SENTENCE_INDEX].apply(self.remove_stop_words)

    def get_proportion_label(self,label,value):
        return sum(array(self._csv[label].values) == value)/len(self._csv[label].values)

csv_file_path = 'advanced_trainset.csv'
reader = csv(csv_file_path)

label = 'Sentiment'
value = 'negative'

index = 2
reader.remove_all_stop_words()
example_sentence = reader.get_individual_entry(index)
print(example_sentence)
