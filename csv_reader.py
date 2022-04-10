import pandas as pd
from numpy import array
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer

ps = PorterStemmer()

class csv():

    SENTENCE_INDEX = 'Sentence'
    SENTIMENT_INDEX = 'Sentiment'
    STOP_WORDS = (' '.join(stopwords.words('english'))).lower().split(' ')
    PUNKT = ['.',',','-','(',')','$','\'s','n\'t','r\'e','\'d','%',':','``','&','\'','!','@']
    word_encoder = []

    bag_of_words_key = []

    def __init__(self,csv_file_path):
        self._csv = pd.read_csv(csv_file_path)

    def get_target_variables(self):
        return self._csv[self.SENTIMENT_INDEX].values

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
        sentence = word_tokenize(sentence)
        corrected_sentence = ' '.join(list(filter(lambda word: word.lower() not in self.STOP_WORDS and word not in self.PUNKT,
                                                  sentence)))
        return corrected_sentence

    '''
        removes all stop words from the entire data set, replacing the sentences with their corrected version
    '''
    def remove_all_stop_words(self):
        self._csv[self.SENTENCE_INDEX] = self._csv[self.SENTENCE_INDEX].apply(self.remove_stop_words)

    def stem_sentence(self,sentence):
        sentence = word_tokenize(sentence)
        stemmed_sentence = ' '.join([ps.stem(word) for word in sentence])
        return stemmed_sentence

    def stem_all_data(self):
        self._csv[self.SENTENCE_INDEX] = self._csv[self.SENTENCE_INDEX].apply(self.stem_sentence)

    def set_word_encoder(self,encoder):
        self.word_encoder = encoder

    def get_one_hot_encoding(self,sentence):
        one_hot_vector = list(map(
            lambda word:1 if word in sentence.split(' ') else 0,self.word_encoder
        ))

        return one_hot_vector

    def bag_of_words(self):
        self.remove_all_stop_words()
        feature_vectors = self._csv[self.SENTENCE_INDEX].apply(self.get_one_hot_encoding).values

        return list(feature_vectors)

    def get_proportion_label(self,label,value):
        return sum(array(self._csv[label].values) == value)/len(self._csv[label].values)


    def tf_idf(self):
        vectorizer=TfidfVectorizer(min_df=0.01)
        self.remove_all_stop_words()
        self.stem_all_data()
        corpus = self._csv[self.SENTENCE_INDEX]
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        print(feature_names)


path_to_csv = 'advanced_trainset.csv'
reader = csv(path_to_csv)
reader.tf_idf()






