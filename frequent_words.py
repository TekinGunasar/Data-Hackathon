import pandas as pd
from nltk.util import ngrams
import pandas as pd
from numpy import array
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from csv_reader import csv
import matplotlib.pyplot as plt

import nltk

class Ngrams:
    def __init__(self, ngram, data):
        self.ngram = ngram
        self.data = data
#     def myfunction(ngrams):
#         print('The ngram is ' + str(ngrams.ngram))
    def split_labels(self):
        '''
        Split the dataset into each sentiment
        '''
        self.positive = self.data[self.data['Sentiment'] == 'positive']
        self.negative = self.data[self.data['Sentiment'] == 'negative']
        self.neutral = self.data[self.data['Sentiment'] == 'neutral']
#         print(self.positive.head())
#         print(self.negative.head())
#         print(self.neutral.head())
    def get_positive_ngram(self):
        '''
        Return a table of positive sentiment with a new column with all the ngrams
        '''
        if self.ngram == 1:
            self.positive[str(self.ngram) + ' gram'] = self.positive['Sentence'].apply(lambda x: list(ngrams(x.split(), 1)))
            self.positive[str(self.ngram) + ' gram'] = self.positive[str(self.ngram) + ' gram'].apply(lambda x: [a[0] for a in x])
        if self.ngram == 2:
            self.positive[str(self.ngram) + ' gram'] = self.positive['Sentence'].apply(lambda x: list(ngrams(x.split(), 2)))
            self.positive[str(self.ngram) + ' gram'] = self.positive[str(self.ngram) + ' gram'].apply(lambda x: [" ".join(a) for a in x])
        return self.positive

    def plot_positive_ngram_frequency(self):
        self.split_labels()
        series_form = self.get_positive_ngram()[str(self.ngram) + ' gram'].apply(pd.Series).stack()
        fdist_dict = FreqDist(series_form)
        fdist_dict.plot(20, cumulative = False)

    def get_negative_ngram(self):
        '''
        Return a table of positive sentiment with a new column with all the ngrams
        '''
        if self.ngram == 1:
            self.negative[str(self.ngram) + ' gram'] = self.negative['Sentence'].apply(
                lambda x: list(ngrams(x.split(), 1)))
            self.negative[str(self.ngram) + ' gram'] = self.negative[str(self.ngram) + ' gram'].apply(
                lambda x: [a[0] for a in x])
        if self.ngram == 2:
            self.negative[str(self.ngram) + ' gram'] = self.negative['Sentence'].apply(
                lambda x: list(ngrams(x.split(), 2)))
            self.negative[str(self.ngram) + ' gram'] = self.negative[str(self.ngram) + ' gram'].apply(
                lambda x: [" ".join(a) for a in x])
        return self.negative

    def get_neutral_ngram(self):
        '''
        Return a table of positive sentiment with a new column with all the ngrams
        '''
        if self.ngram == 1:
            self.neutral[str(self.ngram) + ' gram'] = self.neutral['Sentence'].apply(
                lambda x: list(ngrams(x.split(), 1)))
            self.neutral[str(self.ngram) + ' gram'] = self.neutral[str(self.ngram) + ' gram'].apply(
                lambda x: [a[0] for a in x])
        if self.ngram == 2:
            self.neutral[str(self.ngram) + ' gram'] = self.neutral['Sentence'].apply(
                lambda x: list(ngrams(x.split(), 2)))
            self.neutral[str(self.ngram) + ' gram'] = self.neutral[str(self.ngram) + ' gram'].apply(
                lambda x: [" ".join(a) for a in x])
        return self.neutral

    def plot_negative_ngram_frequency(self):
        series_form = self.negative[str(self.ngram) + ' gram'].apply(pd.Series).stack()
        fdist_dict = FreqDist(series_form)
        fdist_dict.plot(20, cumulative=False)

    def get_entire_ngram(self):
        '''
        Return a table of positive sentiment with a new column with all the ngrams
        '''
        if self.ngram == 1:
            self.data[str(self.ngram) + ' gram'] = self.data['Sentence'].apply(lambda x: list(ngrams(x.split(), 1)))
            self.data[str(self.ngram) + ' gram'] = self.data[str(self.ngram) + ' gram'].apply(
                lambda x: [a[0] for a in x])
        if self.ngram == 2:
            self.data[str(self.ngram) + ' gram'] = self.data['Sentence'].apply(lambda x: list(ngrams(x.split(), 2)))
            self.data[str(self.ngram) + ' gram'] = self.data[str(self.ngram) + ' gram'].apply(
                lambda x: [" ".join(a) for a in x])
        return self.data

    def plot_entire_ngram_frequency(self):
        series_form = self.data[str(self.ngram) + ' gram'].apply(pd.Series).stack()
        fdist_dict = FreqDist(series_form)
        fdist_dict.plot(20, cumulative=False)
        freq_table = pd.DataFrame.from_dict(fdist_dict, orient='index')
        freq_table = freq_table.reset_index()
        freq_table = freq_table.sort_values(by=[0], ascending=False)
        freq_table = freq_table.reset_index(drop=True)
        return freq_table

    def get_most_frequent(self,top):
        self.split_labels()
        self.get_entire_ngram()
        series_form = self.data[str(self.ngram) + ' gram'].apply(pd.Series).stack()
        fdist_dict = FreqDist(series_form)
        freq_table = pd.DataFrame.from_dict(fdist_dict, orient='index')
        freq_table = freq_table.reset_index()
        freq_table = freq_table.sort_values(by=[0], ascending=False)
        freq_table = freq_table.reset_index(drop=True)
        return ' '.join(freq_table.head(top)['index'].values).lower().split(' ')

    def get_most_positive_frequent(self,top):
        self.split_labels()
        series_form = self.get_positive_ngram()[str(self.ngram) + ' gram'].apply(pd.Series).stack()
        fdist_dict = FreqDist(series_form)
        freq_table = pd.DataFrame.from_dict(fdist_dict, orient='index')
        freq_table = freq_table.reset_index()
        freq_table = freq_table.sort_values(by=[0], ascending=False)
        freq_table = freq_table.reset_index(drop=True)
        return ' '.join(freq_table.head(top)['index'].values).lower().split(' ')

    def get_most_negative_frequent(self,top):
        self.split_labels()
        series_form = self.get_negative_ngram()[str(self.ngram)+' gram'].apply(pd.Series).stack()
        fdist_dict = FreqDist(series_form)
        freq_table = pd.DataFrame.from_dict(fdist_dict, orient='index')
        freq_table = freq_table.reset_index()
        freq_table = freq_table.sort_values(by=[0], ascending=False)
        freq_table = freq_table.reset_index(drop=True)
        return ' '.join(freq_table.head(top)['index'].values).lower().split(' ')

    def get_most_neutral_frequent(self,top):
        self.split_labels()
        series_form = self.get_neutral_ngram()[str(self.ngram)+' gram'].apply(pd.Series).stack()
        fdist_dict = FreqDist(series_form)
        freq_table = pd.DataFrame.from_dict(fdist_dict, orient='index')
        freq_table = freq_table.reset_index()
        freq_table = freq_table.sort_values(by=[0], ascending=False)
        freq_table = freq_table.reset_index(drop=True)
        return ' '.join(freq_table.head(top)['index'].values).lower().split(' ')













