from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from frequent_words import Ngrams
from csv_reader import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from numpy import logspace
import xgboost as xgb
from numpy import unique

class model_trainer():

    def __init__(self,model,x,y):
        self.model = model
        self.x = x
        self.y = y

    def get_grid_serach_results(self,parameter_grid,num_folds):
        metrics_of_interest = ['param_n_estimators','param_max_depth','mean_test_score']
        clf = GridSearchCV(self.model, parameter_grid, cv=num_folds, return_train_score=False)
        clf.fit(self.x, self.y)

        return clf.cv_results_

    def get_grid_search_dataframe(self, cv_results):
        return DataFrame(cv_results)[['param_subsample','param_max_depth','param_colsample_bytree','mean_test_score']].sort_values('mean_test_score',
                                                                                                       ascending=False)


csv_file_path = 'advanced_trainset.csv'
reader = csv(csv_file_path)
reader.remove_all_stop_words()
reader.stem_all_data()
top = 200

n = Ngrams(1, reader._csv)
n = Ngrams(1, reader._csv)
most_frequent_positive = set(n.get_most_positive_frequent(top))
most_frequent_negative = set(n.get_most_negative_frequent(top))
most_frequent_neutral = set(n.get_most_neutral_frequent(top))

pos_neg_difference = most_frequent_positive.difference(most_frequent_negative)
pos_neutral_difference = most_frequent_positive.difference(most_frequent_neutral)
pos_encoders = pos_neg_difference.intersection(pos_neutral_difference)

neg_pos_difference = most_frequent_negative.difference(most_frequent_positive)
neg_neutral_difference = most_frequent_negative.difference(most_frequent_neutral)
neg_encoders = neg_pos_difference.intersection(neg_neutral_difference)

neutral_pos_difference = most_frequent_neutral.difference(most_frequent_positive)
neutral_neg_difference = most_frequent_neutral.difference(most_frequent_negative)
neutral_encoders = neutral_pos_difference.intersection(neutral_neg_difference)

encoder = n.get_most_frequent(top)

reader.set_word_encoder(encoder)
feature_vectors = reader.bag_of_words()
targets = reader.get_target_variables()

codified = []
dict = {
    'positive':0,
    'negative':1,
    'neutral':2
}
for value in targets:
    codified.append(dict[value])

clf = xgb.XGBClassifier(
            objective= 'multi:softprob',
            booster='gbtree',
            eval_metric = 'mlogloss',
            use_label_encoder=False,
            verbosity= 0,
            seed=1234
        )

trainer = model_trainer(clf,feature_vectors,codified)

# for svc param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
param_grid = {
    # 'learning_rate': [.001, .05],
    'max_depth': [3, 5, 10, 20],
    # 'min_child_weight': [1,5,10],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    # 'gamma': [0.5, 1, 1.5, 2, 5],
}
num_folds = 5
cv_results = trainer.get_grid_serach_results(param_grid,num_folds)
df_results = trainer.get_grid_search_dataframe(cv_results)

print(df_results)


