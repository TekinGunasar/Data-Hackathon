from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


class model_trainer():

    def __init__(self,models,training_set):
        self.models = models
        self.x = training_set[0]
        self.y = training_set[1]

    def get_grid_serach_results(self,model,parameter_grid,num_folds):
        metrics_of_interest = ['param_n_estimators','param_max_depth','mean_test_score']
        _model = GridSearchCV(model, parameter_grid, cv=num_folds, return_train_score=False, n_jobs=8)
        _model.fit(self.x, self.y)

        return _model.cv_results_[metrics_of_interest].sort_values('mean_test_score,ascending=False)

from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


class model_trainer():

    def __init__(self,models,training_set):
        self.models = models
        self.x = training_set[0]
        self.y = training_set[1]

    def get_grid_serach_results(self,model,parameter_grid,num_folds):
        metrics_of_interest = ['param_n_estimators','param_max_depth','mean_test_score']
        _model = GridSearchCV(model, parameter_grid, cv=num_folds, return_train_score=False, n_jobs=8)
        _model.fit(self.x, self.y)

        return _model.cv_results_[metrics_of_interest].sort_values('mean_test_score,ascending=False)

