import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from joblib import dump

class Square:
   def __init__(self,val):
      self.val=val
   def getVal(self):
      return self.val*self.val

#define loss scorers
def root_mean_squared_log_error(y_true, y_pred):
    t = np.array(y_true)
    p = np.array(y_pred)
    log_error = np.log(1+t) - np.log(1+p)
    return ((log_error**2).mean())**0.5

def root_mean_squared_log_error_neg(y_true, y_pred):
    t = np.array(y_true)
    p = np.array(y_pred)
    log_error = np.log(1+t) - np.log(1+p)
    return -1 * ((log_error**2).mean())**0.5

class Model_trainer:
    def __init__(self, data_csv_path):
        self.pipe = None
        self.data = pd.read_csv(data_csv_path)


    def set_pipeline(self):
        pproc_cont_imp = SimpleImputer()
        pproc_cont_scaler = MinMaxScaler()
        pproc_cont = make_pipeline(pproc_cont_imp, pproc_cont_scaler)

        pproc_cat_imp = SimpleImputer(strategy='most_frequent')
        pproc_cat_encoder = OneHotEncoder(handle_unknown='ignore')
        pproc_cat = make_pipeline(pproc_cat_imp, pproc_cat_encoder)

        pproc = make_column_transformer(
            (pproc_cont, make_column_selector(dtype_exclude=["O"])),
            (pproc_cat, make_column_selector(dtype_include=["O"])))

        model = XGBRegressor()

        self.pipe = make_pipeline(pproc, model)

    def train_save(self):
        X = self.data.drop(columns=['Id','SalePrice'])
        y = self.data.SalePrice

        self.pipe.fit(X,y)
        dump(self.pipe, 'model.joblib')
