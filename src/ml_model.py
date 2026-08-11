from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd

def fit_model(X,y): return HistGradientBoostingRegressor(max_iter=500,learning_rate=.03,max_leaf_nodes=31,random_state=0).fit(X,y)
def predict(model,X,index): return pd.Series(model.predict(X),index=index,name='feature_model')
