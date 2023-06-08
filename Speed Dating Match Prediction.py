#!/usr/bin/env python
# coding: utf-8

# # Problem Formulation
# 
# 
# ##### We want to find and predict the Speed Dating Match given the other features .
# 
# ##### Input :- Features collected from Surveys from partners.
# 
# ##### Output :- Predected Matching for partners .
# 
# ##### Data Mining Function :- Manipulating ,analyzing , preprocessing the data.
# 
# #### Challenges  :
# 
# ##### Nan cells.
# ##### Unused and unimportnat column.
# ##### Convert the dtype.
# ##### convert strings by One Hot encoding.
# ##### Handling unbalanced data by over sampling
# #### Impact  : 
# ##### Predicting the match of the partners that will lead to a successful match.

# # Questions
# 

# ## Why a simple linear regression model (without any activation function) is not good for classification task, compared to Perceptron/Logistic regression? 
# ---------------------------------
# 
# #### Simple linear regression models are not good for classification tasks because they don't provide a binary output, which is required for classification problems. Linear regression models predict continuous output values that can take on any value within a given range, while classification models require binary outputs, where the predicted value is either 0 or 1.
# 
# #### Perceptron and logistic regression models, on the other hand, are designed specifically for binary classification tasks and use activation functions (such as the sigmoid function) to produce a binary output. They are able to learn decision boundaries that separate different classes and can handle non-linear relationships between input features and output labels. Therefore, they are more suitable for classification tasks compared to simple linear regression models

# ## What's a decision tree and how it is different to a logistic regression model?
# -----------------
# 
# #### A decision tree is a type of supervised learning algorithm that is used for both classification and regression tasks. It is a tree-like model that uses a sequence of decisions on input features to arrive at a prediction. Each internal node in the tree represents a decision based on a feature, while each leaf node represents a class label or a numerical value.
# 
# #### On the other hand, logistic regression is a type of linear regression that is used for binary classification tasks. It models the probability of the occurrence of an event using a logistic function, which maps any real-valued input to a value between 0 and 1. The decision boundary in logistic regression is a straight line, while in decision trees, it is a series of decisions on input features.
# 
# #### The main difference between a decision tree and a logistic regression model is that decision trees are non-linear models and can handle non-linear relationships between input features and output labels, while logistic regression is a linear model that assumes a linear relationship between input features and output labels. Decision trees can also handle categorical and ordinal data, while logistic regression requires numerical input features. However, decision trees can be prone to overfitting, while logistic regression is less prone to overfitting.

# ##  What's the difference between grid search and random search?
# --------------
# #### Grid search and random search are both hyperparameter optimization techniques used to find the best set of hyperparameters for a machine learning model. The main difference between them is the way they search the hyperparameter space.
# 
# #### Grid search exhaustively searches over a predefined set of hyperparameters by creating a grid of all possible hyperparameter combinations. It evaluates each combination using cross-validation and returns the combination that produces the best performance.
# 
# #### Random search, on the other hand, randomly samples hyperparameters from a predefined distribution. It searches over a larger hyperparameter space compared to grid search and can potentially find better hyperparameters. It also evaluates each combination using cross-validation and returns the combination that produces the best performance.
# 
# #### Overall, grid search is a good option when the hyperparameter space is small and the possible combinations are easily enumerable, while random search is a good option when the hyperparameter space is large and the possible combinations are not easily enumerable. Random search is also more computationally efficient when the number of hyperparameters is large.

# ## What's the difference between bayesian search and random search?
# -----
# 
# #### Bayesian search and random search are both hyperparameter optimization techniques used to find the best set of hyperparameters for a machine learning model. The main difference between them is the way they search the hyperparameter space and update the search process.
# 
# #### Random search randomly samples hyperparameters from a predefined distribution and evaluates each combination using cross-validation. It does not take into account the results of previous evaluations and is a purely random process.
# 
# #### Bayesian search, on the other hand, uses the results of previous evaluations to update the search process. It models the distribution of the objective function (i.e., the performance metric of the model) as a probabilistic model and uses it to select the next set of hyperparameters to evaluate. The model updates its beliefs about the objective function as more evaluations are conducted and becomes more certain about the optimal hyperparameters over time.
# 
# #### Overall, Bayesian search is a more intelligent approach than random search as it leverages the information gathered during the search process to guide the search towards promising regions of the hyperparameter space. This can lead to faster convergence and better performance compared to random search. However, Bayesian search requires more computational resources and is more complex to implement compared to random search.

# ## importing libraries  

# In[46]:


## For uploading and accessing the data
import pandas as pd
import numpy as np 

# for Encode categorical features as a numeric values
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


# for Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# for fitting Models
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# ## Load Data and Explore it

# In[47]:


# reading the training dataset 

data= pd.read_csv("train.csv")
data


# In[48]:


# reading the training dataset 

data_test= pd.read_csv("test.csv")
data_test


# In[49]:


data.shape


# In[50]:


data_test.shape


# In[51]:


# anything missing?
data.isnull().sum().sort_values(ascending=False)


# In[52]:


# checking types:
data.info()


# In[53]:


#looking for the columns which have more than 64 % null
check = data.isnull().sum() / len(data) 

cols = check[check > 0.64].index


# In[54]:


list(cols)


# In[55]:


# Below code gives percentage of null in every column
null_percentage = data.isnull().sum()/data.shape[0]*100
list(null_percentage)


# In[56]:



# Below code gives list of columns having more than 64% null
col_to_drop = null_percentage[null_percentage>64].keys()

data = data.drop(col_to_drop, axis=1)
list(col_to_drop)


# In[57]:


# Below code gives percentage of null in every column
null_percentage = data_test.isnull().sum()/data_test.shape[0]*100

# Below code gives list of columns having more than 64% null
col_to_drop = null_percentage[null_percentage>64].keys()

data_test = data_test.drop(col_to_drop, axis=1)
list(col_to_drop)


# In[58]:


data.shape


# In[59]:


data_test.shape


# In[15]:


#get the non-numeric columns from training data
non_numeric_cols = list(data.select_dtypes(exclude=['number']).columns)
list(non_numeric_cols)


# In[16]:


#get the non-numeric columns from test data

non_numeric_cols_test = list(data_test.select_dtypes(exclude=['number']).columns)
list(non_numeric_cols_test)


# In[17]:


#convert it to categories types 
for i in non_numeric_cols:
    data[i] = data[i].astype("category")


# In[18]:


#same for testing data
for i in non_numeric_cols_test:
    data_test[i] = data_test[i].astype("category")


# In[19]:


data.info()


# In[20]:


data['match']


# In[21]:


#droping id cloumn 
data.drop(['id'], axis=1)


# In[22]:


# now we can split the data
import numpy as np
from sklearn.model_selection import train_test_split

y = data['match'] # lower case for vector
X = data.drop('match', axis=1) # upper case for matrix
print('original shape', X.shape, y.shape)


# In[23]:


data.match.value_counts()


# In[24]:


# we extract numeric features and categorical features names
# for later use

# numeric features can be selected by: (based on the df2.info() output )
features_numeric = list(X.select_dtypes(include=['float64', 'int64']))

# categorical features can be selected by: (based on the df2.info() output )
features_categorical = list(X.select_dtypes(include=['category']))

print('numeric features:', features_numeric)
print('categorical features:', features_categorical)


# In[25]:


get_ipython().system('pip install xgboost')


# In[26]:


import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

# define a pipe line for numeric feature preprocessing
# we gave them a name so we can set their hyperparameters
transformer_numeric = Pipeline(
    steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())]
)

# define a pipe line for categorical feature preprocessing
# we gave them a name so we can set their hyperparameters
transformer_categorical = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
# define the preprocessor 
# we gave them a name so we can set their hyperparameters
# we also specify what are the categorical 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformer_numeric, features_numeric),
        ('cat', transformer_categorical, features_categorical)
    ]
)

# combine the preprocessor with the model as a full tunable pipeline
# we gave them a name so we can set their hyperparameters
full_pipline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('my_classifier', svm.SVC())
    ]
)

full_pipline


# ## Grid Search with Cross-validation
# 
# ### First Trail

# In[27]:



# here we specify the search space
# `__` denotes an attribute of the preceeding name
# (e.g. my_classifier__n_estimators means the `n_estimators` param for `my_classifier`)

#trying with first Classifier RandomForest
full_pipline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('my_classifier', RandomForestClassifier())
    ]
)

# these parameters which I had got from a lot of many trials give me this best one

param_grid = {'my_classifier__class_weight': ['balanced'],
              'my_classifier__max_depth': [20],
              'my_classifier__max_features': [None],
              'my_classifier__min_samples_leaf': [2], 
              'my_classifier__min_samples_split': [2], 
              'my_classifier__n_estimators': [300]}





# cv=5 means five-fold cross-validation
# n_jobs means the cucurrent number of jobs

#here we define our gridSearch and passing the pipline as a pramater and our grid search 
grid_search = GridSearchCV(
    full_pipline, param_grid, cv=5, verbose=10, n_jobs=2, 
    scoring='roc_auc')

grid_search.fit(X, y)

print('best score {}'.format(grid_search.best_score_))
print('best score {}'.format(grid_search.best_params_))


# In[31]:


data.shape


# In[60]:


#predict with test data and save the result
y_pred1= grid_search.predict_proba(data_test)


# In[61]:


y_pred1


# In[64]:


#saving the output on csv file for submission
submission = pd.DataFrame()

submission['id'] = data_test['id']

submission['match'] = y_pred1[:,1]

submission.to_csv('sample_submission_Rf.csv' , index= False)


# ## Random Search
# ### Second Trail 

# In[69]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm

#then in trail 2 we trying Random Search with SVC model

full_pipline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('my_classifier', svm.SVC(probability=True))
    ]
)


param_grid = {

    'my_classifier__C': [0.1, 1, 10],
    'my_classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'my_classifier__degree': [2, 3, 4],
    'my_classifier__gamma': ['scale', 'auto'],
    'my_classifier__class_weight': [None, 'balanced']
   
} 

random_search = RandomizedSearchCV(
    full_pipline, param_grid, cv=5, verbose=10, n_jobs=2, 
    scoring='roc_auc')

random_search.fit(X, y)

print('best score {}'.format(random_search.best_score_))
print('best score {}'.format(random_search.best_params_))


# In[70]:


##predict with test data and save the result

y_pred2= random_search.predict_proba(data_test)
y_pred2


# In[71]:


#saving the output on csv file for submission

submission = pd.DataFrame()

submission['id'] = data_test['id']

submission['match'] = y_pred2[:,1]

submission.to_csv('sample_submission_SVM.csv' , index= False)


# ### Third Trial

# In[27]:


from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

#in third trail we trying to get better scrore so we tried XGB Classifier with Random Search 
XGB_pipline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('my_classfier', XGBClassifier())
    ]
)
# these parameters which I had got from a lot of many trials give me this best one


param_grid = {
    'my_classfier__learning_rate': [0.1],
    'my_classfier__max_depth': [10],
    'my_classfier__n_estimators': [500],
    'my_classfier__subsample': [0.9],
    'my_classfier__colsample_bytree': [0.5, 0.7, 0.9],
    'my_classfier__reg_alpha': [0.1],
    'my_classfier__reg_lambda': [0.1],
    'my_classfier__gamma': [0]
}


rand2_search = RandomizedSearchCV(
    XGB_pipline, param_grid, cv=5, verbose=10, n_jobs=2, 
    scoring='roc_auc')

rand2_search.fit(X, y)

print('best score {}'.format(rand2_search.best_score_))
print('best score {}'.format(rand2_search.best_params_))


# In[100]:


#predict with test data and save the result
y_pred3= rand2_search.predict_proba(data_test)
y_pred3


# In[ ]:


#saving the output on csv file for submission
submission = pd.DataFrame()

submission['id'] = data_test['id']

submission['match'] = y_pred3[:,1]

submission.to_csv('sample_submission_XGB1.csv' , index= False)


# ### in third Trail i got best result for now XGBoost Classifier with Random Search
# 

# In[112]:


from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

XGB_pipline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('my_classfier', XGBClassifier())
    ]
)

# these parameters which I had got from a lot of many trials give me this best one

param_grid = {
    
'my_classfier__colsample_bytree': [1.0],
    'my_classfier__gamma': [0.5], 
    'my_classfier__learning_rate': [0.1],
    'my_classfier__max_depth': [10], 
    'my_classfier__n_estimators':[ 900],
    'my_classfier__reg_alpha': [0.5],
    'my_classfier__reg_lambda': [2], 
    'my_classfier__subsample': [0.7]
}


grid2_search = GridSearchCV(
    XGB_pipline, param_grid, cv=5, verbose=10, n_jobs=2, 
    scoring='roc_auc')

grid2_search.fit(X, y)

print('best score {}'.format(grid2_search.best_score_))
print('best score {}'.format(grid2_search.best_params_))


# In[110]:


y_pred4= grid2_search.predict_proba(data_test)
y_pred4


# In[111]:


submission = pd.DataFrame()

submission['id'] = data_test['id']

submission['match'] = y_pred4[:,1]

submission.to_csv('sample_submission_XGB_bayes.csv' , index= False)


# ## Bayes Search
# 

# In[33]:


# Let's try this with xgb model with bayes Search 
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import xgboost as xgb


full_pipline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('my_classfier', XGBClassifier())
    ]
)

# these parameters which I had got from a lot of many trials give me this best one

param_grid = {
    'my_classfier__learning_rate': [0.1],
    'my_classfier__max_depth': [10],
    'my_classfier__n_estimators': [500],
    'my_classfier__subsample': [0.9],
    'my_classfier__colsample_bytree': [0.5, 0.7, 0.9],
    'my_classfier__reg_alpha': [0.1],
    'my_classfier__reg_lambda': [0.1],
    'my_classfier__gamma': [0]
}


bayes_search = BayesSearchCV(full_pipline,param_grid, cv=5, verbose=1, n_jobs=2, n_iter=3,scoring='roc_auc')

bayes_search.fit(X,y)

print('best score {}'.format(bayes_search.best_score_))

print('best score {}'.format(bayes_search.best_params_))


# In[ ]:


y_pred5= bayes_search.predict_proba(data_test)
y_pred5


# ### Finally i got the best score with XGBoost with Random Search 
# ### 0.8668970498513879
# ### with these hayperparmaters
# ----
# ###### {'my_classifier__n_estimators': 300, 'my_classifier__min_samples_split': 2, 'my_classifier__min_samples_leaf': 2, 'my_classifier__max_features': None, 'my_classifier__max_depth': 20, 'my_classifier__class_weight': 'balanced'}

# In[ ]:




