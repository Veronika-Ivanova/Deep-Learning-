import numpy as np
import pandas as pd
import pickle

embedding_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                 'sex', 'native-country']
nrof_emb_categories = {}
unique_categories = {}

with open('/content/SiriusDL/week08/data/train_adult.pickle', 'rb') as f:
    data, _, _ = pickle.load(f)

for cat in embedding_columns:
    nrof_unique = np.unique(data[cat].values.astype(np.str))
    # data.groupby(cat).agg({cat: 'count'})
    unique_categories[cat] = nrof_unique
    nrof_emb_categories[cat] = len(nrof_unique)
    data[cat + '_cat'] = [np.where(nrof_unique == val)[0][0] for i, val in enumerate(data[cat].values.astype(np.str))]

data.dropna(axis=0,inplace=True)
min_age = data.age.min()  
max_age = data.age.max()
step = (max_age - min_age)/10

feature_list = data['age'].unique()

for i in feature_list:
    mask = (data['age'] == i)
    g = np.floor((i - min_age)/ step)
    data.loc[mask, 'age' + '_bin'] = g

data = data.drop(columns=["age"])  
data.age_bin = data.age_bin.astype(int)
with open('/content/SiriusDL/week08/data/train_adult.pickle', 'wb') as f:
    pickle.dump([data, nrof_emb_categories, unique_categories], f)


with open('/content/SiriusDL/week08/data/valid_adult.pickle', 'rb') as f:
    data, _, _ = pickle.load(f)

for cat in embedding_columns:
    data[cat + '_cat'] = [np.where(unique_categories[cat] == val)[0][0] for i, val in enumerate(data[cat].values.astype(np.str))]
data.dropna(axis=0,inplace=True)
feature_list = data['age'].unique()

for i in feature_list:
    mask = (data['age'] == i)
    g = np.floor((i - min_age)/ step)
    data.loc[mask, 'age' + '_bin'] = g

data = data.drop(columns=["age"])  
data.age_bin = data.age_bin.astype(int)
with open('/content/SiriusDL/week08/data/valid_adult.pickle', 'wb') as f:
    pickle.dump([data, nrof_emb_categories, unique_categories], f)