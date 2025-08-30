#%%

import pandas as pd
import numpy as np
from datetime import datetime
#%%

raw_sample_path = '../data/raw_sample.csv'
ad_feature_path = '../data/ad_feature.csv'
user_profile_path = '../data/user_profile.csv'
#%%

raw_sample = pd.read_csv(raw_sample_path)
print(raw_sample.head(10))
#%%

raw_sample['time_stamp'] = raw_sample['time_stamp'].map(lambda x: datetime.fromtimestamp(x))
print(raw_sample.head(10))
#%%

ad_feature = pd.read_csv(ad_feature_path)
print(ad_feature.head(10))

#%%

user_profile = pd.read_csv(user_profile_path)
print(user_profile.head(10))
#%%

raw_sample.columns = ['user_id', 'time_stamp', 'adgroup_id', 'pid', 'nonclk', 'clk']
user_profile.columns = ['user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level','pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']
print(raw_sample.head(10))
print(user_profile.head(10))
#%%

data = raw_sample.merge(user_profile,on='user_id',how='left')
data = data.merge(ad_feature, on='adgroup_id',how='left')
print(data.shape)
print(data.head(10))
#%%

data = data.fillna(value=0)
print(data.head(10))
#%%

split_time_stamp = '2017-05-12 00:00:00'
train_data = data[data['time_stamp'] < split_time_stamp]
test_data = data[data['time_stamp'] >= split_time_stamp]
print(train_data.shape)
print(test_data.shape)
#%%

print(train_data['shopping_level'].value_counts())
#%%

labels = ['clk']
sparse_features = ['user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
                    'age_level', 'pvalue_level', 'occupation', 'new_user_class_level',
                    'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price', 'pid', 'shopping_level']
domains = ['shopping_level']
columns = sparse_features + labels
print(len(columns))
#%%

train_data = train_data[columns]
print(train_data.shape)
print(train_data.head(10))
#%%

train_data['domain_id'] = train_data[domains].copy().astype(int)
print(train_data.shape)
print(train_data.head(10))
#%%

print(train_data['domain_id'].value_counts())
#%%

test_data = test_data[columns]
test_data['domain_id'] = test_data[domains].copy().astype(int)
print(test_data.shape)
print(test_data.head(10))
#%%

train_data.to_csv('../data/train_data.csv', index=False)
test_data.to_csv('../data/test_data.csv', index=False)
#%%

train_data = pd.read_csv('../data/train_data.csv')
print(train_data.shape)
print(train_data.head(10))
#%%