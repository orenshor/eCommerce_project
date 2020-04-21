import pandas as pd  # pandas is a data manipulation library
import numpy as np  # provides numerical arrays and functions to manipulate the arrays efficiently
import random

import matplotlib.pyplot as plt  # data visualization library
import operator
from os import path
# import turicreate as tc
from datetime import datetime

"""Load dataset to memory:"""

m_cols = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u1.base', delimiter='\t', names=m_cols, encoding='latin-1')
print(df.shape)

# Add movies names

items = pd.read_csv('u.item', delimiter='|', encoding='latin-1')
new_columns = items.columns.values
new_columns[0] = 'item_id'
new_columns[1] = "item_name"
items.columns = new_columns
df = pd.merge(df, items[["item_id", "item_name"]], on='item_id', how='inner')
print(df.shape)

# Add user gender

users = pd.read_csv('u.user', delimiter='|', encoding='latin-1')
new_columns = users.columns.values
new_columns[0] = 'user_id'
new_columns[2] = "gender"
users.columns = new_columns
df = pd.merge(df, users[["user_id", "gender"]], on='user_id', how='inner')
print(df.shape)

"""Ex 1.a."""

dftmp = df[['item_id', 'item_name', 'rating']].groupby('item_id').agg(
    {'rating': np.mean, 'item_name': lambda x: x.iloc[0]})

dftmp = dftmp.sort_values(by='rating')

dftmp.hist()
print("Three highest ranking movies:\n", dftmp.sort_values(by=['rating'], ascending=False).head(3))

"""Ex 1.b."""

dftmp = df[['item_id', 'gender', 'item_name', 'rating']].groupby(['item_id', 'gender'], as_index=False).agg(
    {'rating': np.mean, 'item_name': lambda x: x.iloc[0]})

# print(dftmp.shape)
# dftmp = dftmp.sort_values(by='rating')
# genders = dftmp['gender'].unique()
# plt.hist([dftmp.loc[dftmp.gender == x, 'rating'] for x in genders])
#
# plt.show()
#
# print("Three highest ranking movies for Males:\n",
#       dftmp.loc[dftmp['gender'] == 'M'].sort_values(by=['rating'], ascending=False).head(3))
# print("Three highest ranking movies for Females:\n",
#       dftmp.loc[dftmp['gender'] == 'F'].sort_values(by=['rating'], ascending=False).head(3))
#
# ratings_by_gender = df.pivot_table('rating', index=['item_id'], columns='gender', aggfunc='mean')
# ratings_by_gender['delta'] = (ratings_by_gender.F - ratings_by_gender.M)
# print(ratings_by_gender.head())
# print(ratings_by_gender.columns.values)


""" Ex 2.a """
print("all df:")
print(df)
m_cols = ['user_id', 'item_id', 'rating', 'timestamp']
df_test = pd.read_csv('u1.test', delimiter='\t', names=m_cols, encoding='latin-1')
items = pd.read_csv('u.item', delimiter='|', encoding='latin-1')
print("items df:")
print(items.columns)
