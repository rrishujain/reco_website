import pandas as pd
import numpy as np
import datetime
import numpy as np
import math
import os
import requests


from data_extractions import extract_click_data
start_date = str(datetime.datetime.now().date()-datetime.timedelta(days =1))
end_date = str(datetime.datetime.now().date()-datetime.timedelta(days =1))
import google.oauth2.service_account as service_account
bv_credentials = service_account.Credentials.from_service_account_file('bv-demand-dfd5d066bc83.json')
dc_credentials = service_account.Credentials.from_service_account_file('leisure-bi-33a0c20900d8.json')
click_data  = extract_click_data(start_date, end_date, bv_credentials, dc_credentials)
from hyperparameters import *
clicks_vs_recency = 0.94
multiple_for_prob = 1000
walk_length = 10
walk_times = 300

import time
current_timestamp = int(time.time())
current_timestamp = click_data.hit_sec.max() + 60*60
click_data['time_diff'] = (current_timestamp - click_data['hit_sec'])/(24*60*60)
click_data['time_diff_decay'] = clicks_vs_recency**click_data['time_diff']
user_click_data = click_data.groupby(['clientId', 'pid', 'country'])['time_diff_decay'].sum().reset_index()
xboost_data = pd.read_csv('xgboost_prob.csv', dtype = {'clientId':'str', 'product_id':'str','prediction':'float64'})
user_click_data = user_click_data.merge(xboost_data, left_on=['clientId', 'pid'], right_on=['clientId', 'product_id'], how='left')
user_click_data = user_click_data.fillna(0)
user_click_data['pseudo_client_id'] = user_click_data.groupby('clientId').ngroup() + 1
user_click_data['score'] = ((user_click_data['time_diff_decay'] + 0.9*user_click_data['prediction'])*multiple_for_prob).astype('int')


user_graphs = {}
product_graphs = {}
user_total = {}
product_total = {}

unique_country_list = user_click_data['country'].unique()

def get_graphs(df):
    df['cum_score'] = df.groupby('pseudo_client_id')['score'].cumsum()
    user_graph = df.groupby('pseudo_client_id')[['pid', 'cum_score']].apply(lambda x : list(zip(x['pid'].to_list(), x['cum_score'].to_list()))).to_dict()
    user_total = df.groupby('pseudo_client_id')['cum_score'].max().rename('total_score').to_dict()
    
    df['cum_score_2'] = df.groupby('pid')['score'].cumsum()
    product_graph = df.groupby('pid')[['pseudo_client_id', 'cum_score_2']].apply(lambda x : list(zip(x['pseudo_client_id'].to_list(), x['cum_score_2'].to_list()))).to_dict()
    product_total = df.groupby('pid')['cum_score_2'].max().rename('total_score').to_dict()
    
    return (user_graph, user_total, product_graph, product_total)

for country in unique_country_list:
    a,b,c,d = get_graphs(user_click_data[user_click_data['country'] == country])
    user_graphs[country] = a
    user_total[country] = b
    product_graphs[country] = c
    product_total[country] = d

def search_index(given_list, value):
    start = 0
    end = len(given_list) - 1
    
    ans = -1
    while(start <= end):
        mid = (start + end) // 2
        if (given_list[mid][1] <= value): 
            start = mid + 1
        else:
            ans = mid
            end = mid - 1
    return given_list[ans][0]

import random

def get_reco(client_id, country):
    
    reco_dict = {}
    for i in range(0, walk_times):
        current_id = client_id
        for j in range(0, walk_length):
            product_id = search_index(user_graphs[country][current_id], int(random.random()*user_total[country][current_id]))
            
            if product_id in reco_dict:
                reco_dict[product_id] = reco_dict[product_id] + 1
            else:
                reco_dict[product_id] = 1
            
            current_id = search_index(product_graphs[country][product_id], int(random.random()*product_total[country][product_id]))
    return reco_dict

who_to_push = user_click_data.groupby(['pseudo_client_id', 'country'])['time_diff_decay'].sum().reset_index()
who_to_push['time_diff_decay_max'] = who_to_push.groupby('pseudo_client_id')['time_diff_decay'].transform('max')
who_to_push = who_to_push[who_to_push['time_diff_decay'] == who_to_push['time_diff_decay_max']]
who_to_push = who_to_push[['pseudo_client_id', 'country']].groupby('pseudo_client_id').first()
who_dict = who_to_push.to_dict()['country']


pseudo_df = user_click_data[['clientId', 'pseudo_client_id']].groupby('clientId').first().to_dict()['pseudo_client_id']

def reco_25(user_id):
    if user_id == -1:
        return []
    temp = get_reco(user_id, who_dict[user_id])
    return sorted(temp, key=temp.get, reverse=True)[:25]

last_users = pd.read_csv('users_to_push.csv', dtype = {'clientId':'str', 'last_date':'str'})
last_users = last_users.sort_values(by='last_date', ascending=False)
last_users['pseudo_client_id'] = last_users['clientId'].apply(lambda x: pseudo_df[x] if x in pseudo_df else -1)

from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()

last_users['reco'] = last_users['pseudo_client_id'].progress_apply(reco_25)
send_reco = last_users[last_users['pseudo_client_id'] != -1]
send_reco = send_reco[send_reco['reco'].str.len() > 9]

import json


temp = send_reco.iloc[:, [0,3]].reset_index(drop=True)

temp.to_csv('pushed_+'+end_date+'.csv', index=False)

cookies = {'test.1086':'a', 'test.1095':'a', 'test.segmentation':'a', 'test.1099':'a', 'test.1098':'b', 'test.1101':'a',
'test.1066':'b', 'test.1091':'a', 'test.remarketing':'a', 'test.1065':'c', 'test.1059':'b', 'test.1001':'a',
'__cfduid':'d271efdfbce5a3fd77c11bf8b940cc83e1587721599', 'ERBooking':'220026185',
'lsbrbvcom':'eat86csrac8g41m0kkaphp5nj4','version':'version_x'
}

method = 'POST'
url = "https://www.belvilla.com/user/property_suggestion"
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'authToken':'cGxhdGZvcm06cGxhdGZvcm1QYXNzd29yZA=='
}

start = 0

response_dict = {} 
for start in range(start, temp.shape[0], 3000):
    end = start+3000
    send_final = temp.iloc[start:end, :].groupby('clientId')['reco'].first().to_dict()
    send_final_json = json.dumps(send_final)
    print(start)
    print(len(send_final.keys()))
    response  = requests.request(method = method, url = url,headers= headers, data=send_final_json, cookies=cookies, verify=False)
    print(response.content)
    response_dict[start] = response.content
    time.sleep(80)


# In[50]:


response_dict

bookings_data['pseudo_client_id'] = bookings_data['clientId'].apply(lambda x: pseudo_df[x] if x in pseudo_df else -1)

