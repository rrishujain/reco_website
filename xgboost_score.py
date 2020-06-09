import xgboost as xgb
import pandas as pd
import numpy as np
import datetime
import numpy as np
import math
import os
import requests
import pickle

import google.oauth2.service_account as service_account

bv_credentials = service_account.Credentials.from_service_account_file('bv-demand-dfd5d066bc83.json')
dc_credentials = service_account.Credentials.from_service_account_file('leisure-bi-33a0c20900d8.json')
start_date = str(datetime.datetime.now().date()-datetime.timedelta(days =60))
end_date = str(datetime.datetime.now().date()-datetime.timedelta(days =1))

product_data = """SELECT
  date,
  clientId,
  visitId,
  visitStartTime,
  hits.time,
  hits.hitNumber,
  MAX(IF(hits.customDimensions.index = 1, hits.customDimensions.value, NULL)) AS user_id_session,
  MAX(IF(hits.customDimensions.index = 2, hits.customDimensions.value, NULL)) AS user_id_user,
  MAX(IF(hits.customDimensions.index = 6, hits.customDimensions.value, NULL)) AS variant,
  MAX(IF(hits.customDimensions.index=59, hits.customDimensions.value, NULL)) AS product_id,
  MAX(IF(hits.customDimensions.index=60, hits.customDimensions.value, NULL)) AS product_name,
  MAX(IF(hits.customDimensions.index=61, hits.customDimensions.value, NULL)) AS product_country,
  MAX(IF(hits.customDimensions.index=62, hits.customDimensions.value, NULL)) AS product_region,
  MAX(IF(hits.customDimensions.index=63, hits.customDimensions.value, NULL)) AS product_ski_region,
  MAX(IF(hits.customDimensions.index=64, hits.customDimensions.value, NULL)) AS product_city,
  MAX(IF(hits.customDimensions.index=65, hits.customDimensions.value, NULL)) AS product_type,
  MAX(IF(hits.customDimensions.index=66, hits.customDimensions.value, NULL)) AS product_pax,
  MAX(IF(hits.customDimensions.index=67, hits.customDimensions.value, NULL)) AS product_bedrooms,
  MAX(IF(hits.customDimensions.index=68, hits.customDimensions.value, NULL)) AS product_stars,
  MAX(IF(hits.customDimensions.index=69, hits.customDimensions.value, NULL)) AS product_picture_count,
  MAX(IF(hits.customDimensions.index=70, hits.customDimensions.value, NULL)) AS product_review_count,
  MAX(IF(hits.customDimensions.index=71, hits.customDimensions.value, NULL)) AS product_average_rating,
  MAX(IF(hits.customDimensions.index=72, hits.customDimensions.value, NULL)) AS product_discount_percentage,
  MAX(IF(hits.customDimensions.index=73, hits.customDimensions.value, NULL)) AS product_discount_absolute,
  MAX(IF(hits.customDimensions.index=74, hits.customDimensions.value, NULL)) AS product_contract_type,
  MAX(IF(hits.customDimensions.index=75, hits.customDimensions.value, NULL)) AS product_wifi,
  MAX(IF(hits.customDimensions.index=76, hits.customDimensions.value, NULL)) AS product_pool,
  MAX(IF(hits.customDimensions.index=77, hits.customDimensions.value, NULL)) AS product_pets,
  MAX(IF(hits.customDimensions.index=78, hits.customDimensions.value, NULL)) AS product_deposit,
  MAX(IF(hits.customDimensions.index=79, hits.customDimensions.value, NULL)) AS product_additional_cost
FROM
  TABLE_DATE_RANGE([bv-demand:114875145.ga_sessions_],
    TIMESTAMP('"""+ start_date +"""'), TIMESTAMP('"""+ end_date +"""'))
WHERE
  hits.eventInfo.eventCategory = 'product detail'
  AND hits.eventInfo.eventAction = 'page open'
GROUP BY
  1,
  2,
  3,
  4,
  5,
  6"""


product_click_data  = pd.read_gbq(product_data, project_id = 'bv-demand',dialect = 'legacy', credentials = bv_credentials)
product_click_data['select_top'] = product_click_data.sort_values(by=['clientId','visitStartTime', 'hits_hitNumber'], ascending=[True, False, False]).groupby(['clientId', 'product_id']).cumcount()+1
selected_columns_product = ['clientId', 'product_id', 'product_pax', 'product_stars', 'product_bedrooms', 'product_wifi', 'product_pool','product_pets']
product_click_data_final = product_click_data[product_click_data['select_top']==1][selected_columns_product]
product_click_data_final = product_click_data_final[product_click_data_final['product_id'] != 'Not Applicable']
product_click_data_final = product_click_data_final.replace('NaN', '0')
product_click_data_final = product_click_data_final.astype({'product_pax':'int', 'product_stars':'int','product_bedrooms':'int', 'product_wifi':'int', 'product_pool':'int', 'product_pets':'int'})

user_data = """SELECT
  date,
  clientId,
  visitId,
  visitStartTime,
  hits.time,
  hits.hitNumber,
  MAX(IF(hits.customDimensions.index = 1, hits.customDimensions.value, NULL)) AS user_id_session,
  MAX(IF(hits.customDimensions.index = 2, hits.customDimensions.value, NULL)) AS user_id_user,
  MAX(IF(hits.customDimensions.index = 6, hits.customDimensions.value, NULL)) AS variant,
  MAX(IF(hits.customDimensions.index=13, hits.customDimensions.value, NULL)) AS pax,
  MAX(IF(hits.customDimensions.index=27, hits.customDimensions.value, NULL)) AS amenities,
  MAX(IF(hits.customDimensions.index=29, hits.customDimensions.value, NULL)) AS pets
FROM
  TABLE_DATE_RANGE([bv-demand:114875145.ga_sessions_],
    TIMESTAMP('"""+ start_date +"""'), TIMESTAMP('"""+ end_date +"""'))
WHERE
  hits.eventInfo.eventCategory = 'lister'
  AND hits.eventInfo.eventAction = 'page open'
GROUP BY
  1,
  2,
  3,
  4,
  5,
  6"""

user_data  = pd.read_gbq(user_data, project_id = 'bv-demand',dialect = 'legacy', credentials = bv_credentials)
user_data['select_top'] = user_data.sort_values(by=['clientId','visitStartTime', 'hits_hitNumber'], ascending=[True, False, False]).groupby(['clientId']).cumcount()+1
selected_columns_user = ['clientId', 'pax', 'amenities', 'pets']
user_data_final = user_data[user_data['select_top']==1][selected_columns_user]
user_data_final['wifi'] = user_data_final['amenities'].apply(lambda x: 1 if 'wifi' in x else 0)
user_data_final['pool'] = user_data_final['amenities'].apply(lambda x: 1 if 'pool' in x else 0)
user_data_final = user_data_final.fillna('0')
user_data_final = user_data_final.replace(['None','Not Applicable', '', '5 normandie','n'], '0')
user_data_final = user_data_final.astype({'pax':'int', 'pets':'int'})
user_data_final = user_data_final.drop('amenities', axis=1)

booking_data = """SELECT
  clientId,
  hits.product.productSKU AS product_id,
  1 as booking_flag
FROM
  TABLE_DATE_RANGE([bv-demand:114875145.ga_sessions_],
    TIMESTAMP('"""+ start_date +"""'), TIMESTAMP('"""+ end_date +"""'))
WHERE hits.eCommerceAction.action_type = '6'
group by 1,2,3
"""
booking_data = pd.read_gbq(booking_data, project_id = 'bv-demand',dialect = 'legacy', credentials = bv_credentials)
final = product_click_data_final.merge(user_data_final, on = 'clientId', how='left')
final = final.merge(booking_data, on = ['clientId', 'product_id'], how='left')
final.isna().sum()
final = final.fillna(0)
final['made_booking'] = final.groupby('clientId')['booking_flag'].transform('sum')
final = final[final['made_booking'] >0 ]
final = final.drop('made_booking', axis=1)

columns = ['product_pax', 'product_stars','product_bedrooms', 'product_wifi', 'product_pool', 'product_pets','pax', 'pets', 'wifi', 'pool']
final_dmatrix = xgb.DMatrix(data=final[columns])
model = pickle.load( open( "xgb.p", "rb" ) )
pred = model.predict(final_dmatrix)
final['prediction'] = pred
to_save = ['clientId', 'product_id', 'prediction']
final[to_save].to_csv('xgboost_prob.csv', index=False)