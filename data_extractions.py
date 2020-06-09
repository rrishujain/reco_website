import pandas as pd


def extract_click_data(start_date, end_date, bv_creds, dc_creds):
    query_click = """SELECT clientId, cast((visitStartTime + hits.time/1000) as INTEGER) AS hit_sec, hits.product.productSKU AS pid
    FROM (TABLE_DATE_RANGE([bv-demand.114875145.ga_sessions_], TIMESTAMP('"""+ start_date +"""'), TIMESTAMP('"""+ end_date +"""') ))
    WHERE hits.eCommerceAction.action_type = '2'
    GROUP BY 1, 2, 3"""

    click_data = pd.read_gbq(query_click, project_id = 'bv-demand',dialect = 'legacy', credentials = bv_creds)

    query_country = """SELECT PropertyCode as pid, Country as country
    FROM (
    select * from
    (SELECT PropertyCode, Country
    FROM (TABLE_DATE_RANGE([leisure-bi.LANDING.ACCOMMODATION_BASIC_], TIMESTAMP('"""+ end_date +"""'), TIMESTAMP('"""+ end_date +"""') ))
    GROUP BY 1, 2
    ),(
    SELECT PropertyCode, Country
    FROM (TABLE_DATE_RANGE([leisure-bi.LANDING.DC_ACCOMMODATION_BASIC_], TIMESTAMP('"""+ end_date +"""'), TIMESTAMP('"""+ end_date +"""') ))
    GROUP BY 1, 2 )
    ) group by 1,2"""

    country_data = pd.read_gbq(query_country, project_id = 'leisure-bi',dialect = 'legacy', credentials = dc_creds)

    click_user_data = click_data.merge(country_data, on='pid', how='inner')
    return click_user_data