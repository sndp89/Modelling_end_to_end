# XM VOLUNTARY CHURN MODEL, 06/2019 
# PREDICTS PROBABILITY AN XM MOBILE CUSTOMER WILL CHURN
# MICHAEL MANLEY



from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import sklearn
import warnings
from numpy import nan
import datetime
from datetime import date
import time
import pickle
from sklearn.externals import joblib
   

from automata import auto


MODEL_ID = 'EBI_N19_06_001'
MODEL_PREFIX = "SUB_XM_VOL_DNG"


def datamap(data, obj, varbs):

    values = {
    'call_svc_oth_cnt_7d': 0,
    'call_agt_hand_cnt': 0, 
    'r_hsd_upload_gb' : 17.8175598504,
    'r_no_products': 2 
    } 

    for name in values: 
        data[[name]] = data[[name]].fillna(values[name])
    

    data['score'] = obj.predict_proba(data[varbs])[:,1]
    
    return data


def get_output(df):
    df.columns = [x.upper() for x in df.columns]
    df['GUID'] = df['ACCOUNT']
    df['SNAPSHOT_DATE'] = auto.process_date
    df['YEAR'] = int(str(auto.process_date)[0:4])
    df['MONTH'] = int(str(auto.process_date)[4:6])
    df['MODEL_OUTPUT_ID'] = MODEL_ID
    df['MODEL_SCORE'] = df['SCORE']
    df['MODEL_DECILE'] = ''
    df['MODEL_DECILE_REG'] = ''
    df['MODEL_DECILE_DIV'] = ''
    df['MODEL_SEGMENT'] = ''
    df['MODEL_CLASSIFICATION'] = ''
    df['MODEL_INSERTION_DATE'] = date.today().strftime("%Y-%m-%d")
    df['MODEL_PREFIX'] = MODEL_PREFIX

    score_output = df.ix[:,[
                        'GUID',
                        'ACCOUNT',
                        'ACCOUNTID',
                        'CORP_SYSPRIN',
                        'HOUSEKEY',
                        'SNAPSHOT_DATE',
                        'YEAR',
                        'MONTH',
                        'MODEL_OUTPUT_ID',
                        'MODEL_PREFIX',
                        'MODEL_SCORE',
                        'MODEL_DECILE',
                        'MODEL_DECILE_REG',
                        'MODEL_DECILE_DIV',
                        'MODEL_SEGMENT',
                        'MODEL_CLASSIFICATION',
                        'MODEL_INSERTION_DATE',
                        'EPS_ACCT_ID',
                        'EPS_BUSN_ID',
                        'EPS_ADDR_ID']]
    
    score_output.columns = ['GUID'
                        ,'ACCOUNT_NUMBER'
                        ,'CUSTOMER_ACCOUNT_ID'
                        ,'CORP_SYSPRIN'
                        ,'HOUSE_KEY'
                        ,'SNAPSHOT_DATE'
                        ,'YEAR'
                        ,'MONTH'
                        ,'MODEL_OUTPUT_ID'
                        ,'MODEL_PREFIX'
                        ,'MODEL_SCORE'
                        ,'MODEL_DECILE'
                        ,'MODEL_DECILE_REG'
                        ,'MODEL_DECILE_DIV'
                        ,'MODEL_SEGMENT'
                        ,'MODEL_CLASSIFICATION'
                        ,'MODEL_INSERTION_DATE'
                        ,'EPS_ACCT_ID'
                        ,'EPS_BUSN_ID'
                        ,'EPS_ADDR_ID']
    
    return(score_output)


def run():
    filename = '/home/ebisasicep/sasproj/models/ebi_n19_06_001/ebi_n19_06_001.pkl' 
    
    with open(filename, 'rb') as f: 
        clf = pickle.load(f, encoding='latin1', fix_imports=True) 

    df = auto.read_table("EBI_N19_06_001_HIVE")

    #filename = '/home/ebisasicep/sasproj/models/ebi_n19_06_001/ebi_n19_06_001.pkl'

    # clf = pickle.load(open(filename, 'rb'))

    # df = auto.read_table("TEST_MODEL_PYTHON_HIVE_TABLE")

    column_names = [i[1] for i in df.columns.str.split('.')]
    df.columns = column_names
    
    features = ['call_svc_oth_cnt_7d', 'call_agt_hand_cnt', 'r_hsd_upload_gb', 'r_no_products'] 

    df1 = datamap(df, clf, features)

    score_output = get_output(df1)
    auto.write_file(score_output, 'EBI_N19_06_001') 

    # clean up temporary folder if you use hdfs version read & write
    auto.clean_up()

