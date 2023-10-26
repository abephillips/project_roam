import pandas as pd
import numpy as np
import gspread
from google.cloud import storage
import networkx as nx
import time

class SheetsToGCP: 
    def __init__(self, sheet_name, bucket_name): 
        """class to bring data from ONE sheet to ONE bucket"""
        
        # grab svc account credentials
        self.gc = gspread.service_account()
        self.sh = self.gc.open(sheet_name)
        
        # grab svc account credentials
        self.storage_client = storage.Client.from_service_account_json('service_account.json')
        
        # point to gcp bucket
        self.bucket = self.storage_client.get_bucket(bucket_name)

    def download_from_sheets(self, page_name): 
        """ Download PLX data from a prepared google sheet"""
        
        # point to worksheet
        worksheet = self.sh.worksheet(page_name)

        # get_all_values gives a list of rows.
        rows = worksheet.get_all_values()

        # Convert to a DataFrame and render.
        df_from_sheets = pd.DataFrame.from_records(rows[1:])
        df_from_sheets.columns = rows[0]   

        return df_from_sheets

    def upload_to_bucket(self, blob_name, df_to_push):
        """ Upload data to a bucket"""

        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(df_to_push.to_csv(), 'text/csv')

        #returns a public url
        return blob.public_url

    
def check_for_completeness(d): 
    """ checks for a complete path for """
    # d = ptrs_lats.loc[(ptrs_lats.rpo_location == l) & (ptrs_lats.rpo_recruiting_group == r)]

    # get ptrs
    m = d[d.index[d.index.str.startswith('ptr')]].replace({'' : 0}).astype(float)
    m = m.loc[m > 0].reset_index()
    
    # get rid of nans and 0s
    xy = m['index'].str.split('_')
    m['source'] = xy.str[1]
    m['target'] = xy.str[2]
    
    G = nx.DiGraph()
    G.add_edges_from(m[['source', 'target']].values)

    try: 
        return nx.has_path(G, 'ac', 'oa')
    except: 
        return False
    

if __name__ == "__main__": 
    
    start = time.time()
    
    # initialize pipeline endpoints
    # sheet: hc_dashboard_data -> bucket: hc_dashboard_ptrs_lats
    stg = SheetsToGCP(sheet_name = 'hc_dashboard_appdata', bucket_name = 'hc_dashboard_ptrs_lats')
    
    # ptrs and lats pipeline   
    ptrs_lats = stg.download_from_sheets('ptrs_lats_simple')
    
    # check ptrs and lats for complete paths
    ptrs_lats = ptrs_lats[ptrs_lats.apply(check_for_completeness, axis = 1)].assign(
        lookback_window = ptrs_lats.lookback_window.astype(int)
    )
    
    # get the earliest complete path
    ptrs_lats['rnk'] = ptrs_lats.groupby(['rpo_location', 'rpo_recruiting_group']).lookback_window.rank()
    ptrs_lats = ptrs_lats.loc[ptrs_lats.rnk == 1]
    
    
    blob_url_ptrs = stg.upload_to_bucket('app_data/ptrs_lats_complete_paths.csv', ptrs_lats)
    
    # current in_progress pipeline
    curr_in_progress = stg.download_from_sheets('curr_in_progress')
    curr_in_progress['n_weeks'] = (curr_in_progress['application_closed_week'].astype(int) + 1).apply(np.arange)
    blob_url_curr = stg.upload_to_bucket('app_data/curr_in_progress_trunc.csv', curr_in_progress)
    
    end = time.time()
    print(f'Data Transported. Elapsed time: {end - start}')
    