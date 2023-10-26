##### streamlit app

import streamlit as st
import time
import joblib
import simpy

import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

import openpyxl
import tempfile
from utils.helpers import *
import random

from models.ROAM_v2 import *

st.set_page_config(
    page_title="Current Pipeline",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")
@st.cache_data 
def load_data(): 
    try: 
        data = pd.read_csv('gs://hc_dashboard_ptrs_lats/app_data/ptrs_lats_complete_paths.csv',
                     storage_options={"token": "utils/service_account.json"})
        data = data.rename(columns = {'rpo_location':'location', 'rpo_recruiting_group': 'role'}).fillna(0)

        candidates_in_progress = pd.read_csv('gs://hc_dashboard_ptrs_lats/app_data/curr_in_progress_trunc.csv',
                               storage_options={"token": "utils/service_account.json"})
    except: 
        data = pd.read_csv('appdata/ptrs_lats_complete_paths.csv')

        candidates_in_progress = pd.read_csv('appdata/curr_in_progress_trunc.csv')
    
    return data, candidates_in_progress

data, candidates_in_progress = load_data()
rng = pd.date_range(
    start = pd.to_datetime('2023/03/13'), 
    end = pd.to_datetime('2023/12/25'), freq = '7D')

if 'change_dates' not in st.session_state: 
    st.session_state['change_dates'] = {}
    
if 'count' not in st.session_state:
    st.session_state['count'] = 0 

# import models

with open('./models/model_dictionary.pkl', 'rb') as f:
    clf_dict = pickle.load(f)
    
with open('./models/outcome_classifier.pkl', 'rb') as f:
    outcome_clf = pickle.load(f)

# attrtition model

# lump phone interview with ac and hc with os
candidates_in_progress['Phone Interview'] = candidates_in_progress['phone_interview'].where(candidates_in_progress['phone_interview'].isna(), '✅').fillna('')

candidates_in_progress['Onsite Interview'] = candidates_in_progress['onsite_interview'].where(candidates_in_progress['onsite_interview'].isna(), '✅').fillna('')

candidates_in_progress['Hiring Committee'] = candidates_in_progress['hiring_committee'].where(candidates_in_progress['hiring_committee'].isna(), '✅').fillna('')

candidates_in_progress['Offer Extended'] = candidates_in_progress['offer_extended'].where(candidates_in_progress['offer_extended'].isna(), '✅').fillna('')

candidates_in_progress['Offer Accepted'] = candidates_in_progress['offer_accepted'].where(candidates_in_progress['offer_accepted'].isna(), '✅').fillna('')

candidates_in_progress['Application ID'] = candidates_in_progress['application_id']


candidates_in_progress['Current Status'] = candidates_in_progress[['phone_interview', 'onsite_interview', 'hiring_committee', 'offer_extended', 'offer_accepted']].idxmax(axis = 1).fillna('application_created')

candidates_in_progress['Current Status'] = candidates_in_progress['Current Status'].replace({'phone_interview': 'application_created', 'hiring_committee': 'onsite_interview'})

def get_projected_outcomes(candidates_in_progress, clf = None): 
    X = candidates_in_progress[['offer_accepted', 'offer_extended', 'onsite_interview', 'phone_interview', 'application_closed_week']].rename(columns = {'application_closed_week': 'n_weeks'})
    
    X[['offer_accepted', 'offer_extended', 'onsite_interview', 'phone_interview']] = (X[['offer_accepted', 'offer_extended', 'onsite_interview', 'phone_interview']] > 0).astype(int)
    
    y = clf.predict_proba(X)
    return y

candidates_in_progress[['ACCEPTED', 'DECLINED', 'REJECTED', 'WITHDRAWN']] = get_projected_outcomes(candidates_in_progress, outcome_clf)


def assign_cohorts(n_acc, n_rej, n_wit, n_dec, cip): 
    cd = cip.sort_values(['ACCEPTED', 'WITHDRAWN', 'REJECTED', 'DECLINED'], ascending = False).reset_index()
    
    proj = np.hstack([['ACCEPTED' ] * n_acc, ['WITHDRAWN'] * n_wit, ['REJECTED'] * n_rej, ['DECLINED'] * n_dec])
    cd['Projected Outcome'] = proj
    cd['Confidence'] = cd.apply(lambda row : '{:,.2f}%'.format(row[row['Projected Outcome']] * 100), axis = 1)    
 
    
    return cd
    
    

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    col_map = {
        "IN_PROGRESS": "#fff700", 
        "ACCEPTED": "#65fe08", 
        "REJECTED": "#fa8072", 
        "DECLINDED": "blue", 
        "WITHDRAWN": "grey", 
    }
    try: 
        background_color = col_map[val]
        color = 'white' if val != 'IN_PROGRESS' and val != 'ACCEPTED' else '#4f555b'
    except KeyError: 
        background_color = 'white'
        color = 'black'
    return f'background-color: {background_color};\n color: {color};'


# define some re-used variables
start_date = pd.to_datetime('today').to_period('W').start_time + pd.Timedelta(6, unit='d')
end_date = pd.to_datetime('12/31/2024').to_period('W').start_time + pd.Timedelta(6, unit='d')
        


st.markdown("""
<h1 style='text-align: center;'>Current Pipeline</h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
# Control Panel
Choose the location and role. Then choose pick OA goals that the simulation will target
""")
location = st.sidebar.selectbox(
    'Location',
    data.location.unique(),
)

options = location_roles(location, data)
role = st.sidebar.selectbox(
    'Role',
    options 
    # index = 4
    
)

    # dl
    
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         summarize_data(f)
#         print('created temporary directory', tmpdirname)

#         # directory and contents have been removed

#         with open('./vovo_output.xlm', "rb") as k:
#             st.download_button('Download the Google-approved summary of this data', data = k, use_container_width=True, 
#                                mime = "application/vnd.google-apps.spreadsheet"
#                               )


d = data.loc[(data.location == location) & (data.role == role)]
d_actual = d.copy().reset_index()
# add ptrs and lats to session state
# st.session_state['ptrs_lats'] = {
#     ('ac', 'pi'): {'ptr': d['ptr_ac_pi'].values[0], 'lat': d['lat_ac_pi'].values[0]}, 
#     ('ac', 'os'): {'ptr': d['ptr_ac_os'].values[0], 'lat': d['lat_ac_os'].values[0]}, 
#     ('pi', 'os'): {'ptr': d['ptr_pi_os'].values[0], 'lat': d['lat_pi_os'].values[0]}, 
#     ('os', 'hc'): {'ptr': d['ptr_os_hc'].values[0], 'lat': d['lat_os_hc'].values[0]}, 
#     ('os', 'oe'): {'ptr': d['ptr_os_oe'].values[0], 'lat': d['lat_os_oe'].values[0]},
#     ('hc', 'oe'): {'ptr': d['ptr_hc_oe'].values[0], 'lat': d['lat_hc_oe'].values[0]},
#     ('oe', 'oa'): {'ptr': d['ptr_oe_oa'].values[0], 'lat': d['lat_oe_oa'].values[0]}
# }


col1, col3, col4, col5 = st.columns(4)

with col1:
    st.markdown("""#### AC : OS""") 

    ac_os_ptr = st.number_input('PTR', 
                                min_value=0.0, max_value=1.0, 
                                value=d['ptr_ac_os'].values[0], 
                                label_visibility="visible", step = 0.01, key = 'ac_os_ptr')

    ac_os_lat = st.number_input('Latency (Weeks)',
                                min_value=0, 
                                value=min(int(d['lat_ac_os'].values[0] // 7), 4), 
                                label_visibility="visible", step = 1, key = 'ac_os_lat')

    subcol1, subcol2 = st.columns(2)
    with subcol1: 
        value = d_actual['ptr_ac_os'][0]
        st.metric('Data PTR', value = value, delta = np.round(value - ac_os_ptr, 2))

    with subcol2: 
        lat_value = d_actual['lat_ac_os'][0] // 7
        st.metric('Data Latency', value = lat_value, delta = lat_value - ac_os_lat)

with col3: 
    st.markdown("""#### OS : OE""")
    os_oe_ptr = st.number_input('PTR',
                                min_value=0.0, max_value=1.0, 
                                value=d['ptr_os_oe'].values[0], 
                                label_visibility="visible", step = 0.01, key = 'os_oe_ptr')
    os_oe_lat = st.number_input('Latency (Weeks)',
                                min_value=0, 
                                value=min(int(d['lat_os_oe'].values[0] // 7), 4), 
                                label_visibility="visible", step = 1, key = 'os_oe_lat')

    subcol1, subcol2 = st.columns(2)
    with subcol1: 
        value = d_actual['ptr_os_oe'][0]
        st.metric('Data PTR', value = value, delta = np.round(value - os_oe_ptr, 2))

    with subcol2: 
        lat_value = d_actual['lat_os_oe'][0] // 7
        st.metric('Data Latency', value = lat_value, delta = lat_value - os_oe_lat)

with col4: 
    st.markdown("""#### OE : OA""")

    oe_oa_ptr = st.number_input('PTR',
                                min_value=0.0, max_value=1.0, 
                                value=d['ptr_oe_oa'].values[0], 
                                label_visibility="visible", step = 0.01, key = 'oe_oa_ptr')
    oe_oa_lat = st.number_input('Latency (Weeks)', 
                                min_value=0, value=min(int(d['lat_oe_oa'].values[0] // 7), 4), 
                                label_visibility="visible", step = 1, key = 'oe_oa_lat')

    subcol1, subcol2 = st.columns(2)
    with subcol1: 
        value = d_actual['ptr_oe_oa'][0]
        st.metric('Data PTR', value = value, delta = np.round(value - oe_oa_ptr, 2))

    with subcol2: 
        lat_value = d_actual['lat_oe_oa'][0] // 7
        st.metric('Data Latency', value = lat_value, delta = lat_value - oe_oa_lat)
            
with col5: 
    st.markdown("""#### PPR""")
    ppr = st.number_input('PPR', min_value=0, max_value=100, value=4, label_visibility="visible", step = 1, key = 'ppr')


    ppr_value = 4
    st.metric('Target PPR', value = ppr_value, delta = ppr - ppr_value)
    

d1 = []
ptrs_lats = dir()
for name in ptrs_lats: 
    if (name.endswith('lat')) | (name.endswith('ptr')) | (name == 'ppr'): 
        d1.append({'variable': name, 'value': eval(name)})
        
d2 = pd.DataFrame(d1)
v2 = d2.variable.str.split("_")

d2 = d2.assign(
    stat = v2
)

d2['x'] = d2.stat.str[0]
d2['y'] = d2.stat.str[1]
d3 = d2.groupby(['x', 'y']).apply(lambda g : g['value'].values)

d3.name = 'stats'
d3 = d3.reset_index()

d_ptr_dic = {}
for i, r in d3[['x', 'y', 'stats']].iterrows(): 
    d_ptr_dic[(r['x'], r['y'])] = {'ptr_log': np.log(r['stats'][1]), 'ptr': r['stats'][1], 'lat': r['stats'][0]}
#
st.session_state['ptrs_lats'] = d_ptr_dic
G = nx.Graph()
G.add_edges_from(d3[['x', 'y']].values)
nx.set_edge_attributes(G, values = d_ptr_dic)

def get_path_attrs(G, x, y): 
    p = 0
    l = 0
    i = 0
    
    for path in nx.all_simple_paths(G, x, y, cutoff=None): 
        p += np.exp(nx.path_weight(G, path, 'ptr_log'))
        l += nx.path_weight(G, path, 'lat')
        i += 1
        
    return p, l/i

ac_oa_ptr, ac_oa_lat = get_path_attrs(G, 'ac', 'oa')
# pi_oa_ptr, pi_oa_lat = get_path_attrs(G, 'pi', 'oa')
os_oa_ptr, os_oa_lat = get_path_attrs(G, 'os', 'oa')
# hc_oa_ptr, hc_oa_lat = get_path_attrs(G, 'hc', 'oa')
oe_oa_ptr, oe_oa_lat = get_path_attrs(G, 'oe', 'oa')

ac_ppr = (d2[d2.variable == 'ppr']['value'] / ac_oa_ptr).values[0] / 4.6

taba, tabb = st.tabs(['Current Pipeline', 'Applications'])

with taba: 
    cip = candidates_in_progress.loc[
        (candidates_in_progress.rpo_location == location) & (candidates_in_progress.rpo_recruiting_group == role)
    ]
    pline = cip.groupby('Current Status')['Application ID'].nunique().reset_index()
    stage_metrics = []
    stage_metric_str = ''
    
    for i, pipeline_stage in pline.iterrows(): 
        if i == len(pline) - 1: 
            stage_metric_str += f'col{i}'
        else: 
            stage_metric_str += f'col{i}, '
            
        stage_metrics.append((pipeline_stage['Current Status'], pipeline_stage['Application ID']))
    
    stage_metric_str += f' = st.columns({i+1})'
    
    stage_metric_str += '\n'
    
    for j, stage_metric in enumerate(stage_metrics): 
        t, v = stage_metric
        
        stage_metric_str += f"""with col{j}: \n st.markdown('**{t}**'.replace('_', ' ').title()) \n st.markdown('### {v}')
        """
        stage_metric_str += '\n'
        
    exec(stage_metric_str)
     
    
    ac = np.round(pline.loc[pline['Current Status'] == 'application_created']['Application ID'])
    # pi = np.round(pline.loc[pline['Current Status'] == 'phone_interview']['Application ID'])
    os = np.round(pline.loc[pline['Current Status'] == 'onsite_interview']['Application ID'])
    oe = np.round(pline.loc[pline['Current Status'] == 'offer_extended']['Application ID'])
    oa = np.round(pline.loc[pline['Current Status'] == 'offer_accepted']['Application ID'])
    
    ac = 0 if len(ac) == 0 else ac.values[0]
    # pi = 0 if len(pi) == 0 else pi.values[0]
    os = 0 if len(os) == 0 else os.values[0]
    oe = 0 if len(oe) == 0 else oe.values[0]
    oa = 0 if len(oa) == 0 else oa.values[0]
    
    acoa = np.round(ac*ac_oa_ptr)
    # pioa = np.round(pi*pi_oa_ptr)
    osoa = np.round(os*os_oa_ptr)
    oeoa = np.round(oe*oe_oa_ptr)
    
    ac -= acoa
    # pi -= pioa
    os -= osoa
    oe -= oeoa

    total_oa = acoa + osoa + oeoa + oa
    total_non_oa = ac + os + oe
    withdrawn = np.round(0.2 * total_non_oa)
    rej = total_non_oa - withdrawn
    
    total = len(cip)    
    
    cip = assign_cohorts(np.int(total_oa), np.int(rej), np.int(withdrawn), np.int(0),  cip)
    
    st.markdown("""
    #### Current Pipeine Projection
    """)

    df = pd.DataFrame(
        dict(
        Stage = ['Applications Created', 'Onsite Interview', 'Offer Extended', 'Offer Accepted',
                 'Applications Created', 'Onsite Interview', 'Offer Extended', 'Offer Accepted'], 
        count = [acoa, osoa, oeoa, oa, 
                 ac, os, oe, 0],
        Projection = ['Not Accepted',  'Not Accepted', 'Not Accepted', 'Not Accepted', 
                    'Accepted',  'Accepted', 'Accepted', 'Accepted', ][::-1]
        )
    )
    colalpha, colbeta = st.columns([0.8, 0.2])
    with colalpha: 
        fig = px.funnel(df, x = 'count', y = 'Stage', color = 'Projection')
        st.plotly_chart(fig, use_container_width=True)
        
    with colbeta: 
        
        st.metric('Total Pipeline OAs', value = acoa  + osoa + oeoa + oa)
        st.metric('Total Rejected', value = ac+os+oe)
        st.metric('Pipeline Health', value = 0.2)
        
    
    coleta, coltau = st.columns([0.8, 0.2])
    with coleta: 
        st.markdown("""
            #### Projected Cummulative Offers Accepted
        """)
        partition_date = pd.to_datetime('today')
        pipeline = cip.copy().assign(
            n_weeks = cip.application_closed_week, 
            candidate_pid = cip['Application ID'], 
            date = partition_date, 
            month = partition_date.month, 
            current_outcome = cip['Projected Outcome'],
            event = cip['Current Status']
        )
        pipeline['N'] = 1
        
        capacity_list = np.ones(52)
        capacity_list[:] = 500
        sim_len = 52

        columns = ['in_progress', 'ACCEPTED', 'DECLINED', 'REJECTED', 'WITHDRAWN']
#        columns.extend(outcomes)

        records = pd.DataFrame(columns = columns)
        records['date'] = pd.date_range(start = partition_date,
                                        end = partition_date + pd.Timedelta(sim_len, unit = 'W'), 
                                        freq = '7D')
        records['wall time'] = np.arange(sim_len+1)
        records = records.fillna(0)
        OA_df = simple_scheduler(clf_dict = clf_dict, 
                              pipeline = pipeline, 
                              os_interview_capacity=capacity_list, 
                              tps_interview_capacity = capacity_list, 
                              classes = None, records = records,
                              features_in = ['n_weeks'], 
                              sim_duration=sim_len)

            

        k = OA_df.assign(
            accepted_cum = OA_df.ACCEPTED.cumsum(), 
            rejected_cum = OA_df.REJECTED.cumsum()
        )
          
            
        fig = px.line(k, x = 'date', y = 'accepted_cum', line_shape = 'hv')
        fig['data'][0]['showlegend'] = True
        fig['data'][0]['name'] = 'Estimated Cummulative Offers Accepted'
        
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Cummulative Offers Accepted"
        )
        
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        # fig.update_layout(title_text="Projected Pipeline Outcome", font_size=18)
        st.plotly_chart(fig, use_container_width=True)
            
with tabb: 
    application_search = st.text_input('Search In Progress Applications', value="")
    
    m1 = cip["Application ID"].astype(str).str.contains(application_search)
    
    cip_search = cip[m1]
    
    if application_search: 
        st.dataframe(
            data = cip_search[['Application ID', 'Phone Interview', 'Onsite Interview', 'Offer Extended', 'Offer Accepted', 'Current Status', 'Projected Outcome', 'Confidence']].style.applymap(color_negative_red),
            use_container_width = True
        )    
        
        
st.download_button('Download this data', data = candidates_in_progress.to_csv().encode('utf-8'), use_container_width=True)