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
    data = pd.read_csv('gs://hc_dashboard_ptrs_lats/app_data/ptrs_lats_complete_paths.csv',
                 storage_options={"token": "utils/service_account.json"})
    data = data.rename(columns = {'rpo_location':'location', 'rpo_recruiting_group': 'role'}).fillna(0)

    candidates_in_progress = pd.read_csv('gs://hc_dashboard_ptrs_lats/app_data/curr_in_progress_trunc.csv',
                           storage_options={"token": "utils/service_account.json"})
    
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

candidates_in_progress['Phone Interview'] = candidates_in_progress['phone_interview'].where(candidates_in_progress['phone_interview'].isna(), '✅').fillna('')

candidates_in_progress['Onsite Interview'] = candidates_in_progress['onsite_interview'].where(candidates_in_progress['onsite_interview'].isna(), '✅').fillna('')

candidates_in_progress['Offer Extended'] = candidates_in_progress['offer_extended'].where(candidates_in_progress['offer_extended'].isna(), '✅').fillna('')

candidates_in_progress['Offer Accepted'] = candidates_in_progress['offer_accepted'].where(candidates_in_progress['offer_accepted'].isna(), '✅').fillna('')

candidates_in_progress['Application ID'] = candidates_in_progress['application_id']


candidates_in_progress['Current Status'] = candidates_in_progress[['phone_interview', 'onsite_interview', 'offer_extended', 'offer_accepted']].idxmax(axis = 1).fillna('application_created')

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

def reset_change_dates(): 
    st.session_state.count = 0
    st.session_state.change_dates = {}

    
def increment_count():
    st.session_state.count += 1
    return 
    
def render_elem(): 
    s = ''
    for row in range(st.session_state.count): 
        s += f"""
col1, col2, col3 = st.columns(3)
with col1: 
    st.markdown("Target {row + 2}")
with col2: 
    change_date{row+1} = st.date_input('Target Date', 
                                value=change_date{row} + pd.Timedelta(13, unit = 'W'),
                                min_value=change_date{row} + pd.Timedelta(13, unit = 'W'), 
                                max_value=end_date, key = 'change_date_{row+1}'
    )
with col3: 
    change_amount{row+1} = st.number_input('OA Goal', min_value=0.0, max_value=10000.0, value=change_amount{row}, key = 'change_amount_{row+1}', 
                                    label_visibility="visible")
        """
    return s

def record_change_dates(): 
    for row in range(st.session_state.count): 
        st.session_state.change_dates[row + 2] = {'date': eval(f'change_date{row+1}'), 'quantity': eval(f'change_amount{row+1}'), 'event': 'oa'}
    return 

# attrtition model

attrition_model = joblib.load("./models/attrition_model.joblib")

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
st.download_button('Download this data', data = candidates_in_progress.to_csv().encode('utf-8'), use_container_width=True)
    
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         summarize_data(f)
#         print('created temporary directory', tmpdirname)

#         # directory and contents have been removed

#         with open('./vovo_output.xlm', "rb") as k:
#             st.download_button('Download the Google-approved summary of this data', data = k, use_container_width=True, 
#                                mime = "application/vnd.google-apps.spreadsheet"
#                               )


d = data.loc[(data.location == location) & (data.role == role)]

# add ptrs and lats to session state
st.session_state['ptrs_lats'] = {
    ('ac', 'pi'): {'ptr': d['ptr_ac_pi'].values[0], 'lat': d['lat_ac_pi'].values[0]}, 
    ('ac', 'os'): {'ptr': d['ptr_ac_os'].values[0], 'lat': d['lat_ac_os'].values[0]}, 
    ('pi', 'os'): {'ptr': d['ptr_pi_os'].values[0], 'lat': d['lat_pi_os'].values[0]}, 
    ('os', 'hc'): {'ptr': d['ptr_os_hc'].values[0], 'lat': d['lat_os_hc'].values[0]}, 
    ('os', 'oe'): {'ptr': d['ptr_os_oe'].values[0], 'lat': d['lat_os_oe'].values[0]},
    ('hc', 'oe'): {'ptr': d['ptr_hc_oe'].values[0], 'lat': d['lat_hc_oe'].values[0]},
    ('oe', 'oa'): {'ptr': d['ptr_oe_oa'].values[0], 'lat': d['lat_oe_oa'].values[0]}
}


_, _, _, col7, _, _ = st.columns(6)

with col7: 
    with st.expander("""Project Greenlight\n##### OS : OE"""):
    
        os_oe_ptr = st.number_input('PTR',
                                    min_value=0.0, max_value=1.0,
                                    value=d['ptr_os_oe'].values[0], 
                                    label_visibility="visible", step = 0.01, key = 'os_oe_ptr')
        os_oe_lat = st.number_input('Latency (Weeks)',
                                    min_value=0, 
                                    value=int(d['lat_os_oe'].values[0] // 7), 
                                    label_visibility="visible", step = 1, key = 'os_oe_lat')

        
        subcol1, subcol2 = st.columns(2)
        with subcol1: 
            value = 0.30
            st.metric('Target PTR', value = value, delta = np.round(value - os_oe_ptr, 2))

        with subcol2: 
            lat_value = 4
            st.metric('Target Latency', value = lat_value, delta = lat_value - os_oe_lat)

col1, col2, col3, col4, col5, colm = st.columns(6)

with col1:
    with st.expander("""#### AC : PI"""): 

        ac_pi_ptr = st.number_input('PTR', 
                                    min_value=0.0, max_value=1.0 - st.session_state['ptrs_lats'][('ac', 'os')]['ptr'], 
                                    value=d['ptr_ac_pi'].values[0], 
                                    label_visibility="visible", step = 0.01, key = 'ac_pi_ptr')
        
        ac_pi_lat = st.number_input('Latency (Weeks)',
                                    min_value=0, 
                                    value=int(d['lat_ac_pi'].values[0] // 7), 
                                    label_visibility="visible", step = 1, key = 'ac_pi_lat')

        subcol1, subcol2 = st.columns(2)
        with subcol1: 
            value = 0.85
            st.metric('Target PTR', value = value, delta = np.round(value - ac_pi_ptr, 2))

        with subcol2: 
            lat_value = 4
            st.metric('Target Latency', value = lat_value, delta = lat_value - ac_pi_lat)

    
with col2:
    with st.expander("""#### PI : OS"""):
        pi_os_ptr = st.number_input('PTR',
                                    min_value=0.0, max_value=1.0,
                                    value=d['ptr_pi_os'].values[0], 
                                    label_visibility="visible", step = 0.01, key = 'pi_os_ptr')
        pi_os_lat = st.number_input('Latency (Weeks)',
                                    min_value=0, 
                                    value=int(d['lat_pi_os'].values[0] // 7), 
                                    label_visibility="visible", step = 1, key = 'pi_os_lat')

        subcol1, subcol2 = st.columns(2)
        with subcol1: 
            value = 0.30
            st.metric('Target PTR', value = value, delta = np.round(value - pi_os_ptr, 2))

        with subcol2: 
            lat_value = 4
            st.metric('Target Latency', value = lat_value, delta = lat_value - pi_os_lat)

with col3: 
    with st.expander("""#### OS : HC"""):
        os_hc_ptr = st.number_input('PTR',
                                    min_value=0.0, max_value=1.0, 
                                    value=d['ptr_os_hc'].values[0], 
                                    label_visibility="visible", step = 0.01, key = 'os_hc_ptr')
        os_hc_lat = st.number_input('Latency (Weeks)',
                                    min_value=0, 
                                    value=int(d['lat_os_hc'].values[0] // 7), 
                                    label_visibility="visible", step = 1, key = 'os_hc_lat')

        subcol1, subcol2 = st.columns(2)
        with subcol1: 
            value = 0.25
            st.metric('Target PTR', value = value, delta = np.round(value - os_hc_ptr, 2))

        with subcol2: 
            lat_value = 4
            st.metric('Target Latency', value = lat_value, delta = lat_value - os_hc_lat)

with col4: 
    with st.expander("""#### HC : OE"""):

        hc_oe_ptr = st.number_input('PTR',
                                    min_value=0.0, max_value=1.0, 
                                    value=d['ptr_hc_oe'].values[0],
                                    label_visibility="visible", step = 0.01, key = 'hc_oe_ptr')
        hc_oe_lat = st.number_input('Latency (Weeks)',
                                    min_value=0, 
                                    value=int(d['lat_hc_oe'].values[0] // 7), 
                                    label_visibility="visible", step = 1, key = 'hc_oe_lat')

        subcol1, subcol2 = st.columns(2)
        with subcol1: 
            value = 0.95
            st.metric('Target PTR', value = value, delta = np.round(value - hc_oe_ptr, 2))

        with subcol2: 
            lat_value = 0
            st.metric('Target Latency', value = lat_value, delta = lat_value - hc_oe_lat)

with col5: 
    with st.expander("""#### OE : OA"""):

        oe_oa_ptr = st.number_input('PTR',
                                    min_value=0.0, max_value=1.0, 
                                    value=d['ptr_oe_oa'].values[0], 
                                    label_visibility="visible", step = 0.01, key = 'oe_oa_ptr')
        oe_oa_lat = st.number_input('Latency (Weeks)', 
                                    min_value=0, value=int(d['lat_oe_oa'].values[0] // 7), 
                                    label_visibility="visible", step = 1, key = 'oe_oa_lat')

        subcol1, subcol2 = st.columns(2)
        with subcol1: 
            value = 0.95
            st.metric('Target PTR', value = value, delta = np.round(value - oe_oa_ptr, 2))

        with subcol2: 
            lat_value = 0
            st.metric('Target Latency', value = lat_value, delta = lat_value - oe_oa_lat)
            
with colm: 
    with st.expander("""#### PPR"""): 
        ppr = st.number_input('PPR', min_value=0, max_value=100, value=4, label_visibility="visible", step = 1, key = 'ppr')


        ppr_value = 4
        st.metric('Target PPR', value = ppr_value, delta = ppr - ppr_value)

_, col6, _, _, _, _ = st.columns(6)

with col6: 
    with st.expander("""Direct to Onsite\n##### AC : OS"""):
    
        ac_os_ptr = st.number_input('PTR', 
                                    min_value=0.0, max_value=1.0 - ac_pi_ptr, 
                                    value=d['ptr_ac_os'].values[0], 
                                    label_visibility="visible", step = 0.01, key = 'ac_os_ptr')
        
        
        ac_os_lat = st.number_input('Latency (Weeks)',
                                    min_value=0, 
                                    value=int(d['lat_ac_os'].values[0] // 7),
                                    label_visibility="visible", step = 1, key = 'ac_os_lat')

        subcol1, subcol2 = st.columns(2)
        with subcol1: 
            value = 0.30
            st.metric('Target PTR', value = value, delta = np.round(value - ac_os_ptr, 2))

        with subcol2: 
            lat_value = 4
            st.metric('Target Latency', value = lat_value, delta = lat_value - ac_os_lat)


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
print(d3)

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
        
    return p, (l/i)

ac_oa_ptr, ac_oa_lat = get_path_attrs(G, 'ac', 'oa')
pi_oa_ptr, pi_oa_lat = get_path_attrs(G, 'pi', 'oa')
os_oa_ptr, os_oa_lat = get_path_attrs(G, 'os', 'oa')
hc_oa_ptr, hc_oa_lat = get_path_attrs(G, 'hc', 'oa')
oe_oa_ptr, oe_oa_lat = get_path_attrs(G, 'oe', 'oa')

ac_ppr = (d2[d2.variable == 'ppr']['value'] / ac_oa_ptr).values[0] / 4.6

    
st.sidebar.markdown(f"""
### Simulation Adjustments
Select a starting date and targets for the simulation
""")


start_date_ = st.sidebar.date_input('Start Date', 
                           value=pd.to_datetime('today'),
                           max_value=end_date
)

start_date = start_date_ - pd.Timedelta(start_date_.weekday(), unit = 'D')


with st.sidebar.expander("Targets"): 

    cola, colb, colc = st.columns(3)

    with cola: 
        st.markdown("Target 1")

    with colb: 
        change_date0 = st.date_input('Target Date',
                                     value=max([pd.Timestamp('2024/03/25').date(), start_date + pd.Timedelta(ac_oa_lat * 2, unit = 'W')]), 
                                     min_value=start_date + pd.Timedelta(ac_oa_lat, unit = 'W'), 
                                     max_value=end_date
        )
    with colc: 
        change_amount0 = st.number_input('OA Goal', min_value=0.0, max_value=10000., value=200.0,
                                        label_visibility="visible")
    exec(render_elem())
    st.session_state.change_dates[1] = {'date': change_date0, 'quantity': change_amount0, 'event': 'oa'}
    record_change_dates()
    count_reached = st.session_state.count > 2
    add_intervention = st.button('Add Target', use_container_width=True, on_click = increment_count, disabled = count_reached)
    reset_interventions = st.button('Reset', use_container_width=True, on_click = reset_change_dates)

def calc_goals_split(goal_row, G, other_goals):
    oa_goal = goal_row['quantity']
    i = goal_row['goal_order']
    Paths = []
    path_weights = []
    path_lengths = []
    for path in nx.all_simple_paths(G, 'oa', 'ac', cutoff=None): 
        Paths.append(path)
        path_weights.append(np.exp(nx.path_weight(G, path, 'ptr_log')))
        path_lengths.append(nx.path_weight(G, path, 'lat'))
    path_weights = np.array(path_weights)    
    normalized_path_weights = path_weights / np.sum(path_weights)

    for j, (p, w, l) in enumerate(zip(Paths, normalized_path_weights, path_lengths)): 
        mini_path = p
        individual_oa_goal = np.ceil(w * 200)
        individual_deadline = goal_row['date']
        goal = individual_oa_goal
        deadline = individual_deadline
        other_goals.append({'date': deadline, 'quantity':  goal, 'event' : 'oa', 'goal_order': i, 'path': j})
        for x, y in zip(mini_path[:-1], mini_path[1:]): 
            edge = G.edges[y, x]
            pt = edge['ptr']
            lt = edge['lat']
            goal = np.ceil(goal / pt)
            deadline -= pd.Timedelta(lt, unit = 'W')
            if y == 'ac': 
                
                other_goals.append({'date': deadline, 'quantity':  goal, 'event' : y, 'ac_oa_lat': l, 'goal_order': i, 'path': j})
            else: 
                other_goals.append({'date': deadline, 'quantity':  goal, 'event' : y, 'goal_order': i, 'path': j})

    
def det_nec_cap(change_dates, G): 
    goals = pd.DataFrame(change_dates)
    goals['goal_order'] = goals.date.rank(ascending = True) - 1
    other_goals = []
    _ = goals.apply(lambda row : calc_goals_split(row, G, other_goals), axis = 1)
    other_goals = pd.DataFrame(other_goals)
    deadlines = other_goals.loc[other_goals.event == 'ac'][['date', 'ac_oa_lat', 'goal_order', 'path']]
    # durrs = get_duration(start_date, deadline, ac_oa_lat)
    deadlines['durr'] = (deadlines.date - start_date).dt.days // 7
    deadlines['durr'] = deadlines[['durr', 'ac_oa_lat']].max(axis = 1)
    
    
    dl = other_goals.merge(deadlines[['goal_order', 'durr', 'path']], on = ['goal_order', 'path']).sort_values(['goal_order', 'event'])
    L = []
    _ = dl.apply(realize, L = L, axis = 1)
    
    dll = pd.concat(L).groupby(['date', 'event'], as_index = False).sum()
    dll2 = dll.pivot(index = ['date'], columns = 'event', values = 'quantity')
    dll2 = dll2.fillna(method = 'ffill')
#     L = []
#     for i, durr in enumerate(durrs[::-1]): 
#         dl.loc[dl.goal_order == i, 'durr'] = durr

    
    return dll2, dll, dl

    

det_nec_cap(st.session_state['change_dates'].values(), G)    

dll2, dll, dl = det_nec_cap(st.session_state['change_dates'].values(), G)  

target_capacity_ac = dll2['ac']
target_capacity_ac = target_capacity_ac[target_capacity_ac > 0]
act = get_initials(location, role, actuals = data)
ppr_list = [n * ac_ppr for n in [0.25, 0.5, 0.75, 1]]
ramping_function_rec = lambda n : ramping_function(n, ppr_list = ppr_list)

LL_agg, LL_agg_loc = Simulation(target_capacity_ac, ramping_function = ramping_function_rec, attrition_model = attrition_model)
st.markdown('-----')

# col1, col2 = st.columns(2)
# with col1:
#     st.markdown("""
#     ### Deadlines
#     """)

#     fig = px.bar(dl.sort_values(['event', 'date'], ascending = False), x = 'date', y = 'quantity', color = 'event')
#     st.plotly_chart(fig, use_container_width=True)
# with col2: 
#     st.markdown(f"""
#     ### Required Weekly Yield 
#     """)

#     fig = px.line(dll.sort_values(['event', 'date'], ascending = False), y = 'quantity', x = 'date', color = 'event', line_shape = 'hv')
#     st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<h3 style='text-align: center;'>Estimated Required Headcount Over Time</h3>
""", unsafe_allow_html=True)



st.markdown("""
#### Recruiters/Sourcers
""")

tab1, tab2 = st.tabs(['Raw Headcount', 'Broken Down by Experience Level'])

with tab1: 
    fig = px.bar(LL_agg_loc, x = 'date', y = 'total_team_size')
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Estimated Required Headcount"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2: 
#         st.markdown("""
#         ### Estimated Headcount Broken Down by Productivity
#         """)
    fig = px.bar(LL_agg_loc, x = 'date', y = [0, 1, 2, 3, '5+'])
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Headcount"
    )
    st.plotly_chart(fig, use_container_width=True)

def markov(r, G, caps): 
    cap = r.global_capacity
    date = r.date
    caps.append({'event': 'ac', 'quantity':  cap, 'date': date})
    for path in nx.all_simple_paths(G, 'ac', 'oa', cutoff=None): 
        cap = r.global_capacity
        date = r.date
        for x, y in zip(path[:-1], path[1:]): 
            edge = G.edges[(x, y)]
            ptr = edge['ptr']
            lat = edge['lat']
            
            cap *= ptr
            date += pd.Timedelta(lat, unit = 'W')
            caps.append({'event': y, 'quantity':  cap ,'date': date})
    
LL_agg_loc = LL_agg_loc[['date', 'global_capacity']].assign(
    oas_achieved = (LL_agg_loc.global_capacity * ac_oa_ptr).shift(int(ac_oa_lat))
)
caps = []
LL_agg_loc.apply(lambda r : markov(r, G, caps), axis = 1)
T = pd.DataFrame(caps).groupby(['date', 'event']).median().reset_index()
T = T.pivot('date', 'event', 'quantity').reset_index()
K = target_capacity_ac.cumsum().reset_index()

dll2 = dll2.reset_index()

st.markdown("""
<h3 style='text-align: center;'>Modeled Offers Accepted vs. Target Offers Accepted</h3>
""", unsafe_allow_html=True)

dll2['Cummulative Offers Accepted'] = dll2['oa'].cumsum()
LL_agg_loc['Estimated Cummulative Offers Accepted'] = LL_agg_loc['oas_achieved'].cumsum()
fig = px.line(LL_agg_loc, x = 'date', y = 'Estimated Cummulative Offers Accepted', line_shape = 'hv')
fig['data'][0]['showlegend'] = True
fig['data'][0]['name'] = 'Estimated Cummulative Offers Accepted'


fig.add_trace(
    go.Scatter(x = dll2.date, y = dll2['Cummulative Offers Accepted'], mode='lines', name = 'Target Weekly Offers Accepted', line=dict(
        shape='hv'
    ), 
              line_color='black')
)
# fig.update_yaxes(range=[0, 2.0 * initial_headcount])
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cummulative Offers Accepted"
)
st.plotly_chart(fig, use_container_width=True)