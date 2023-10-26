##### streamlit app

import streamlit as st
import time
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.ROAM import *
import random

# load, no need to initialize the loaded_rf
clf2_pure = joblib.load("models/general_co_model.joblib")

prd = pd.date_range(start = pd.to_datetime('01/01/2023'), end = pd.to_datetime('12/31/2023'), freq = 'W')
sourcing_list = pd.DataFrame(
    data = {
        'date': prd, 'n_acs': 0
        }
)
sourcing_list.loc[0, 'n_acs'] = 1e6

os_capacity_list = np.ones(len(sourcing_list))
os_capacity_list[:] = 1e6 # infinite capacity until change date

tps_capacity_list = os_capacity_list.copy()
tps_capacity_list[:] = 1e6



@st.cache_data
def Simulation(job, location, sourcing_list=sourcing_list, os_capacity_list=os_capacity_list, tps_capacity_list=tps_capacity_list):


    OA = simple_scheduler(
      individual_oa_model = clf2_pure,
      feature_list = ['SWE', 'non-SWE', 'APAC', 'Americas', 'EMEA', 'week'],
      sourcing_list = sourcing_list, job = job, location = location,
      os_interview_capacity = os_capacity_list,
      tps_interview_capacity = tps_capacity_list,
      sim_duration=len(sourcing_list)
    )

    OA_df = pd.DataFrame(OA)

    accepted = OA_df.groupby(['wall time'], as_index = False).agg(
      accepted = ('accepted', 'sum'),
      phone = ('phone', 'sum'),
      phone_backlog = ('phone_backlog', 'mean')
    )
    
    sums = accepted.sum()

    accepted = accepted.assign(
      accepted_perc = accepted.accepted/ sums.accepted,
      phone_perc = accepted.phone / sums.phone
    )
    
    return accepted

st.markdown("# ROAM 🎈")
st.markdown("**R**ecruitment **O**ddysey **A**daptive **M**odel")
st.sidebar.markdown("""
# ROAM 🎈
Put brief directions here.
""")
location = st.sidebar.selectbox(
    'Location',
    ('APAC','Americas', 'EMEA')
)

job = st.sidebar.selectbox(
    'Job',
    ('SWE', 'non-SWE')
)

accepted = Simulation(job, location)

accepted = accepted.rename(columns = {'accepted_perc': 'Accepted', 
                                      'phone_perc': 'Phone Interviews', 
                                      'phone_backlog': 'Phone Interview Backlog', 
                                      'date': 'Date'
                                     })

st.bar_chart(data=accepted, y='Accepted', x='wall time', use_container_width=True)

st.bar_chart(data=accepted, y = ['Phone Interviews'], x = 'wall time') 