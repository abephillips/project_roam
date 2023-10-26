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

attrition_model = joblib.load("models/attrition_model.joblib")
reason_function = joblib.load("models/reason_function.joblib")
regrettable_function = joblib.load("models/regrettable_function.joblib")



# attriton online
def attrtion_forcast(N, start_date = pd.to_datetime('01/01/2024')):
    L = []
    n_weeks = 0
    date = start_date
    month = date.month
    attrit = 0
    while N > 0:
        li = {'n_weeks': n_weeks, 'month': month, 'date': date, 'Headcount': N, 'Attrit': attrit}
        L.append(li)

        n_weeks += 1 # increase week
        date += pd.Timedelta(1, unit = 'W')
        month = date.month

        l = [[n_weeks, month]]
        p = attrition_model.predict_proba(l)[0]

        leaving = binom.rvs(N, p)[0]
        attrit = N - leaving
        
        c = reason_function.predict_proba(np.array([month, ]).reshape(-1,1).tolist())
        d = regrettable_function.predict_proba(np.array([month, ]).reshape(-1,1).tolist())

        reason_spread =  multinomial.rvs(n = attrit, p = c[0])
        regrettable_spread = multinomial.rvs(n = attrit, p = d[0])
        
        N = leaving

    return pd.DataFrame(L)



# first row
col1, col2  = st.sidebar.columns(2)

with col1:
    start_date = st.date_input('Start Date', label_visibility="visible")

with col2:
    n_starting = st.number_input(f'Number Starting', key  = f'cap', min_value=0, max_value=10000, value=100, step=10)

LL  = []
for _ in range(1):
    L = attrtion_forcast(n_starting, start_date)
    LL.append(L)


D = pd.concat(LL)
D = D.groupby('date', as_index = False).agg(
    HC_mean = ('Headcount', 'mean'),
    HC_std = ('Headcount', 'std')
)

D = D.assign(
    HC_min = D.HC_mean - D.HC_std,
    HC_max = np.clip(D.HC_mean + D.HC_std, a_min = 0, a_max=None)
)

st.line_chart(data = D, x = 'date', y = 'HC_mean')

# sns.lineplot(data = D, x = 'date', y= 'HC_mean', ax = ax)
# ax.fill_between(x = D.date, y1 = D.HC_max, y2 = D.HC_min, alpha = 0.3)
# ax.set_ylabel('Headcount')
# ax.set_xlabel('Date')

# ax.set_title('Headcount of a Team Starting Production Today')