from fastapi import FastAPI
import time
import joblib
import simpy
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import openpyxl
import tempfile
import random

from utils.helpers import *

attrition_model = joblib.load("./models/attrition_model.joblib")

app = FastAPI()

def parse_csv(df):
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    return parsed

# temporaty! define ramping function
def ramping_function(week, ppr_list): 
    for w, tickets in enumerate(ppr_list): 
        if week == w: 
            return tickets
    return ppr_list[-1]

ppr_list = [13, 26, 39, 65]
ramping_function_sched = lambda n : ramping_function(n, ppr_list = ppr_list)


def Simulation(target_capacity, 
               ramping_function = ramping_function_sched,
               attrition_model = attrition_model):
    
    start_date = target_capacity.index.min()
    end_date = target_capacity.index.max()
    target_capacity = target_capacity.values
    N = np.ceil(target_capacity[0] / ramping_function(5))
    mode = 'capacity'
    
    n_weeks = 12

    sim_len = (end_date - start_date).days // 7    
    record = np.zeros((sim_len, n_weeks + sim_len + 2))

    env = simpy.Environment()
    team = Team(env, attrition_model, ramping_function)
    
    env.process(
        sustain(env = env, N = N, record = record, start_date = start_date, end_date = end_date, n_weeks = n_weeks, target_capacity = target_capacity,
                team = team, mode = mode)
    )
    env.run(sim_len)
    
    rf = np.vectorize(ramping_function)
    period = pd.date_range(start_date, end_date, freq = 'W', inclusive = 'left')
    df = pd.DataFrame(record)
    df = df.assign(
        global_capacity = np.dot(df.iloc[:, :-2].values, rf(df.iloc[:, :-2].columns).T),
        global_headcount = (df.iloc[:, -2] + df.iloc[:, -1]), 
        vovo_headcount = df.iloc[:, -2], 
        remote_headcount = df.iloc[:, -1], 
        date = period
    )
    
    r, c = record.shape
    df['weekly_attrition'] = df[0]

    LL_agg = df.copy()

    LL_agg['month'] = LL_agg.date.dt.month
    LL_agg['Year'] = LL_agg.date.dt.year
    LL_agg['Quarter'] = np.nan
    LL_agg.loc[LL_agg.month == 1, 'Quarter'] = 1
    LL_agg.loc[LL_agg.month == 4, 'Quarter'] = 2
    LL_agg.loc[LL_agg.month == 7, 'Quarter'] = 3
    LL_agg.loc[LL_agg.month == 10, 'Quarter'] = 4

    LL_agg = LL_agg.fillna(method = 'ffill')
    LL_agg['last_q'] = LL_agg['Quarter'].shift(1)

    LL_agg['cum_tix'] = LL_agg.groupby(['Quarter', 'Year']).global_capacity.cumsum()
    LL_agg_loc = LL_agg.copy()

    LL_agg_loc['total_team_size'] = LL_agg_loc.loc[:, 0:c-1].sum(axis = 1)
    LL_agg_loc['5+'] = LL_agg_loc.loc[:, 4:c-1].sum(axis = 1)
    LL_agg_loc['date'] = pd.to_datetime(LL_agg_loc['date']).dt.date

    return LL_agg, LL_agg_loc

app = FastAPI()

@app.get("/")
async def root():
    target_capacity = pd.DataFrame(index = pd.date_range('07/03/2023', '12/26/2023', freq = 'W'))
    target_capacity['values'] = 3000
    
    LL_agg, LL_agg_loc = Simulation(target_capacity)
    return {"message": parse_csv(LL_agg)}