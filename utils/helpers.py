##### streamlit app

import streamlit as st
import time
import joblib
import simpy

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import openpyxl
import tempfile

# from models.VOVO import *
import random

class Team: 
    def __init__(self, env, attrition_model, ramping_function): 
        self.Headcount = simpy.Container(env)
        self.Capacity = simpy.Container(env)
        self.attrition_model = attrition_model
        self.ramping_function = ramping_function
        
        
def attrit(env, N, record, date, team, n_weeks, init): 

    month = date.month
    if init == 1:
        pass
    else:
        team.Headcount.put(N)
        team.Capacity.put(team.ramping_function(n_weeks) * N)
    record[env.now, [n_weeks, -2]] += N
    while N > 0:

        n_weeks += 1 # increase week
        date += pd.Timedelta(1, unit = 'W')
        month = date.month


        l = [[n_weeks, month]]
        p = team.attrition_model.predict_proba(l)[0]

        N_new = int(N*p[0])#binom.rvs(N, p)[0]
        leaving = N - N_new
        staying = N_new
        N = N_new

        if leaving > 0:
            team.Headcount.get(leaving)
            team.Capacity.get(team.ramping_function(n_weeks - 1) * leaving)

        if staying > 0: 
            team.Capacity.get(team.ramping_function(n_weeks - 1) * staying)
            team.Capacity.put(team.ramping_function(n_weeks) * staying)

        yield env.timeout(1)

        record[env.now, [n_weeks, -2]] += N

                
def sustain(env, N, record, start_date, end_date, n_weeks, target_capacity, mode, team):   

    date = start_date
    month = date.month
    initial_capacity = team.ramping_function(n_weeks) * (N)
    print(initial_capacity)
    initial_headcount = N
    # N = initial_headcount

    team.Headcount.put(initial_headcount)
    record[env.now, [n_weeks-1, -1]] += N

    team.Capacity.put(initial_capacity)
    init = 0


    while True: 

        team_size = team.Headcount.level
        current_capacity = team.Capacity.level 
        capacity_difference = target_capacity[env.now] - current_capacity



        if mode == 'capacity': 

            new_weeks = 3

            new_folks = capacity_difference//team.ramping_function(new_weeks)
            if new_folks > 0:
                record[env.now - 3, [0, -2]] += new_folks
                record[env.now - 2, [1, -2]] += new_folks
                record[env.now - 1, [2, -2]] += new_folks

        elif mode == 'headcount': 
            new_folks = target_headcount[env.now] - team_size

            new_weeks = 0
        else: 
            raise ValueError('Not a valid mode')

        if new_folks > 0: 
            init += 2
            env.process(
                attrit(env, new_folks, record, date, team, new_weeks, init)
            )

        n_weeks += 1
        date += pd.Timedelta(1, unit = 'W')
        month = date.month  


        l = [[n_weeks, month]]
        p = team.attrition_model.predict_proba(l)[0]

        N_new = int(N*p[0])#binom.rvs(N, p)[0]

        additional_loss = 0
#         if len(effective_changes) > 0: 
#             cohort = N_new
#             for i, row in effective_changes.iterrows():
#                 change_amount = team_size - target_headcount[env.now]
#                 layoff = np.clip(int(change_amount), a_min = 0, a_max = cohort)
#                 cohort = cohort - layoff
#                 additional_loss += layoff

        leaving = (N - N_new) + additional_loss
        staying = N_new - additional_loss
        N = staying

        if leaving > 0:
            team.Headcount.get(leaving)
            team.Capacity.get(team.ramping_function(n_weeks-1) * leaving)

        if staying > 0: 
            team.Capacity.get(team.ramping_function(n_weeks - 1) * staying)
            team.Capacity.put(team.ramping_function(n_weeks) * staying)


        yield env.timeout(1)
        # record the headcount of this wave
        record[env.now, [n_weeks, -1]] += N

# temporaty! define ramping function
def ramping_function(week, ppr_list): 
    for w, tickets in enumerate(ppr_list): 
        if week == w: 
            return tickets
    return ppr_list[-1]

def location_roles(location, actuals): 
    roles = actuals.loc[actuals.location == location].role.unique()
    return tuple(roles)

def get_initials(location, role, actuals): 
    act = actuals.loc[
        (actuals.location == location) & (actuals.role == role)
    ]
    
    return act
    
        
def calc_goals(d3, goal, deadline, event, other_goals = []): 

    n = d3[d3['y'] == event]
    l, val = n['stats'].values[0]
    goal /= val
    deadline -= pd.to_timedelta(l, unit = 'W')
    
    other_goals.append({'date': deadline, 'quantity': np.ceil(goal), 'event': n['x'].values[0]})
    
    try: 
        return calc_goals(d3, goal, deadline, n['x'].values[0], other_goals)
    except IndexError: 
        return other_goals
    
def get_duration(todays_date, deadlines, ac_oa_lat): 
    durrs = []
    for deadline in deadlines: 
        durr = (deadline - todays_date).days // 7
        durrs.append(min([durr, ac_oa_lat]))
    return durrs


def realize(row, L):
    dr = pd.date_range(row['date'] - pd.Timedelta(row['durr'] + 1, unit = 'W'), row['date'], freq = 'W')
    e = np.ceil(row['quantity'] / len(dr))
    cl = row['event']

    df = pd.DataFrame()
    df['date'] = dr
    df['quantity'] = e
    df['event'] = cl

    L.append(df)
    return

# @st.cache_resource
def Simulation(target_capacity, 
               ramping_function,
               attrition_model):
    
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
        sustain(env, N, record, start_date, end_date, n_weeks, target_capacity = target_capacity,
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

#     LL_agg['month'] = LL_agg.date.dt.month
#     LL_agg['Year'] = LL_agg.date.dt.year
#     LL_agg['Quarter'] = np.nan
#     LL_agg.loc[LL_agg.month == 1, 'Quarter'] = 1
#     LL_agg.loc[LL_agg.month == 4, 'Quarter'] = 2
#     LL_agg.loc[LL_agg.month == 7, 'Quarter'] = 3
#     LL_agg.loc[LL_agg.month == 10, 'Quarter'] = 4

#     LL_agg = LL_agg.fillna(method = 'ffill')
#     LL_agg['last_q'] = LL_agg['Quarter'].shift(1)

    LL_agg['cum_tix'] = LL_agg.global_capacity.cumsum()
    LL_agg_loc = LL_agg.copy()

    LL_agg_loc['total_team_size'] = LL_agg_loc.loc[:, 0:c-1].sum(axis = 1)
    LL_agg_loc['5+'] = LL_agg_loc.loc[:, 4:c-1].sum(axis = 1)
    LL_agg_loc['date'] = pd.to_datetime(LL_agg_loc['date']).dt.date

    return LL_agg, LL_agg_loc
