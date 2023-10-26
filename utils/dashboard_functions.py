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
        (actuals.location == location) & (actuals.role.isin(role))
    ]
    
    return act
    
def summarize_data(l): 
    pass
#     wb = openpyxl.load_workbook("./models/VOVO_output_template.xlsx")
#     ws = wb['NA Transition Plan']
    
#     l['Month'] =  .to_datetime(l.date) +  pd.offsets.MonthBegin(-1)
#     l = l .loc[(l.Month >= pd.to_datetime('2023/06/01')) & (l.Month <= pd.to_datetime('2024/04/01'))]
#     li = l.groupby(['Role', 'Month'], as_index = False)['optimal_starting'].sum()
    
#     li = li.loc[(li.Month >= pd.to_datetime('2023/06/01')) & (li.Month <= pd.to_datetime('2024/04/01'))]
#     li_pivot = li.pivot(index = 'Role', columns = 'Month', values = 'optimal_starting')
    
#     l['mo_week'] = l.groupby(['Role', 'Month']).rank()['date']
#     lj = l.loc[l.mo_week == 1.]
#     lj['prev_mo_remote'] = lj.groupby('Role').remote_headcount.shift(1)
#     lj = lj.assign(
#         remote_depreciation = -1*(lj['prev_mo_remote'] - lj['remote_headcount'])
#     )
#     remote_hc_ramp_down = lj.groupby('Month')['remote_depreciation'].sum().values
    
#     starters = lj.loc[lj.date == lj.date.min()]
#     enders = lj.loc[lj.date == lj.date.max()]
    
#     for i,row in enumerate(li_pivot.iterrows()): 
#         ws[f'A{i+5}'] = row[0]
#         ws[f'B{i+5}'] = starters.loc[starters.Role == row[0]]['Total Headcount'].values[0]
#         ws[f'C{i+5}'] = starters.loc[starters.Role == row[0]]['vovo_headcount'].values[0]
#         ws[f'D{i+5}'] = starters.loc[starters.Role == row[0]]['remote_headcount'].values[0]
        
        
    
#     ws[f'P{i+5}'] = enders.loc[enders.Role == row[0]]['vovo_headcount'].values[0]
#     # remote_hc_ramp_down = lj.loc[lj.Role == row[0]]['remote_depreciation'].values
#     for col, vovo_val, remo_val in zip('EFGHIJKLMNO', row[1].values, remote_hc_ramp_down): 
#         ws[col+f'{i+5}'] = vovo_val
#         ws[col+'13'] = remo_val
        
#     wb.save('./vovo_output.xlsx')
        
#     return 

def calc_goals(d3, goal, deadline, event, other_goals = []): 
    st.write(event, other_goals)
    n = d3[d3['y'] == event]
    val, l = n['stats'].values[0]
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
    # return df, initial_headcount