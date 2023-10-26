import pandas as pd
import numpy as np
import secrets


candidates = pd.read_csv('co_model - General_in_progress.csv')

candidates['application_create_week'] = pd.to_datetime(candidates['application_create_week'])

# we had to aggregate the data so that it could be brought into sheets.
duplicate_rows = []

for i, j in candidates.iterrows(): 
    for k in range(j['N']): 
        duplicate_rows.append(j.to_dict())
        
all_candidates = pd.DataFrame(duplicate_rows)

all_candidates['n_weeks'] = (all_candidates['application_closed_week'] + 1).apply(np.arange)
all_candidates['candidate_pid'] = all_candidates.apply(lambda row: secrets.token_hex(8), axis = 1) # assign a unique candidate identifier


if __name__ == "__main__": 
    all_candidates.to_csv('in_progress_pipeline.csv')
    

# interm_events = all_candidates[['candidate_pid', 'phone_interview', 'onsite_interview']]
# terminal_events = all_candidates.pivot('candidate_pid', 'current_outcome', 'application_closed_week').reset_index()

# events_flags = terminal_events#.merge(interm_events, on = 'candidate_pid')

# events = events_flags.melt(id_vars = ['candidate_pid'], var_name = 'event', value_name = 'n_weeks')
# outcomes = all_candidates[['rpo_group', 'application_create_week', 'candidate_pid', 'current_outcome','application_closed_week', 'n_weeks']].explode('n_weeks')

# candidates_long = events.merge(outcomes, on = ['candidate_pid', 'n_weeks'], how = 'right')

