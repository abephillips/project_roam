import random
import statistics
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functools import reduce

from scipy.stats import gamma, poisson, randint, binom, bernoulli, multinomial, norm, expon
import random, string

class Event(simpy.Container):
    def __init__(self, env):
        self.env = env
        super().__init__(env)
    def _put(self, amount):
        if amount > 0:
            yield self.put(amount)
            
    def _get(self, amount):
        if amount > 0:
            yield self.get(amount)




class RecruitmentEvent(object):
    def __init__(self, env, capacity):
        self.env = env
        self.capacity = capacity
        self.OS_interview_queue = Event(env)
        self.OS_interview = Event(env)
    
    def put(self, amount):

        #### action 1: if there is *still* room in the interview room, take some from the backlog until it is full.

        # How much room is in the interview room?
        room_available  = self.capacity[self.env.now] - self.level()

        # How many people are in the queue?
        queue = self.queue()

        # take the maximum number of people in the queue that can fit into the interview
        from_queue = np.min([room_available, queue])


        if from_queue > 0:
            yield self.OS_interview_queue.get(from_queue)
            
            yield self.OS_interview.put(from_queue)

        #### action 2: if there is room in the interview room, fill it up.

        # How much room is in the interview room now?
        room_available  = room_available - from_queue

        to_room = np.min([room_available, amount - from_queue])

        if to_room > 0:
            # If there's room in the interview room, put maximum available many canidates in that room
            yield self.OS_interview.put(to_room)

        to_queue = amount - from_queue - to_room

        if to_queue > 0:
            # If there's any left over, put them in the queue
            yield self.OS_interview_queue.put(to_queue)
    
    def get(self, amount):
    # release candidates

        if amount > 0:
            yield self.OS_interview.get(amount)


    def level(self):
        return self.OS_interview.level

    def queue(self):
        return self.OS_interview_queue.level


    
class Scheduling(object):
    """
    Here we have a new class for the scheduling and hiring process
    The scheduler has the job of connecting candidates with interviewers, so in this scenerio, a sucessful candidate goes through a technical phone screen (TPS),
    gets an onsight interview (OS), and accepts a job offer (OA). Time here is displayed in weeks, so the timeout for each of these events is 1 week.
    """
    def __init__(self,
                 env,
                 scheduler_capacity,
                 os_interview_capacity,
                 phone_interview_capacity,
                 clf_dict
                 ):

        self.env = env
        self.OS_interview = RecruitmentEvent(env, os_interview_capacity) # infinite capacity -- as many people as possible can be waiting for an interview
        self.Phone_interview = RecruitmentEvent(env, phone_interview_capacity)


def google_recruitment(env, 
                       pipeline, 
                       origin_date, 
                       recruitment, 
                       outcome, 
                       clf, classes, features_in,
                       records):
    """
    This is the google recruitment simulation. A lot of the passthrough rates and latencies need to be looked at.
    TODO: Make it so the probabilities are accepted as an argument here.
    """
    date = origin_date
    X = pipeline
    X['month'] = date.month
    X['init_weeks'] = X['n_weeks']
    
    # record = len(X.loc[X.event == outcome])
    X_out = X.copy()


    while True:        
        record = len(X_out.loc[X_out.event == outcome])

        records.loc[records.date == date, 'in_progress'] += len(X)
        records.loc[records.date == date, outcome] += record
        if len(X) > 0: 

            date += pd.Timedelta(1, unit = 'W')


            X['N'] = 1 # a count variable, for when we aggregate similar rows and take samples from the multinomial dist. 
            X['n_weeks'] = X['init_weeks'] + env.now 
            X['month'] = date.month
            X_out, X = sequence_classification(X, clf, outcome)

            # env.process(
            #     recruitment.OS_interview.put(os_interviewed)
            # )
            # env.process(
            #     recruitment.Phone_interview.put(phone_interviewed)
            # )

            # os_interviewed = recruitment.OS_interview.level()

            # phone_interviewed = recruitment.Phone_interview.level()

            # env.process(
            #     recruitment.OS_interview.get(os_interviewed)
            # )
            # env.process(
            #     recruitment.Phone_interview.get(phone_interviewed)
            # )

            #X = X_new.copy()
        else: 
            break


        yield env.timeout(1)


def run_recruitment(env,
                    scheduler_capacity,
                    os_interview_capacity,
                    tps_interview_capacity,
                    pipeline, records,
                    clf_dict, classes, features_in, 
                   ):

    scheduling = Scheduling(env, scheduler_capacity, os_interview_capacity, tps_interview_capacity, clf_dict)
    ACCEPTED = 0
    for date, P in pipeline.groupby('date'): 
        for outcome in pipeline.current_outcome.unique():
            clf = clf_dict[outcome]
            P_outcome = P.loc[P.current_outcome == outcome]
            if outcome == 'ACCEPTED': 
                ACCEPTED += len(P_outcome)
                print('N ACCEPTED ', ACCEPTED)
            env.process(google_recruitment(env, pipeline = P_outcome, origin_date = date, recruitment = scheduling, outcome = outcome, 
                                             clf = clf, classes = classes, features_in = features_in, records = records))
        yield env.timeout(1)


def simple_scheduler(os_interview_capacity,
                     tps_interview_capacity,
        scheduler_capacity = 1e6, # capacity is set to 1e6 to illistrate that the default scheduler has no capacity limitations.
        sim_duration = 10, features_in = None, 
        clf_dict= None, pipeline = None, classes = [], records = []
        ):



    env = simpy.Environment()
    env.process(
        run_recruitment(
            env,
            scheduler_capacity,
            os_interview_capacity,
            tps_interview_capacity,
            pipeline, 
            records,
            clf_dict = clf_dict,
            classes = classes, features_in = list(features_in), 
            )
        )
    env.run(until = sim_duration)
    return records

def sort_into_events(row):
    A = []
    i = 0
    for j in row['gr']: 
        A.append(row['cands'][int(i):int(i+j)])
        i += j
    return A

def sequence_classification(X, clf, outcome):
    
    features_in = clf.feature_names_in_
    classes = [outcome, 'in_progress']
    
    X_model_in = X.groupby(list(features_in), as_index = False).agg(
        N = ('N', 'sum'), 
        candidate_pid = ('candidate_pid', 'unique')
    )
    
    # multinomial(row['N'], p = np.absolute(row['ps'])).rvs()

    # X_model_in['event'] = [classes]*len(X_model_in)
    X_model_in['ps'] = list(clf.predict_proba(X_model_in[features_in]))
    X_model_in['event'] = X_model_in.apply(lambda row : np.random.choice(classes, size = row['N'], p = row['ps']), axis= 1)# row['N'], p = np.absolute(row['ps'])).rvs(), axis = 1)
    # X_model_in['event'] = X_model_in.apply(sort_into_events, axis = 1 )
    X_new = X_model_in[['event', 'candidate_pid', 'n_weeks']].explode(['event', 'candidate_pid']).explode('candidate_pid')
    X_new = pd.concat([X_new[['candidate_pid', 'event']], pd.get_dummies(X_new.event)], axis = 1)
    X_out = X_new.loc[X_new.event == outcome]

    removed_candidates = X_out.candidate_pid.unique()

    X_in = X[['init_weeks', 'candidate_pid']].merge(
        X_new.loc[~X_new.candidate_pid.isin(removed_candidates)], 
        on = ['candidate_pid'], suffixes = ["_prev", "_curr"]
    )
    return X_out, X_in
