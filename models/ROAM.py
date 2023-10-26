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
            self.get(amount)




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
          # If there's room in the interview room, put maximum available many canidates in that ro
            yield self.OS_interview.put(to_room)

        to_queue = amount - from_queue - to_room

        if to_queue > 0:
          # If there's any left over, put them in the queue
            yield self.OS_interview_queue.put(to_queue)


    def get(self, amount):
        # release candidates

        if amount > 0:
            yield self.OS_interview.get(amount)
            
    
    def decay(self, p):
        in_queue = self.queue()
        leaving = binom.rvs(int(in_queue), p = p)
        if leaving > 0:
            yield self.OS_interview_queue.get(leaving)
        
        


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
                 individual_oa_model
                 ):

        self.env = env
        self.OS_interview = RecruitmentEvent(env, os_interview_capacity) # infinite capacity -- as many people as possible can be waiting for an interview
        self.Phone_interview = RecruitmentEvent(env, phone_interview_capacity)


def google_recruitment(env, tickets, recruitment, individual_oa_model, records, i, pred_features):
    """
    This is the google recruitment simulation. A lot of the passthrough rates and latencies need to be looked at.
    TODO: Make it so the probabilities are accepted as an argument here.
    """
    date = tickets['date']
    tickets = tickets['n_acs']

    accepted = 0
    rejected = 0
    phone_interviewed = 0
    os_interviewed = 0

    in_progress = tickets

    while True:
        
        in_progress = in_progress + phone_interviewed + os_interviewed


        l = pred_features.copy()
        l[-1] = env.now - i
        p = individual_oa_model.predict_proba(np.array(l).reshape(1,-1).tolist())
        
        # 'ACCEPTED', 'IN_PROGRESS', 'ONSITE', 'PHONE', 'REJECTED', 'WITHDRAWN'
        accepted, in_progress, os_interviewed, phone_interviewed, rejected, withdrawn = multinomial.rvs(in_progress, p = p[0]) # markov chain!'

        #in_progress = in_progress + phone_interviewed + os_interviewed


        env.process(
          recruitment.OS_interview.put(os_interviewed)
        )
        env.process(
          recruitment.Phone_interview.put(phone_interviewed)
        )

        os_interviewed = recruitment.OS_interview.level()

        phone_interviewed = recruitment.Phone_interview.level()

        env.process(
          recruitment.OS_interview.get(os_interviewed)
        )
        env.process(
          recruitment.Phone_interview.get(phone_interviewed)
        )

        date += pd.Timedelta(1, unit = 'W')
        
        record = {
          'date': date,
          'wall time': env.now, 'wave': i, 'wave time': env.now - i,
          'in_progress': in_progress,
          'accepted': accepted,
          'rejected': rejected,
          'os_backlog':  recruitment.OS_interview.queue(),
          'phone_backlog' :  recruitment.Phone_interview.queue(),
          'phone' : phone_interviewed,
          'onsite' : os_interviewed,
        }
        
        records.append(record)


        yield env.timeout(1)



def run_recruitment(env,
                    scheduler_capacity,
                    os_interview_capacity,
                    tps_interview_capacity,
                    sourcing_list, records,
                    individual_oa_model,
                    feature_list,
                    job,
                    location
                   ):

    pred_features = np.zeros(len(feature_list))
    pred_features[feature_list == job] = 1
    pred_features[feature_list == location] = 1
    pred_features = pred_features.tolist()

    scheduling = Scheduling(env, scheduler_capacity, os_interview_capacity, tps_interview_capacity, individual_oa_model)


    for i, tickets in sourcing_list.iterrows():

        env.process(google_recruitment(env, tickets, scheduling, individual_oa_model, records, i, pred_features))

        yield env.timeout(1)



def simple_scheduler(os_interview_capacity,
                     tps_interview_capacity,
        scheduler_capacity = 1e6, # capacity is set to 1e6 to illistrate that the default scheduler has no capacity limitations.
        sim_duration = 52,
        job = 'SWE',
        location = 'EMEA',
        individual_oa_model = None, feature_list = None,
        sourcing_list = None
        ):

    records = []

    env = simpy.Environment()
    env.process(
        run_recruitment(
            env,
            scheduler_capacity,
            os_interview_capacity,
            tps_interview_capacity,
            sourcing_list,
            records,
            individual_oa_model=individual_oa_model,
            feature_list = feature_list,
            job = job, 
            location = location
            )
        )

    env.run(until = sim_duration)
    return records

