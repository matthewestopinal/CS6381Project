#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the implementation of the Job object. This object
stores the information of a running job such as arrival time, resource 
requirements, completion time, and job duration
'''

import pandas as pd
import numpy as np
from random import randint
import random

#Function to load a list of jobs stored in a CSV file
#Params:
#   filename: (str) path of CSV file to read
#   resource_names: (str) or (list) of strings representing the columns
#                   in the CSV for each resource
#Returns (list) of (Job) objects
def load_jobs_from_csv(filename, resource_names=[]):
    jobs = []
    job_DF = pd.read_csv(filename)

    job_dicts = job_DF.to_dict(orient='records')

    for job in job_dicts:
        requirements = []
        for resource in resource_names:
            requirements.append(job[resource])
        
        my_job = Job(requirements=requirements,
                    arrival_time=job['Arrival'],
                    duration=job['Duration'])
        
        jobs.append(my_job)

    return jobs

#Keeps track of job data
class Job:
    def __init__(self, requirements=[], arrival_time=0, duration=1):
        self.requirements = requirements
        self.arrival_time = arrival_time
        self.duration = duration

        self.start_time = 0
        self.finish_time = 0

    #Return (list) resource requirements 0-1
    def get_requirements(self):
        return self.requirements

    #Returns (int) time that job started running
    def get_start_time(self):
        return self.start_time
    
    #Sets the time the job started running
    #(int) start_time: time job started
    def set_start_time(self, start_time):
        self.start_time = start_time

    #Returns (int) job duration
    def get_duration(self):
        return self.duration

    #Returns (int) of job arrival time
    def get_arrival_time(self):
        return self.arrival_time

    #VOID
    #Sets job completion time as finish time (int)
    def set_finish_time(self, finish_time):
        self.finish_time = finish_time

    #Returns (int) time of job completion
    def get_finish_time(self):
        return self.finish_time

#Function to create a randomized job within parameters
#Params:
#   arrival_time: (int) time step when job was generated
#   duration_split: (float) between 0-1 to determine probability
#                   of fast or slow job
#   short_duration: (tuple) of ints, range of time for short jobs
#   long_duration: (tuple) of ints, range of time for long jobs
#   main_resource_range: (tuple) of float cost of main Job resource
#   secondary_resource_range: (tuple) of cost of secondary resource(s)
#   num_resources: (int) number of resources
#Returns: Job object
def build_random_job(arrival_time=0,
                    duration_split=0.8,
                    short_duration=(1,60),
                    long_duration=(200,300),
                    main_resource_range=(0.025, 0.05),
                    secondary_resource_range=(0.005, 0.01),
                    num_resources=2):
    #Build our job

    #Choose our resource utilizations
    resource_utilizations = []

    main_resource = randint(0, num_resources-1)
    for i in range(num_resources):

        if i == main_resource:
            #Choose our main resource utilization
            utilization = random.uniform(main_resource_range[0], main_resource_range[1])
            resource_utilizations.append(utilization)
        else:
            #Choose a secondary utilization
            utilization = random.uniform(secondary_resource_range[0], secondary_resource_range[1])
            resource_utilizations.append(utilization)
    
    #Choose our duration
    duration = 0
    duration_roll = random.random()
    if duration_roll > duration_split:
        #Long Job
        duration = random.randint(long_duration[0], long_duration[1])
    else:
        #Short Job
        duration = random.randint(short_duration[0], short_duration[1])

    my_job = Job(requirements = resource_utilizations, arrival_time=arrival_time, duration=duration)
    return my_job

#Function to create jobs for the simulator
#Params:
#   duration_split: (float) between 0-1 to determine probability
#                   of fast or slow job
#   short_duration: (tuple) of ints, range of time for short jobs
#   long_duration: (tuple) of ints, range of time for long jobs
#   main_resource_range: (tuple) of float cost of main Job resource
#   secondary_resource_range: (tuple) of cost of secondary resource(s)
#   desired_utilization: (float) between 0-1, expected total utilization of system
#   num_clusters: (int) number of clusters
#   num_resources: (int) number of resources
#   timesteps: (int) duration of simulation
#Returns: [Job, Job, ...] or [Job, [Job, Job], ]
#   Each index corresponds to a timestep, None if no job
def generate_bernoulli_jobs(duration_split=0.8,
                            short_duration=(1,60),
                            long_duration=(200,300),
                            main_resource_range=(0.025, 0.05),
                            secondary_resource_range=(0.005, 0.01),
                            desired_utilization=0.7,
                            num_clusters=3,
                            num_resources=2,
                            timesteps=800):

    #Calculate expected job duration and utilization
    ave_short = (short_duration[1] + short_duration[0]) / 2
    ave_long = (long_duration[1] + long_duration[0]) / 2
    ave_main_resource = (main_resource_range[1] + main_resource_range[0])

    #We will consider all non-main resources as using the same
    ave_secondary_resource = (secondary_resource_range[1] - secondary_resource_range[0]) / 2

    expected_duration = (duration_split * ave_short + (1 - duration_split) * ave_long)
    
    #Weighted average of utilizations
    expected_utilization = (ave_main_resource * (1 / num_resources) + ave_secondary_resource * (1 - 1 / num_resources)) 

    #Calculate how many jobs we want at once
    #Note that this slightly overestimates 
    desired_concurrent_jobs = num_clusters * desired_utilization / expected_utilization

    jobs_per_second = desired_concurrent_jobs / expected_duration

    if(jobs_per_second >= 1):
        print("Want's more than one job per second. How to handle?")
        return []

    #Now that we have our parameters we can start generating jobs
    jobs = []
    for i in range(timesteps):
        #Check if we are submitting a job
        job_roll = random.random()
        if job_roll > jobs_per_second:
            jobs.append(None)
            continue
            
        my_job = build_random_job(arrival_time=i,
                                  duration_split=duration_split,
                                  short_duration=short_duration,
                                  long_duration=long_duration,
                                  main_resource_range=main_resource_range,
                                  secondary_resource_range=secondary_resource_range,
                                  num_resources=num_resources)
        
        jobs.append(my_job)

    return jobs
