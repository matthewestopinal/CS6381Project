#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the implementation of the Job object. This object
stores the information of a running job such as arrival time, resource 
requirements, completion time, and job duration
'''

import pandas as pd

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