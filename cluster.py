#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the implementation of the Cluster Object. This object
stores the state of a single cluster during the simulation and also stores
a history of utillizations and completed jobs
'''


#Import the Job object
from jobs.py import Job

from abc import ABC, abstractmethod
import random

#Maintains resources of a 
class Cluster:
    def __init__(self, resources=2):
        self.timestep = 0
        self.utilization_history = []
        self.cur_utilization = []
        for i in range(resources):
            self.cur_utilization.append(0)

        self.cur_jobs = []
        self.completed_jobs = []
        self.job_queue = []

    #Accepts a (Job) object
    #Returns True if the job can be scheduled right now
    def check_job_possible(self, job):
        count = 0
        for requirement in job.get_requirements():
            if self.resource[count] + requirement > 1:
                return False
        return True

    #Method to add jobs to a queue of a cluster
    #This is useful for when we do round robin / random approaches
    #Unlike RL Model, this assign a cluster and won't delay the job,
    #Just set them to idol
    #Params:
    #   job: (Job) object to queue
    #VOID
    def queue_job(self, job):
        self.job_queue.append(job)

    #Method to attempt to start all possible jobs in the queue
    #VOID
    def start_from_queue(self):
        while len(self.job_queue) > 0:
            cur_job = self.job_queue[0]
            if self.check_job_possible(cur_job):
                self.schedule_job(cur_job)
                self.job_queue.pop(0)
            else:
                return

    #VOID accepts job, and sets utilization
    def schedule_job(self, job):
        count = 0
        for requirement in job.get_requirements():
            self.resource[count] += requirement
        
        #Keep Track of when the job started
        job.set_start_time(self.timestep)
        
        #Keep track of currently running jobs
        self.cur_jobs.append(job)

    #VOID Marks a Job object as complete
    #Moves job from current job list to completed job list
    #Removes it's utilization from resources
    def complete_job(self, job):
        count = 0
        for requirement in job.get_requirements():
            self.resource[count] -= requirement

        job.set_finish_time(self.timestep)
        self.complete_job.append(job)
        self.cur_jobs.remove(job)

    #VOID Advances the timestep of the simulation
    def step(self):
        self.timestep += 1

        #Advance our jobs
        for job in self.cur_jobs[:]:
            if self.timestep - job.get_start_time() >= job.get_duration():
                self.complete_job(job)

        #See if we can move jobs in from the queue
        self.start_from_queue()

        #Maintain history for visualization
        self.utilization_history.append(self.cur_utilization)

    #Returns (list) of resource utilizations
    #Float in range 0-1
    def get_utilization(self):
        return self.cur_utilization

    #Returns (list) of lists
    #Each list corresponds to the resource utilization at a given timestep
    def get_utilization_history(self):
        return self.utilization_history

    #Returns (int) current time step of cluster
    def get_timestep(self):
        return self.timestep

    #Returns (list) of completed Job objects
    #
    def get_completed_jobs(self):
        return self.completed_jobs

#Abstract Base Class for our various scheduler types
class Scheduler(ABC):

    #Method to Schedule Job
    #Params:
    #   clusters: (list) of (Cluster) objects
    #   job: (Job) object to be scheduled
    #Returns:
    #   (int) index of cluster in list to be assigned to
    #       -1 if job is to be delayed and re-scheduled
    @abstractmethod
    def schedule_job(self, clusters, job):
        pass

#Random Scheduler
class RandomScheduler(Scheduler):

    #Outputs a random index in the cluster list
    def schedule_job(self, clusters, job):
        return random.randint(0,len(clusters))

#First Available Scheduler
class FirstAvailableScheduler(Scheduler):

    #Tries to find the first available cluster
    def schedule_job(self, clusters, job):
        for index, cluster in enumerate(clusters):
            if cluster.check_job_possible(job):
                return index

        #Job did not fit on any machine
        return -1

#Round Robin Scheduler
class RoundRobinScheduler(Scheduler):
    def __init__(self):
        self.cur_cluster = 0
        super.__init__()

    def schedule_job(self, clusters, job):
        cluster = self.cur_cluster
        self.cur_cluster += 1
        if self.cur_cluster >= len(clusters):
            self.cur_cluster = 0
        return cluster