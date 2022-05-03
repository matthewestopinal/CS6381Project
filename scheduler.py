#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the Abstract Base Class of the Scheduler Objects, as well
as various implementations following different Scheduling Rules
'''

from abc import ABC, abstractmethod
import random
from cluster.py import Cluster

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

#DeepQLearning Scheduler
#TODO: Implement the deep learning model
#Will want to store the model as a class variable
class RLScheduler(Scheduler):

    def schedule_job(self, clusters, job):
        pass