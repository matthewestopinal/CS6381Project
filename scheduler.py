#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the Abstract Base Class of the Scheduler Objects, as well
as various implementations following different Scheduling Rules
'''

from abc import ABC, abstractmethod
import random
import numpy as np
from cluster import Cluster

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
        #Create a random order of indices
        indices = np.arange(len(clusters))
        np.random.shuffle(indices)

        #Try to fit job in first available randomly drawn
        for index in indices:
            if clusters[index].check_job_possible(job):
                return index
        
        #No fit on clusters
        return -1

#First Available Scheduler
class FirstAvailableScheduler(Scheduler):

    #Tries to find the first available cluster
    def schedule_job(self, clusters, job):
        for index, cluster in enumerate(clusters):
            if cluster.check_job_possible(job):
                return index

        #Job did not fit on any machine
        return -1

#Schedules job in order of Least Load    
class LeastLoadScheduler(Scheduler):

    def schedule_job(self, clusters, job):
        utilizations = []
        indices = []

        #Get a list of total utilization in each cluster
        for index, cluster in enumerate(clusters):
            indices.append(index)
            cluster_utilizations = cluster.get_utilization()
            utilizations.append(sum(cluster_utilizations))

        #Sort indices in order of least utilization
        sorted_indices = [index for _,index in sorted(zip(utilizations, indices))]

        #Try to fit job in order of Least Load
        for index in sorted_indices:
            if clusters[index].check_job_possible(job):
                return index
        
        #No fit on clusters
        return -1

#Round Robin Scheduler
class RoundRobinScheduler(Scheduler):
    def __init__(self):
        self.cur_cluster = 0

    def schedule_job(self, clusters, job):
        cluster = self.cur_cluster

        indices = np.arange(cluster, cluster + len(clusters), dtype=int)
        indices = np.mod(indices, len(clusters))

        #Try to fit job in first available drawn
        for index in indices:
            if clusters[index].check_job_possible(job):
                self.cur_cluster = np.mod(cluster + 1, len(clusters))
                return index
        
        #No fit on clusters
        self.cur_cluster = np.mod(cluster + 1, len(clusters))
        return -1