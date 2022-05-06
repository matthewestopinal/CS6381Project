#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the Abstract Base Class of the Scheduler Objects, as well
as various implementations following different Scheduling Rules
'''

from abc import ABC, abstractmethod
import enum
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

#Schdeuler that maximizes immediate reward (No lookahead)
class InstantGratificationScheduler(Scheduler):
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    #Calculate reward function based on theoritically scheduling each job on each machine
    def schedule_job(self,clusters,job):
        
        indices = np.arange(0, len(clusters), dtype=int)
        
        rewards = self.calculate_rewards(clusters, job)

        #Sort indices in order of highest rewards
        sorted_indices = [index for _,index in sorted(zip(rewards, indices), reverse=True)]

        #Try to fit job in order of Highest Instant Reward
        for index in sorted_indices:
            if clusters[index].check_job_possible(job):
                return index
        
        #No fit on clusters
        return -1

    def calculate_rewards(self, clusters, job):
        rewards = []

        total_util = self.get_total_utilization(clusters, job)
        #Simulate scheduling to each cluster
        for index in range(len(clusters)):
            diff_res = self.get_diff_res(clusters, job, index)
            diff_cluster = self.get_diff_cluster(clusters, job, index)

            rewards.append(self.alpha * total_util - self.beta * diff_cluster - self.gamma * diff_res)

        return rewards

    #Function to calculate different resources utilization between clusters given a job and a cluster to schedule to
    #Params:
    #   clusters: (list) of (Cluster) objects in sim
    #   job: (Job) object to schedule
    #   target_cluster: (int) index of cluster in clusters to predict scheduling
    def get_diff_res(self, clusters, job, target_cluster):
        diff_res = 0
        job_reqs = job.get_requirements()
        num_resources = len(job_reqs)

        for i, cluster in enumerate(clusters):
            i_util = cluster.get_utilization()
            i_util = sum(i_util)

            #If i is the cluster we are predicting to schedule to
            if i == target_cluster:
                i_util += sum(job_reqs)

            for j, cluster in enumerate(clusters):
                j_util = cluster.get_utilization()
                j_util = sum(j_util)

                #If i is the cluster we are predicting to schedule to
                if j == target_cluster:
                    j_util += sum(job_reqs)

                diff_res += np.absolute(i_util - j_util)

        return diff_res


    #Function to calculate different balance of resources within clusters given a job and a cluster to schedule to
    #Params:
    #   clusters: (list) of (Cluster) objects in sim
    #   job: (Job) object to schedule
    #   target_cluster: (int) index of cluster in clusters to predict scheduling
    def get_diff_cluster(self, clusters, job, target_cluster):
        diff_cluster = 0
        job_reqs = job.get_requirements()
        num_resources = len(job_reqs)

        #loop through all clusters
        for m, cluster in enumerate(clusters):
            utilization = cluster.get_utilization()
            for i in range(num_resources):
                for j in range(num_resources):

                    #Predicting job on this cluster
                    if m == target_cluster:
                        diff_cluster += np.absolute(utilization[i] + job_reqs[i] - utilization[j] - job_reqs[j])

                    else:
                        diff_cluster += np.absolute(utilization[i] - utilization[j])

        return diff_cluster


    #Function to calculate predicted utilization upon scheduling this job
    def get_total_utilization(self, clusters, job):
        job_reqs = job.get_requirements()
        
        total_utilization = 0

        for value in job_reqs:
            total_utilization += value

        for cluster in clusters:
            cur_utilization = cluster.get_utilization()
            for value in cur_utilization:
                total_utilization += value
        
        return total_utilization



