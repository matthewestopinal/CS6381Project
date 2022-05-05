#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the code to run the simulation under various different
configurations.
'''
from jobs import Job
from jobs import generate_bernoulli_jobs
from cluster import Cluster
import scheduler as sc
import matplotlib.pyplot as plt
import numpy as np

import argparse

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(description='Run simulation')
    
    parser.add_argument('-t', '--timesteps', default = 800, help='Length of time that jobs can arrive', type=int)
    parser.add_argument('-s', '--scheduler', default='random', help = 'Which scheduler to use')
    parser.add_argument('-o', '--output', default='output.png', help = 'Name of output file for graphed figures')
    return parser.parse_args()

#Function to graph the resource utilization of each cluster
#Params:
#
def graph_utilization(clusters):
    num_clusters = len(clusters)
    fig, axs = plt.subplots(1, num_clusters, constrained_layout=True)

    for index, cluster in enumerate(clusters):
        resources = np.array(cluster.get_utilization_history())
        num_resources = resources.shape[1]

        for resource in range(num_resources):
            resource_values = resources[:, resource]
            #print(resource_values)
            axs[index].plot(resource_values)
        #Want to reshape our resource

        axs[index].set_ylabel('Utilizations')
        axs[index].set_xlabel('Time')
        axs[index].set_ylim([0,1])
    return fig

def main():
    args = parseCmdLineArgs()

    my_scheduler = sc.RandomScheduler()
    
    if args.scheduler == 'first-available':
        my_scheduler = sc.FirstAvailableScheduler()
    elif args.scheduler == 'round-robin':
        my_scheduler = sc.FirstAvailableScheduler()
    elif args.scheduler == 'rl':
        my_scheduler = sc.RLScheduler()
    elif args.scheduler == 'least-load':
        my_scheduler = sc.LeastLoadScheduler()

    num_clusters = 3
    num_resources = 2

    #Create our clusters
    clusters = []
    for i in range(num_clusters):
        clusters.append(Cluster(resources=num_resources))

    job_queue = generate_bernoulli_jobs()

    #Main Loop
    cur_job = 0
    for t in range(args.timesteps):
        #Try to schedule all arrived jobs
        while cur_job <= t:
            if not (job_queue[cur_job] is None):
                #Get the assigned cluster
                assigned_cluster = my_scheduler.schedule_job(clusters, job_queue[cur_job])
                #print(f"Assigned cluster: {assigned_cluster}")
                #Put job on cluster
                if assigned_cluster == -1:
                    break
                else:
                    clusters[assigned_cluster].schedule_job(job_queue[cur_job])
                    cur_job += 1
            else:
                cur_job += 1
        
        #Advance simulation
        for index, cluster in enumerate(clusters):
            #print(f"Advancing step in cluster {index}")
            cluster.step()

    fig = graph_utilization(clusters)
    fig.suptitle(f"Utilizations  with {args.scheduler} Scheduling")
    plt.savefig(args.output)

if __name__ == "__main__":
    main()

