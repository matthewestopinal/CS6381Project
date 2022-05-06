#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the code to run the simulation under various different
configurations.
'''
from re import A
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
    parser.add_argument('-u', '--utilization', default=0.7, type=float, help='Desired total utilization (between 0 and 1)')
    parser.add_argument('-n', '--num_clusters', default=3, type=int, help='Number of clusters to simulate')
    parser.add_argument('-r', '--num_resources', default=2, type=int, help='Number of resources on each cluster')

    parser.add_argument('-a', '--alpha', default = 1, type=float, help='alpha for reward function')
    parser.add_argument('-b', '--beta', default = 1, type=float, help='beta for reward function')
    parser.add_argument('-c', '--gamma', default = 1, type=float, help='gamma for reward function')

    return parser.parse_args()

#Function to graph the resource utilization of each cluster
#Params:
#   clusters: (list) of (Cluster) object whose history we would like to graph
#Returns
#   (pyplot Figure Object) to save or display
def graph_utilization(clusters):
    num_clusters = len(clusters)
    fig, axs = plt.subplots(2, num_clusters, constrained_layout=True)

    for index, cluster in enumerate(clusters):
        resources = np.array(cluster.get_utilization_history())
        num_resources = resources.shape[1]

        #Plot resource utiliztion
        for resource in range(num_resources):
            resource_values = resources[:, resource]
            axs[0,index].plot(resource_values)

        axs[0,index].set_ylabel('Utilizations')
        axs[0,index].set_xlabel('Time')
        axs[0,index].set_ylim([0,1])

        #Get our imbalance and averages
        diff = cluster.get_utilization_history_difference()
        utilization_ave = cluster.get_utilization_history_average()

        axs[1,index].plot(utilization_ave, 'g', label='mean utilization')

        ax2 = axs[1,index].twinx()
        ax2.plot(diff, 'r', label='imbalance')
        ax2.set_ylabel('Degree of Imbalance')
        ax2.set_ylim([0,0.6])
        ax2.legend(loc='upper right', frameon=False)

        axs[1,index].set_xlabel('Time')
        axs[1,index].set_ylim([0,1])
        axs[1,index].set_ylabel('Mean Utilization')
        axs[1,index].legend(loc='upper left', frameon=False)

    fig.set_size_inches((16,9))

    return fig

def main():
    args = parseCmdLineArgs()

    my_scheduler = sc.RandomScheduler()
    
    if args.scheduler == 'first-available':
        my_scheduler = sc.FirstAvailableScheduler()
    elif args.scheduler == 'round-robin':
        my_scheduler = sc.RoundRobinScheduler()
    elif args.scheduler == 'rl':
        my_scheduler = sc.RLScheduler()
    elif args.scheduler == 'least-load':
        my_scheduler = sc.LeastLoadScheduler()
    elif args.scheduler == 'instant-gratification':
        my_scheduler = sc.InstantGratificationScheduler(args.alpha, args.beta, args.gamma)

    num_clusters = args.num_clusters
    num_resources = args.num_resources
    target_utilization = args.utilization

    #Create our clusters
    clusters = []
    for i in range(num_clusters):
        clusters.append(Cluster(resources=num_resources))

    job_queue = generate_bernoulli_jobs(num_clusters=num_clusters, num_resources=num_resources, desired_utilization=target_utilization)

    #Main Loop
    cur_job = 0
    for t in range(args.timesteps):
        #Try to schedule all arrived jobs
        #TODO Fix advancing past current jobs
        while cur_job <= t:
            if len(job_queue[cur_job]) > 0:

                #Create a copy of the list to iterate through
                jobs_to_schedule = job_queue[cur_job]
                jobs_to_schedule = jobs_to_schedule[:]

                jobs_remaining = False
                for job in jobs_to_schedule:
                    #Get the assigned cluster

                    assigned_cluster = my_scheduler.schedule_job(clusters, job)

                    #Put job on cluster
                    if assigned_cluster == -1:
                        jobs_remaining = True
                        break
                    else:
                        clusters[assigned_cluster].schedule_job(job)

                        job_queue[cur_job].remove(job)

                #A little messy but I suppose it is what it is
                if jobs_remaining:
                    break
                else:
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

