#Author - Matthew Estopinal
#Distributed Systems Final Project

#---------------------------------------------
'''
Contains the code to run the simulation under various different
configurations.
'''
from re import A
from jobs import Job
from jobs import generate_bernoulli_jobs, generate_bernoulli_jobs_rl
from cluster import Cluster
import scheduler as sc
import matplotlib.pyplot as plt
import numpy as np
import DoubleDQN as DDQN
import DuelingDQN as DueDQN
import DQN as DQN
import argparse

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(description='Run simulation')
    
    parser.add_argument('-t', '--timesteps', default = 800, help='Length of time that jobs can arrive', type=int)
    parser.add_argument('-s', '--scheduler', default='random', help = 'Which scheduler to use')
    parser.add_argument('-o', '--output', default='output.png', help = 'Name of output file for graphed figures')
    parser.add_argument('-u', '--utilization', default=0.7, type=float, help='Desired total utilization (between 0 and 1)')
    parser.add_argument('-n', '--num_clusters', default=3, type=int, help='Number of clusters to simulate')
    parser.add_argument('-r', '--num_resources', default=2, type=int, help='Number of resources on each cluster')
    parser.add_argument('-p', '--alpha', default = 1.3, type=float, help='alpha for reward function')
    parser.add_argument('-b', '--beta', default = 1, type=float, help='beta for reward function')
    parser.add_argument('-c', '--gamma', default = 1, type=float, help='gamma for reward function')

    parser.add_argument('-a', '--algo', default='double', type=str, help='Reinforcement learning algorithm')

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
        # MAX_EPISODES = 1000
        MAX_EPISODES = 300
        #MAX_STEPS = 500
        batch_size = 32
        #episode_rewards = RL.mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
    elif args.scheduler == 'least-load':
        my_scheduler = sc.LeastLoadScheduler()

    elif args.scheduler == 'instant-gratification':
        my_scheduler = sc.InstantGratificationScheduler(0, args.beta, args.gamma)
        
    num_clusters = args.num_clusters
    num_resources = args.num_resources
    target_utilization = args.utilization


    #env = gym.make(env_id)

    #Create our clusters
    clusters = []
    for i in range(num_clusters):
        clusters.append(Cluster(resources=num_resources))

    job_queue = generate_bernoulli_jobs(num_clusters=num_clusters, num_resources=num_resources, desired_utilization=target_utilization)
    #Main Loop
    cur_job = 0
    average_rewards = []

    if args.scheduler == 'rl':
        episode_rewards = []
        #agent = RL.DuelingAgent(num_clusters, num_resources)
        if args.algo == 'double':
            agent = DDQN.DDQNAgent(num_clusters, num_resources)
        elif args.algo == 'dule':
            agent = DueDQN.DuelingAgent(num_clusters, num_resources)
        elif args.algo == 'dqn':
            agent = DQN.DQNAgent(num_clusters, num_resources)
        job_queue_rl = generate_bernoulli_jobs_rl(num_clusters=num_clusters, num_resources=num_resources,
                                   desired_utilization=target_utilization)
        for episode in range(MAX_EPISODES):
            # generate job sequence
            job_queue_rl_copy = generate_bernoulli_jobs_rl(num_clusters=num_clusters, num_resources=num_resources,
                                   desired_utilization=target_utilization)
            # build clusters
            clusters = []
            for i in range(num_clusters):
                clusters.append(Cluster(resources=num_resources))

            episode_reward = 0
            for step in range(args.timesteps):
                state = []
                total_utilization = 0 # Util(t) in paper: sum of cluster utilizations
                diff_u = 0 # DiffCluster(t) in paper: utilization inbalance between resources, across all clusters
                balance_u = 0 # DiffRes(t) in paper: utilization inbalance between clusters
                clusters_utilization = []

                for index, cluster in enumerate(clusters):
                    # should we skip when i == j? 
                    # With only 2 resources, we might want to just take the difference between the two
                    for i in range(len(cluster.cur_utilization)):
                        for j in range(i, len(cluster.cur_utilization)):
                            diff_u += abs(cluster.cur_utilization[i]-cluster.cur_utilization[j])

                    this_cluster_utilization = np.mean(cluster.cur_utilization) # utilization for this cluster
                    total_utilization += this_cluster_utilization # add this cluster's utilization to Util(t)
                    for u in cluster.cur_utilization: # append resource utilizations for this cluster to state
                        state.append(u) 
                    
                    clusters_utilization.append(this_cluster_utilization)
                
                # calculate DiffRes(t)
                for i in range(len(clusters_utilization)):
                    for j in range(i, len(clusters_utilization)):
                        balance_u += abs(clusters_utilization[i]-clusters_utilization[j])

                if len(job_queue_rl_copy) > 0:
                    jobs_to_schedule = job_queue_rl_copy[0]
                for u in jobs_to_schedule.requirements:
                    state.append(u)
                action = agent.get_action(state)
                if action != 0:
                    if clusters[action-1].check_job_possible(jobs_to_schedule):
                        clusters[action - 1].schedule_job(jobs_to_schedule)
                    else:
                        while True:
                            action = agent.get_action(state)
                            if action == 0:
                                break
                            elif clusters[action - 1].check_job_possible(jobs_to_schedule):
                                clusters[action - 1].schedule_job(jobs_to_schedule)
                                break


                for index, cluster in enumerate(clusters):
                    cluster.step()
                next_state = []
                for index, cluster in enumerate(clusters):
                    for u in cluster.cur_utilization:
                        next_state.append(u)
                if action == 0:
                    jobs_to_schedule_next = job_queue_rl_copy[0]
                else:
                    job_queue_rl_copy.pop(0)
                    jobs_to_schedule_next = job_queue_rl_copy[0]

                for u in jobs_to_schedule_next.requirements:
                    next_state.append(u)
                reward = args.alpha * total_utilization - args.beta * diff_u - args.gamma * balance_u
                agent.replay_buffer.push(state, action, reward, next_state, 0)
                episode_reward += reward

                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)

                if step == (args.timesteps - 1) or len(job_queue_rl_copy) == 1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                    print("Episode loss : " + str(agent.get_loss()))
                    average_reward = np.mean(episode_rewards[:-50])
                    average_rewards.append(average_reward)
                    break
                step += 1
                
            if args.algo == 'double':
                agent.update_target()
        # return episode_rewards
        # print(episode_rewards)
        


    else:
        for t in range(args.timesteps):
            #Try to schedule all arrived jobs
            #TODO Fix advancing past current jobs
            while cur_job <= t:
                if len(job_queue[cur_job]) > 0:

                    #Create a copy of the list to iterate through
                    jobs_to_schedule = job_queue[cur_job]
                    #print(jobs_to_schedule)
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
            #temp = []
            for index, cluster in enumerate(clusters):
                #print(f"Advancing step in cluster {index}")
                #temp.append(cluster.cur_utilization)
                cluster.step()
    # for cluster in clusters:
    #     print(cluster.__dict__)
    fig = graph_utilization(clusters)
    fig.suptitle(f"Utilizations  with {args.scheduler} Scheduling")
    plt.savefig(args.output)

    if average_rewards:
        plt.figure()
        plt.plot(average_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Avg Episodic Reward")
        plt.savefig(f"rewards_{args.output}")
if __name__ == "__main__":
    main()

