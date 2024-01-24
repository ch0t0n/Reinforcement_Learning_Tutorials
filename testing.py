# Implementation of Figure 2.4 from the book
import numpy as np 
import matplotlib.pyplot as plt

# epsilon-greedy method for n-bandits
mu, sigma = 0, 1 # mean and standard deviation. Here variance is same as standard deviation
k = 10
timesteps = 100
n_bandits = 2000
epsilon = [0, 0.1]

# true values
Q_true = np.random.normal(mu, sigma, (n_bandits, k)) # the true rewards are taken from normal distribution
opt_action = np.argmax(Q_true, 1) # the optimal action is the action that have maximum expected rewards

# for plotting graphs
fig1 = plt.figure(1).gca() # Figure 2.4(a) from the book
fig2 = plt.figure(2).gca() # Figure 2.4(b) from the book
x_vals = range(timesteps+1)

for ep in epsilon: # run the full experiment for each epsilon
    print("Current epsilon: ", ep)
    # step 0
    Q_values = np.zeros((n_bandits, k)) # initial values
    rewards = [0] # initial average rewards over all bandits are also zero
    opt_action_ratio = [0] # % of optimal action taken

    # step 1
    st = 0
    if (ep == 0.1):
        N_values = np.ones((n_bandits, k)) # number of actions taken. All actions are taken once at step 1
        Q_init = np.random.normal(Q_true, sigma) # the rewards at step 1 are taken from the true rewards for each action
        rewards.append(np.mean(Q_init)) # average reward after step 1
        opt_action_ratio.append(0) # Optimal action is not taken
        st = 2
    else:
        N_values = np.zeros((n_bandits, k)) # number of actions taken. All actions are taken once at step 1
        st = 1
    # step 2-timesteps
    for t in range(st, timesteps+1):
        print("Running step ", t, "using sample-average method for epsilon: ", ep)
        step_rewards = [] # rewards for this step
        opt_action_taken = 0 # number of optimal actions taken in this step
        for i in range(n_bandits):
            if ep == 0:
                c = 2 # degree of exploration
                sq_term = np.sqrt((np.log(t)/N_values[i]))
                act = np.argmax(Q_values[i]+c*sq_term)
            else:             
                if np.random.random() < ep:
                    act = np.random.randint(k) # random action
                else:
                    act = np.argmax(Q_values[i]) # greedy action
            if act == opt_action[i]:
                opt_action_taken += 1 # if the action taken is the optimal, then increase the count
            R = np.random.normal(Q_true[i][act], sigma) # select the true expected reward
            step_rewards.append(R) # append it in step rewards
            N_values[i][act] += 1 # update the number of actions taken
            Q_values[i][act] = Q_values[i][act] + (R-Q_values[i][act])/N_values[i][act] # update the queue values
        R_avg = np.mean(step_rewards) # average reward for this step
        rewards.append(R_avg) # add the average reward for this step
        opt_action_ratio.append(opt_action_taken*100/n_bandits) # % of optimal action taken is added
    # plotting the graphs
    fig1.plot(x_vals, rewards)
    fig2.plot(x_vals, opt_action_ratio)
# labels and legends for the plots
fig1.set_ylabel('Average Reward')
fig1.set_xlabel('Steps')
fig1.set_xticks(x_vals)
fig1.legend((r'UCB c='+str(c), r'$\epsilon$='+str(epsilon[1])), loc='best')

fig2.set_ylabel(r'$\%$ Optimal Action')
fig2.set_xlabel('Steps')
fig2.set_ylim(0,100)
fig2.legend((r'UCB c='+str(c), r'$\epsilon$='+str(epsilon[1])), loc='best')
plt.show()