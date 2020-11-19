from mdp_util import print_policy,generate_small_mdp,print_action
import mdptoolbox.example
import matplotlib.pyplot as plt
import numpy as np


def small_value_iteration(P,R):
    epsilon_list=[0.01,0.001,0.0001,0.00001]
    x_ticks=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    num_iter_list=[]
    fig,ax=plt.subplots(figsize=(12,6))
    for i in range(4):
        my_mdp=mdptoolbox.mdp.ValueIteration(P,R,0.99,epsilon=epsilon_list[i])
        my_mdp.run()
        print('\n')
        print('policy for \nepsilon='+str(epsilon_list[i]))
        print_policy(my_mdp.policy)
        num_iter_list.append(my_mdp.iter)
        ax.bar(x_ticks-0.2*(2-i),list(my_mdp.V)[:-1],0.2,label='espsilon='+str(epsilon_list[i]))
    ax.set_title('Utility for different epsilon')
    ax.set_xticks(x_ticks)
    ax.set_xlabel('state')
    ax.set_ylabel('utility')
    ax.legend()
    
    print(num_iter_list)
    _,ax_iter=plt.subplots()
    ax_iter.bar([1,2,3,4],num_iter_list)
    ax_iter.set_xticks([1,2,3,4])
    ax_iter.set_xticklabels(['0.01','0.001','0.0001','0.00001'])
    ax_iter.set_xlabel('epsilon')
    ax_iter.set_ylabel('num iteration')
    ax_iter.set_title('num of iteration for different epsilon')
    plt.show()
    
 
def small_policy_iteration(P,R):
    x_ticks=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    my_mdp=mdptoolbox.mdp.PolicyIteration(P,R,0.99)
    my_mdp.run()
    num_iter_list=[]
    print('Optimal policy for grid MDP using Policy Iteration:')
    print_policy(my_mdp.policy)
    num_iter_list.append(my_mdp.iter)
    v_pi=list(my_mdp.V)[:-1]
    my_vi_mdp=mdptoolbox.mdp.ValueIteration(P,R,0.99,epsilon=0.00001)
    my_vi_mdp.run()
    v_vi=list(my_vi_mdp.V)[:-1]
    num_iter_list.append(my_vi_mdp.iter)
    fig1,ax=plt.subplots()
    ax.bar(x_ticks-0.3,v_pi,0.3,label='policy iteration')
    ax.bar(x_ticks,v_vi,0.2,label='value iteration')
    ax.set_xticks(x_ticks)
    ax.legend()
    ax.set_xlabel('state')
    ax.set_ylabel('utility value')
    ax.set_title('utility value between PI and VI')
    
    _,ax1=plt.subplots(figsize=(4,4))
    ax1.bar([1,2],num_iter_list,width=0.5)
    ax1.set_xticks([1,2])
    ax1.set_xticklabels(['PI','VI'])
    ax1.set_title('number of iterations between PI and VI')
    ax1.set_ylabel('iterations')
    plt.show()
    
    
def big_value_iteration(P,R):
    epsilon_list=[0.01,0.001,0.0001,0.00001]
    v_list=[]
    iter_list=[]
    for i in range(4):
        my_mdp=mdptoolbox.mdp.ValueIteration(P,R,0.99,epsilon=epsilon_list[i])
        my_mdp.run()
        print('policy for epsilon='+str(epsilon_list[i]))
        print_action(my_mdp.policy,epsilon_list[i])
        v_list.append(my_mdp.V)
        iter_list.append(my_mdp.iter)

        
    
    for i in range(4):
        plt.plot(np.arange(500)+1,v_list[i],label='epsilon='+str(epsilon_list[i]))
    plt.legend()
    plt.xlabel('state')
    plt.ylabel('utility value')
    plt.title('utility for different epsilon')
    plt.show()
    
    _,ax=plt.subplots(figsize=(4,4))
    ax.bar([1,2,3,4],iter_list,width=0.5)
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['0.01','0.001','0.0001','0.00001'])
    ax.set_title('num of iterations for different epsilon')
    ax.set_ylabel('iterations')
    plt.show()
    
def big_policy_iteration(P,R):
    my_mdp=mdptoolbox.mdp.PolicyIteration(P,R,0.99)
    my_mdp.run()
    print('optimal policy for PI on forest MDP:')
    print_action(my_mdp.policy)
    print(my_mdp.time)
    print('\n')
    my_mdp_vi=mdptoolbox.mdp.ValueIteration(P,R,0.99,epsilon=0.00001)
    my_mdp_vi.run()
    print(my_mdp_vi.time)
    plt.plot(np.arange(500)+1,my_mdp.V,label='PI')
    plt.plot(np.arange(500)+1,my_mdp_vi.V,label='VI')
    plt.legend()
    plt.xlabel('states')
    plt.ylabel('utility value')
    plt.title('utility value between PI and VI')
    plt.show()
    
    
    
    _,ax=plt.subplots(figsize=(4,4))
    ax.bar([1,2],[my_mdp.iter,my_mdp_vi.iter],width=0.5)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['PI','VI'])
    ax.set_title('num of iteration between PI and VI')
    plt.show()
    
    _,ax1=plt.subplots(figsize=(4,4))
    ax1.bar([1,2],[my_mdp.time,my_mdp_vi.time],width=0.5)
    ax1.set_xticks([1,2])
    ax1.set_xticklabels(['PI','VI'])
    ax1.set_title('time comparison between PI and VI')
    plt.show()


if __name__=='__main__':
    P_small,R_small=generate_small_mdp(-0.01)
    small_value_iteration(P_small,R_small)
    small_policy_iteration(P_small,R_small)
    P_big,R_big=mdptoolbox.example.forest(500,10,5,0.1)
    big_value_iteration(P_big,R_big)
    big_policy_iteration(P_big,R_big)
    
