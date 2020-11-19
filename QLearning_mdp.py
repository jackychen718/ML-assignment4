from mdp_util import generate_small_mdp,print_policy,print_action,print_action_large
from my_QLearning import MY_QLearning,MY_QLearning_Large
import mdptoolbox.example
import matplotlib.pyplot as plt
import numpy as np
import mdptoolbox

def small_compare_vi_pi(P,R):
    x_ticks=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    mdp_vi=mdptoolbox.mdp.PolicyIteration(P,R,0.99)
    mdp_vi.run()
    v_pi=list(mdp_vi.V)[:-1]
  
    qlearning_mdp=mdptoolbox.mdp.QLearning(P,R,0.99,n_iter=100000)
    qlearning_mdp.run()
    plt.figure()
    plt.plot(qlearning_mdp.mean_discrepancy)
    plt.title('mean_discrepancy vs every 100 iterations')
    plt.xlabel('100 iterations')
    plt.ylabel('mean_discrepancy')
    plt.show()
    
    
    print('policy by default QLearning')
    print_policy(qlearning_mdp.policy)
    v_default_q=list(qlearning_mdp.V)[:-1]
    fig1,ax1=plt.subplots()
    ax1.bar(x_ticks-0.3,v_pi,0.3,label='policy iteration')
    ax1.bar(x_ticks,v_default_q,0.3,label='default qlearning')
    ax1.set_xlabel('state')
    ax1.set_ylabel('utility')
    ax1.set_title('utility comparison between PI and default qlearning')
    ax1.set_xticks(x_ticks)
    ax1.legend()
    plt.show()
    
    
    my_qlearning_mdp=MY_QLearning(P,R,0.99,n_iter=100000)
    my_qlearning_mdp.run()
    print('policy by modified Qlearning')
    print_policy(my_qlearning_mdp.policy)
    plt.plot(my_qlearning_mdp.mean_discrepancy)
    plt.title('mean_discrepancy vs every 100 iterations')
    plt.xlabel('100 iterations')
    plt.ylabel('mean_discrepancy')
    plt.show()
    
    v_modified_q=list(my_qlearning_mdp.V)[:-1]
    fig2,ax2=plt.subplots()
    ax2.bar(x_ticks-0.3,v_pi,0.3,label='policy iteration')
    ax2.bar(x_ticks,v_modified_q,0.3,label='modified qlearning')
    ax2.set_xlabel('state')
    ax2.set_ylabel('utility')
    ax2.set_title('utility comparison between PI and modified qlearning')
    ax2.set_xticks(x_ticks)
    ax2.legend()
    plt.show()

def large_compare_vi_pi(P,R):
    mdp_pi=mdptoolbox.mdp.PolicyIteration(P,R,0.99)
    mdp_pi.run()
    print('policy for forest MDP using PI:')
    print_action(mdp_pi.policy)
    
    qlearning_large=mdptoolbox.mdp.QLearning(P,R,0.99,n_iter=5000000)
    qlearning_large.run()
    plt.plot(qlearning_large.mean_discrepancy)
    plt.xlabel('100 iterations')
    plt.ylabel('mean_discrepancy')
    plt.title('mean_discrepancy every 100 iterations')
    plt.show()
    print('policy learned by default QLearning:')
    print_action_large(qlearning_large.policy)
    plt.plot(mdp_pi.V,label='PI')
    plt.plot(qlearning_large.V,label='default QLearning')
    plt.xlabel('state')
    plt.ylabel('utility')
    plt.title('utility between PI and default qlearning')
    plt.legend()
    plt.show()
    
    qlearning_large_modified=MY_QLearning_Large(P,R,0.99,random_count=10,n_iter=5000000)
    qlearning_large_modified.run()
    plt.plot(qlearning_large_modified.mean_discrepancy)
    plt.xlabel('100 iterations')
    plt.ylabel('mean_discrepancy')
    plt.title('mean_discrepancy every 100 iterations by modified QLearning')
    plt.show()
    print('policy learned by modified QLearning')
    print_action_large(qlearning_large_modified.policy,title='optimal policy by modified qlearning')
    plt.plot(mdp_pi.V,label='PI')
    plt.plot(qlearning_large_modified.V,label='modified qlearning')
    plt.xlabel('state')
    plt.ylabel('utility')
    plt.title('utility between PI and modified qlearning')
    plt.legend()
    plt.show()
    
    
    qlearning_large_finalized=MY_QLearning_Large(P,R,0.99,random_count=3,n_iter=5000000)
    qlearning_large_finalized.run()
    plt.plot(qlearning_large_finalized.mean_discrepancy)
    plt.xlabel('100 iterations')
    plt.ylabel('mean_discrepancy')
    plt.title('mean_discrepancy every 100 iterations by finalized QLearning')
    plt.show()
    print('policy learned by finalized QLearning')
    print_action_large(qlearning_large_finalized.policy,title="optimal policy by finalized qlearning")
    plt.plot(mdp_pi.V,label='PI')
    plt.plot(qlearning_large_finalized.V,label='finalized qlearning')
    plt.xlabel('state')
    plt.ylabel('utility')
    plt.title('utility between PI and finalized qlearning')
    plt.legend()
    plt.show()
    
    

if __name__=='__main__':
    #P_small,R_small=generate_small_mdp(reward=0.001)
    #small_compare_vi_pi(P_small,R_small)
    P_big,R_big=mdptoolbox.example.forest(500,10,5,0.1)
    large_compare_vi_pi(P_big,R_big)
    
    
    
    