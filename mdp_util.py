import numpy as np
import pandas as pd
import mdptoolbox
import matplotlib.pyplot as plt

def generate_small_mdp(reward=-0.01):
    df1=pd.read_csv('./grid/p1.csv',header=None)
    #print(df1)
    df2=pd.read_csv('./grid/p2.csv',header=None)
    #print(df2)
    df3=pd.read_csv('./grid/p3.csv',header=None)
    #print(df3)
    df4=pd.read_csv('./grid/p4.csv',header=None)
    #print(df4)    
    P1=df1.to_numpy()
    P2=df2.to_numpy()
    P3=df3.to_numpy()
    P4=df4.to_numpy()
    P=np.array([P1,P2,P3,P4])
    R=np.array([reward]*16+[0])
    R[3]=1
    R[9]=-1

    return P,R

def print_policy(policy=(2, 2, 2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 2, 3, 2, 0, 0)):
    action=['^','<','>','v']
    print('+---+---+---+---+') 
    print('|',action[policy[0]],'|',action[policy[1]],'|',action[policy[2]],'|','S','|')
    print('+---+---+---+---+') 
    print('|',action[policy[4]],'|',action[policy[5]],'|',action[policy[6]],'|',action[policy[7]],'|')
    print('+---+---+---+---+') 
    print('|',action[policy[8]],'|','F','|',action[policy[10]],'|',action[policy[11]],'|')
    print('+---+---+---+---+') 
    print('|',action[policy[12]],'|',action[policy[13]],'|',action[policy[14]],'|',action[policy[15]],'|')
    print('+---+---+---+---+') 

def print_action(policy=np.ones(500),epsilon=None):
    b=[]
    for i in range(len(policy)):
        if policy[i]==1:
            b.append([255,0,0])
        else:
            b.append([0,0,255])
    result_action=np.array(b).reshape((25,20,3))
    fig,ax=plt.subplots(figsize=(500,4))
    ax.imshow(result_action)
    if epsilon is not None:
        ax.set_title('policy for espsilon='+str(epsilon))
    else:
        ax.set_title('optimal policy for PI')
    plt.show()

def print_action_large(policy=np.ones(500),title='optimal policy by default qlearning'):
    b=[]
    for i in range(len(policy)):
        if policy[i]==1:
            b.append([255,0,0])
        else:
            b.append([0,0,255])
    result_action=np.array(b).reshape((25,20,3))
    fig,ax=plt.subplots(figsize=(500,4))
    ax.imshow(result_action)
    ax.set_title(title)
    plt.show()