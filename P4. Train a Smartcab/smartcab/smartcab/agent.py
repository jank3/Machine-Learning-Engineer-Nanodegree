import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
import pandas as pd

stateRecord = namedtuple('stateRecord', ['light','oncoming','right','left','nextWaypoint'])
validActions = [None, 'forward', 'left', 'right']

def argMax(Q, s):
        retA = None
        retV = 0
        if s in Q:
            for a in Q[s]:
                if Q[s][a] > retV:
                    retV = Q[s][a]
                    retA = a
        return(retA, retV)                    
        
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.20
        self.explore = 0.99
        self.stateHist = {}
        self.counter = 0
        self.rewards = 0
        self.trip = 1
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if self.counter > 0:
            df = pd.DataFrame(self.stateHist).T
            df.columns = ['trip','light','oncoming','left','right','next_waypoint',
                          'action', 'qA', 'qV', 'reward', 'alpha', 'deadline','Explored']            
            df.to_pickle('runHist.pkl')
            self.trip += 1
            self.counter = 0
                    
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        self.state = stateRecord(inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        if self.state not in self.Q:
            self.Q[self.state] = {'left':0, 'right':0, 'forward':0, None:0}
        
        self.counter += 1
        
        
        # TODO: Select action according to your policy  
        
        qA, qV = argMax(self.Q, self.state)
        choseExpl = False
        if random.randrange(0, 100)/100.0 < self.explore or qV == 0:
            action = validActions[random.randint(0,3)]
            if self.explore > 0.00:
                self.explore = self.explore  - 0.01
            choseExpl = True
        else:
            action = qA
            
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        s = stateRecord(inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        
        self.Q[self.state][action] = self.alpha * (reward + self.gamma * argMax(self.Q, s)[1])
        
        writeout = 'Lights:{} Goal:{} '.format( str(self.state[0]),str(self.state[4])) 
        writeout += 'Action:{} Reward:{} Qest:{} Expl:{} Exp:{}'.format(str(action), str(reward), str(self.gamma * argMax(self.Q, s)[1]), choseExpl, self.explore)
        
        print(writeout)
        
        self.rewards += reward
        self.stateHist[self.trip * 1000 + self.counter] = [self.trip, 
                            inputs['light'], inputs['oncoming'], inputs['left'], 
                            inputs['right'], self.next_waypoint, 
                            action, qA, qV, reward, self.gamma, deadline, choseExpl]

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, step = {}".format(deadline, inputs, action, reward, self.counter)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
