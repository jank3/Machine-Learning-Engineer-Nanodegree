import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple

stateRecord = namedtuple('stateRecord', ['light','oncoming','right','left','nextWaypoint'])

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qTable = {}
        self.counter = 0
        self.rewards = 0
        self.trip = 1
        self.runs = {}        
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if self.counter > 0:
            self.runs[self.trip] = {'Steps':self.counter,'Rewards':self.rewards}
            self.trip += 1
            self.counter = 0
            self.rewards += 1
            print self.runs

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        # First I will create a state to add to the qTable
        self.state = stateRecord(inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        if self.state not in self.qTable:
            self.qTable[self.state] = {'left':[], 'right':[], 'forward':[], None:[]}
        
        self.counter += 1
        
        
        # TODO: Select action according to your policy  
        
        if self.state in self.qTable:
            # I'm storing individual results and using the mean since in 
            # theory there could be an element of randomness that we would 
            # want to account for.
            possible = {}
            for x in self.qTable[self.state]:
                if len(self.qTable[self.state][x]) > 0:
                    possible[x] = np.mean(self.qTable[self.state][x])
                else:
                    possible[x] = random.randint(0, 2)
            
            action, r = None, 0
            for x in possible:
                if possible[x] > r:
                    action, r = x, possible[x]
            
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        self.rewards += reward
        self.qTable[self.state][action].append(reward)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
