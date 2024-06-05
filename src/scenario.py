import numpy as np

from typing import Optional

from regelum import RegelumBase
from regelum.policy import Policy
from regelum.system import System
from regelum.constraint_parser import ConstraintParser, ConstraintParserTrivial
from regelum.objective import RunningObjective
from regelum.observer import Observer, ObserverTrivial
from regelum.utils import Clock, AwaitedParameter, calculate_value

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist

import transformations as tftr 
import threading
import math


class Scenario(RegelumBase):
    def __init__(
        self,
        policy: Policy,
        system: System,
        sampling_time: float = 0.1,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
        discount_factor: float = 1.0,
    ):
        super().__init__()
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.time_old = 0
        self.delta_time = 0
        self.value: float = 0.0

        self.system = system
        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.value_threshold = value_threshold
        self.discount_factor = discount_factor
        self.is_episode_ended = False
        self.constraint_parser = (
            ConstraintParserTrivial()
            if constraint_parser is None
            else constraint_parser
        )
        self.observer = observer if observer is not None else ObserverTrivial()


        self.state = np.zeros(3)
        self.action = self.action_init
        self.observation = AwaitedParameter(
            "observation", awaited_from=self.system.get_observation.__name__
        )

        self.policy = policy
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time)
        self.iteration_counter: int = 0
        self.episode_counter: int = 0
        self.step_counter: int = 0
        self.action_old = AwaitedParameter(
            "action_old", awaited_from=self.compute_action.__name__
        )
        self.running_objective = lambda x: 0

        self.RATE = rospy.get_param('/rate', 10)
        self.lock = threading.Lock()

        # Topics
        
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odometry_callback)

        self.reset()

    def odometry_callback(self, msg):
        self.lock.acquire()
        
        self.get_velocity(msg)
        # Read current robot state
        x = msg.pose.pose.position.x

        # Complete for y and orientation
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
          
        # Transform quat2euler using tf transformations: complete!
        current_rpy = tftr.euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Extract Theta (orientation about Z)
        theta = current_rpy[0]
        
        self.state = [x, y, theta]

        if self.is_continue_update():
            if np.linalg.norm(np.array(self.state)[:2] - self.state_goal[:2]) < 0.1:
                self.state_goal = self.get_next_state_goal()
                print("self.state_goal: ", self.state_goal[:2])
        
        # Make transform matrix from 'robot body' frame to 'goal' frame
        
        theta_goal = self.state_goal[2]
        
        # Complete rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta_goal), -np.sin(theta_goal), 0],
            [np.sin(theta_goal), np.cos(theta_goal), 0],
            [0, 0, 1]
        ])
            
        state_matrix = np.vstack([self.state_goal[0], self.state_goal[1], 0])  # [x, y, 0] -- column   
        
        # Compute Transformation matrix 
        self.t_matrix = np.block([
            [rotation_matrix, state_matrix],
            [np.array([0, 0, 0, 1])]
            ])
        
        # Complete rotation counter for turtlebot3
        ''' Your solution goes here (rotation_counter) '''        
        if math.copysign(1, self.prev_theta) != math.copysign(1, theta) and \
            abs(self.prev_theta) > np.pi:
            if math.copysign(1, self.prev_theta) == -1:
                self.rotation_counter -= 1
            
            else:
                self.rotation_counter += 1
        
        self.prev_theta = theta
        theta = theta + 2 * np.pi * self.rotation_counter
        self.new_theta = theta
        
        # Orientation transform
        new_theta = theta - theta_goal
        
        # Do position transform
        
        ''' 
        Your solution goes here 
        self.new_state = using tranformations :) Have fun!
        '''
        temp = np.array([x, y , 0, 1])
        self.new_state = np.linalg.inv(self.t_matrix) @ temp.T
        self.new_state = [self.new_state[0], self.new_state[1], new_theta]
        
        self.lock.release()

    def run(self):
        for iteration_counter in range(1, self.N_iterations + 1):
            for episode_counter in range(1, self.N_episodes + 1):
                self.run_episode(
                    episode_counter=episode_counter, iteration_counter=iteration_counter
                )
                self.reload_scenario()

            self.reset_iteration()
            if self.sim_status == "simulation_ended":
                break

    def get_action_from_policy(self):
        return self.simulator.system.apply_action_bounds(self.policy.action)

    def run_episode(self, episode_counter, iteration_counter):
        self.episode_counter = episode_counter
        self.iteration_counter = iteration_counter
        rate = rospy.Rate(self.RATE)

        self.episode_start = rospy.get_time()

        while self.sim_status != "episode_ended":
            self.sim_status = self.step()
            rate.sleep()

    def step(self):
        if (not self.is_episode_ended) and \
            (self.value <= self.value_threshold):
            (
                self.time,
                self.state,
                self.observation,
                self.simulation_metadata,
            ) = self.simulator.get_sim_step_data()
            self.time = rospy.get_time() - self.episode_start
            self.observation = self.state = self.new_state

            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time
            
            estimated_state = self.observer.get_state_estimation(
                self.time, self.observation, self.action
            )

            self.action = self.compute_action_sampled(
                self.time,
                estimated_state,
                self.observation,
            )
            
            # Publish action 
            velocity = Twist()

            # Generate ROSmsg from action
            velocity.linear.x = self.action[0]
            velocity.angular.z = self.action[1]
            self.pub_cmd_vel.publish(velocity)

            self.is_episode_ended = self.simulator.do_sim_step() == -1

            return "episode_continues"
        else:
            return "episode_ended"

    @apply_callbacks()
    def reset_iteration(self):
        pass

    @apply_callbacks()
    def reload_scenario(self):
        self.is_episode_ended = False
        self.recent_value = self.value
        self.observation = self.simulator.observation
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.action = self.action_init
        self.simulator.reset()
        self.reset()
        self.sim_status = 0
        return self.recent_value

    @apply_callbacks()
    def post_compute_action(self, observation, estimated_state):
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "time": self.time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
            "action": self.get_action_from_policy(),
            "running_objective": self.current_running_objective,
            "current_value": self.value,
        }

    def compute_action_sampled(self, time, estimated_state, observation):
        self.is_time_for_new_sample = self.clock.check_time(time)
        if self.is_time_for_new_sample:
            self.on_observation_received(time, estimated_state, observation)
            action = self.system.apply_action_bounds(
                self.compute_action(
                    time=time,
                    estimated_state=estimated_state,
                    observation=observation,
                )
            )
            self.post_compute_action(observation, estimated_state)
            self.step_counter += 1
            self.action_old = action
        else:
            action = self.action_old
        return action

    def compute_action(self, time, estimated_state, observation):
        self.issue_action(observation)
        return self.get_action_from_policy()

    def issue_action(self, observation):
        self.policy.update_action(observation)

    def __getattribute__(self, name):
        if name == "issue_action":
            return self._issue_action
        else:
            return object.__getattribute__(self, name)

    def _issue_action(self, observation, *args, **kwargs):
        object.__getattribute__(self, "issue_action")(observation, *args, **kwargs)
        self.on_action_issued(observation)

    def on_action_issued(self, observation):
        self.current_running_objective = self.running_objective(
            observation, self.get_action_from_policy()
        )
        self.value = self.calculate_value(self.current_running_objective, self.time)
        observation_action = np.concatenate(
            (observation, self.get_action_from_policy()), axis=1
        )
        return {
            "action": self.get_action_from_policy(),
            "running_objective": self.current_running_objective,
            "current_value": self.value,
            "observation_action": observation_action,
        }

    def on_observation_received(self, time, estimated_state, observation):
        self.time = time
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "time": time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
        }

    def substitute_constraint_parameters(self, **kwargs):
        self.policy.substitute_parameters(**kwargs)

    def calculate_value(self, running_objective: float, time: float):
        value = (
            self.value
            + running_objective * self.discount_factor**time * self.sampling_time
        )
        return value

    def reset(self):
        """Reset agent for use in multi-episode simulation.

        Only __internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.clock.reset()
        self.value = 0.0
        self.is_first_compute_action_call = True
