from regelum import objective
import numpy as np
from typing import Union
from regelum.utils import rg
from regelum.model import ModelQuadLin

import matplotlib
import math


class ThreeWheeledRobotCostWithSpot(objective.RunningObjective):
    def __init__(
        self,
        quadratic_model: ModelQuadLin,
        spot_gain: float,
        spot_x_center: float,
        spot_y_center: float,
        spot_std: float,
    ):
        self.quadratic_model = quadratic_model
        self.spot_gain = spot_gain
        self.spot_x_center = spot_x_center
        self.spot_y_center = spot_y_center
        self.spot_std = spot_std

    def __call__(
        self,
        observation,
        action,
        is_save_batch_format: bool = False,
    ):
        spot_cost = (
            self.spot_gain
            * rg.exp(
                -(
                    (observation[:, 0] - self.spot_x_center) ** 2
                    + (observation[:, 1] - self.spot_y_center) ** 2
                )
                / (2 * self.spot_std**2)
            )
            / (2 * np.pi * self.spot_std**2)
        )

        quadratic_cost = self.quadratic_model(observation, action)
        cost = quadratic_cost + spot_cost

        if is_save_batch_format:
            return cost
        else:
            return cost[0, 0]

        # if is_save_batch_format:
        #     return rg.array(
        #         rg.array(quadratic_cost, prototype=observation),
        #         prototype=observation,
        #     )
        # else:
        #     return quadratic_cost


def angle_normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class GymPendulumRunningObjective:
    def __call__(self, observation, action):
        if observation.shape[1] == 3:
            cos_angle = observation[0, 0]
            sin_angle = observation[0, 1]
            angle_vel = observation[0, 2]
            torque = action[0, 0]
            angle = np.arctan2(sin_angle, cos_angle)
            return angle_normalize(angle) ** 2 + 0.1 * angle_vel**2 + 0.001 * torque**2
        elif observation.shape[1] == 2:
            angle = observation[0, 0]
            angle_vel = observation[0, 1]
            torque = action[0, 0]
            return angle_normalize(angle) ** 2 + 0.1 * angle_vel**2 + 0.001 * torque**2
        else:
            raise ValueError("Invalid observation shape")

class PushingObjectRunningObjective:
    def __call__(self, state, position, action, current_mass):
        if not hasattr(self, "x"):
            # for reward computation
            c2 = state.shape[0]
            self.x = np.arange(0,c2).reshape(1,c2) * np.ones([c2,1])
            self.y = np.ones([1,c2]) * np.arange(0,c2).reshape(c2,1)

        truncated = False
        terminated = False
        c = (state.shape[0] // 2)
        hsv = matplotlib.colors.rgb_to_hsv(state)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        colored_pixels = (s > 0.3).astype(np.int32)

        csum = colored_pixels.sum()
        # print(hsv.shape, "VMINMAX", "H", h.min(), h.max(), "S", s.min(), s.max(), "V", v.min(), v.max(), csum)

        cog_x = (self.x*colored_pixels).sum()  / csum if csum > 0 else 0.
 
        reward = 1. - math.fabs(cog_x - c) / c

        if csum < 5:  # LOST SIGHT OF OBJECT
            truncated = True
            print("COND: LOST")
            return -1.0, truncated, terminated

        #if position[0] >= 0.85 and math.fabs(position[1]-cube_pos) < 0.15 and reward > 0.5: 
        # # x coord of the robot (closeness to the cube) 0.9 ) collision with cube
        if position[0] >= 0.85 and reward > 0.6: # x coord of the robot (closeness to the cube) 0.9 ) collision with cube
            truncated = True
            if current_mass > 1: # we mean > 0 but safer this way
                reward = -10
            else:
                reward = 10
            print("COND: Pushed object")
            return reward, truncated, terminated
        
        #if position[0] >= 0.85 and math.fabs(position[1]-cube_pos) > 0.15:
        if position[0] >= 0.85 and reward < 0.6: # cube missed
          truncated=True
          terminated=False
          return reward,truncated,terminated ; 

        print("COND: Normal")
        modifier = 0.1 if all(np.isclose(action, np.zeros_like(action))) else 1.0 ; # punish stop action
        
        return reward * modifier, truncated, terminated
    
class LineFollowingRunningObjective:
    def __call__(self, observation):
        truncated = False

        gray_img = observation.mean(axis=2) ;
        c = observation.shape[1] ;
        black_mask = (gray_img[0,:] < 0.1) ;

        if black_mask.astype(np.int32).sum() < 1: # no black pixels visible
          terminated = True
          return -1., truncated, terminated

        x_indices = np.linspace(0.,c,c) ;
        black_indices = x_indices[black_mask] ;
        left_line_index = np.min(black_indices) ;
        
        ret = 1.- math.fabs(left_line_index-c//2) / (c//2) ;

        terminated = ret < 0.
        return ret, truncated, terminated


class RobotPursuitRunningObjective:
    def __call__(self, 
                 state, 
                 catcher_pos, 
                 runner_pos, 
                 action, 
                 step_count, 
                 max_steps_per_episode,
                 runner_collision_thickness,
                 arena_bounds,
                 info: dict):
        if not hasattr(self, "x"):
            # for reward computation
            c2 = state.shape[0]
            self.x = np.arange(0,c2).reshape(1,c2) * np.ones([c2,1])
            self.y = np.ones([1,c2]) * np.arange(0,c2).reshape(c2,1)

        truncated = False
        terminated = False
        c = (state.shape[0] // 2)
        hsv = matplotlib.colors.rgb_to_hsv(state)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        colored_pixels = (s > 0.3).astype(np.int32)

        csum = colored_pixels.sum()
        # print(hsv.shape, "VMINMAX", "H", h.min(), h.max(), "S", s.min(), s.max(), "V", v.min(), v.max(), csum)

        cog_x = (self.x*colored_pixels).sum()  / csum if csum > 0 else 0.
 
        reward = 1. - math.fabs(cog_x - c) / c

        if csum < 5:  # LOST SIGHT OF OBJECT
            truncated = True
            print("COND: LOST")
            return -1.0, truncated, terminated

        collider = runner_collision_thickness / 2.0

        if catcher_pos[0] <= runner_pos[0] + collider and \
                catcher_pos[0] >= runner_pos[0] - collider and \
                catcher_pos[1] <= runner_pos[1] + collider and \
                catcher_pos[1] >= runner_pos[1] - collider:
            truncated = True
            reward = 10
            info["terminate_cond"] = 'Cought_Runner'
        elif catcher_pos[0] < arena_bounds[0] or \
                catcher_pos[0] > arena_bounds[1] or \
                catcher_pos[1] < arena_bounds[2] or \
                catcher_pos[1] > arena_bounds[3]:
            truncated = True
            reward = -10
            info["terminate_cond"] = 'Left_Arena'
        elif csum < 1:  # LOST SIGHT OF OBJECT
            truncated = True
            info["terminate_cond"] = 'Lost_Runner'
            reward = -10
        elif step_count >= max_steps_per_episode:
            terminated = True
            info["terminate_cond"] ='Max_Steps_Reached'
            reward = -10
        else:
            info["terminate_cond"] = f"COND: Normal, REWARD: {reward}"
            modifier = 0.1 if all(np.isclose(action, np.zeros_like(action))) else 1.0 ; # punish stop action
            reward = reward*modifier
        
        print("[Objective Function]", info["terminate_cond"])
              
        return reward , truncated, terminated
