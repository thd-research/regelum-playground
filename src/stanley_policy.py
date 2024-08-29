from regelum.policy import Policy
from math import cos, sin, atan2
import numpy as np
from src.trajectory import TrajectoryGenerator


class StanleyController(Policy):
    def __init__(self, 
                 control_gain=2.5, 
                 softening_gain=1.0, 
                 yaw_rate_gain=0.0, 
                 steering_damp_gain=0.0, 
                 action_bounds=[[-1.2, 1.2], [-np.deg2rad(24), np.deg2rad(24)]],
                 system=None,
                 trajectory_gen: TrajectoryGenerator=None):
        
        """
        Stanley Controller

        At initialisation
        :param control_gain:                (float) time constant [1/s]
        :param softening_gain:              (float) softening gain [m/s]
        :param yaw_rate_gain:               (float) yaw rate gain [rad]
        :param steering_damp_gain:          (float) steering damp gain
        :param action_bounds:               (float) vehicle's linear velocity [m/s] and steering limits [rad]
        :param path_x:                      (numpy.ndarray) list of x-coordinates along the path
        :param path_y:                      (numpy.ndarray) list of y-coordinates along the path
        :param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
        :param dt:                          (float) discrete time period [s]

        At every time step
        :param x:                           (float) vehicle's x-coordinate [m]
        :param y:                           (float) vehicle's y-coordinate [m]
        :param yaw:                         (float) vehicle's heading [rad]
        :param target_velocity:             (float) vehicle's velocity [m/s]
        :param steering_angle:              (float) vehicle's steering angle [rad]

        :return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
        :return target_index:               (int) closest path index
        :return crosstrack_error:           (float) distance from closest path index [m]
        """

        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.wheelbase = system.wheelbase
        self.steering_angle = 0

        self.trajectory_gen = trajectory_gen
        self.px, self.py, self.pyaw = self.trajectory_gen.trajectory

        super().__init__(system=system, action_bounds=action_bounds)

    def find_target_path_id(self, x, y, yaw):  

        # Calculate position of the front axle
        fx = x + self.wheelbase * cos(yaw)
        fy = y + self.wheelbase * sin(yaw)

        dx = fx - self.px    # Find the x-axis of the front axle relative to the path
        dy = fy - self.py    # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy) # Find the distance from the front axle to the path
        target_index = np.argmin(d) # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index]

    def calculate_yaw_term(self, target_index, yaw):

        # yaw_error = self.pyaw[target_index] - yaw
        # while yaw_error > np.pi:
        #         yaw_error -= 2 * np.pi
        # while yaw_error < -np.pi:
        #         yaw_error += 2 * np.pi
        yaw_error = self.normalise_angle(self.pyaw[target_index] - yaw)

        return yaw_error

    def calculate_crosstrack_term(self, target_velocity, yaw, dx, dy, absolute_error):

        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = np.sign(nearest_path_vector@front_axle_vector) * absolute_error

        crosstrack_steering_error = atan2((self.k * crosstrack_error), (self.k_soft + target_velocity))

        return crosstrack_steering_error, crosstrack_error

    def calculate_yaw_rate_term(self, target_velocity, steering_angle):

        yaw_rate_error = self.k_yaw_rate*(-target_velocity*sin(steering_angle))/self.wheelbase

        return yaw_rate_error

    def calculate_steering_delay_term(self, computed_steering_angle, previous_steering_angle):

        steering_delay_error = self.k_damp_steer*(computed_steering_angle - previous_steering_angle)

        return steering_delay_error
    
    def speed_computation(self, yaw_error, target_speed):
        
        #ReLu function
        ReLu = 1 / (max(0.0, abs(yaw_error*10)))
        
        # print([ReLu, yaw_error])
         
        return ReLu * target_speed + 0.1

    def stanley_control(self, x, y, yaw, target_velocity, steering_angle):
        target_index, dx, dy, absolute_error = self.find_target_path_id(x, y, yaw)
        yaw_error = self.calculate_yaw_term(target_index, yaw)
        crosstrack_steering_error, crosstrack_error = self.calculate_crosstrack_term(target_velocity, yaw, dx, dy, absolute_error)
        #yaw_rate_damping = self.calculate_yaw_rate_term(target_velocity, steering_angle)
        
        desired_steering_angle = yaw_error + crosstrack_steering_error# + yaw_rate_damping

        # Constrains steering angle to the vehicle limits
        desired_steering_angle += self.calculate_steering_delay_term(desired_steering_angle, steering_angle)
        limited_steering_angle = np.clip(desired_steering_angle, self.action_bounds[1, 0], self.action_bounds[1, 1])
        
        # p_speed = self.p_controller(self.speed, target_velocity)
        current_speed = self.speed_computation(yaw_error, target_velocity)
        current_speed = min(current_speed, 0.5)
        # print(crosstrack_steering_error)

        return limited_steering_angle, target_index, crosstrack_error, dx, dy, current_speed
    
    def get_action(self, observation: np.ndarray):
        """
        Same as :func:`~CtrlNominal3WRobotNI.compute_action`, but without invoking the internal clock.

        """
                
        (self.steering_angle, 
         target_index, 
         crosstrack_error, 
         dx, 
         dy, 
         current_speed) = self.stanley_control(*observation[0], self.action_bounds[0, 1], self.steering_angle)

        return np.array([[current_speed, self.steering_angle]])
    
    def normalise_angle(self, angle):
        return atan2(sin(angle), cos(angle))
