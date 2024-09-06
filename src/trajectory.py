import numpy as np


class TrajectoryGenerator():
    """Estimates state via adding reference to observation."""

    def __init__(self, 
                 init_state, 
                 trajectory_type: str="Sine"):
        """Instatiate ObserverReference.

        Args:
            reference (Union[np.ndarray, List[float]]): array for
                reference
        """
        # self.init_state = np.array(reference).reshape(1, -1)
        self.init_state = np.array(init_state).reshape(1, -1)

        if trajectory_type == "Sine":
            self.trajectory = trajectory_sine_gen(*init_state[0, :2])
        else:
            raise KeyError

    def get_nearest_point(self, state):
        id = self.get_nearest_idx(state)
        p = self.trajectory[id]
        return [*p], id
    
    def get_nearest_idx(self, state):
        dist = np.linalg.norm(self.trajectory - state)
        return np.argmin(dist)
    
    def is_last_point(self, index):
        return index >= len(self.trajectory[0]) - 1
    
    @classmethod
    def normalise_angle(self, angle):
        return np.atan2(np.sin(angle), np.cos(angle))


def trajectory_sine_gen(x_initial=-3, y_initial=-3, return_vect=False):
    print("trajectory_sine_gen:", x_initial, y_initial)
    x_ref = np.linspace(0, 10, 100)
    y_ref = 2*np.sin(2 * np.pi * x_ref * 0.15)

    # theta_ref = np.arctan2(np.diff(x_ref), np.diff(y_ref)) # dependencies: x_ref[-1], x_ref[-2], y_ref[-1], y_ref[-2]
    theta_ref = np.arctan2(np.diff(y_ref), np.diff(x_ref)) # dependencies: x_ref[-1], x_ref[-2], y_ref[-1], y_ref[-2]
    theta_ref = np.append(theta_ref, theta_ref[-1])

    x_ref = x_ref + (x_initial - x_ref[0])
    y_ref = y_ref + (y_initial - y_ref[0])

    if not return_vect:
        return x_ref, y_ref, theta_ref
    else:
        return np.vstack([x_ref, y_ref, theta_ref])
