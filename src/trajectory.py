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

    def get_observation(self, state, action):
        return 


def trajectory_sine_gen(x_initial=-3, y_initial=-3):
    print("trajectory_sine_gen:", x_initial, y_initial)
    x_ref = np.linspace(0, 10, 100)
    y_ref = 2*np.sin(2 * np.pi * x_ref * 0.15)

    # theta_ref = np.arctan2(np.diff(x_ref), np.diff(y_ref)) # dependencies: x_ref[-1], x_ref[-2], y_ref[-1], y_ref[-2]
    theta_ref = np.arctan2(np.diff(y_ref), np.diff(x_ref)) # dependencies: x_ref[-1], x_ref[-2], y_ref[-1], y_ref[-2]
    theta_ref = np.append(theta_ref, theta_ref[-1])

    x_ref = x_ref + (x_initial - x_ref[0])
    y_ref = y_ref + (y_initial - y_ref[0])

    return x_ref, y_ref, theta_ref
