from regelum.system import System


class ThreeWheelPushingObject(System):
    _name = "pushing_object"
    _system_type = "gz"
    _dim_inputs = 1
    _dim_observation = _dim_state = 1200 # Robot pose + Flatten version of 20 x 20 x 3
    _parameters = {"m": 1, "g": 9.8, "l": 1}
    _observation_naming = _state_naming = ["angle [rad]", "angular velocity [rad/s]"]
    _inputs_naming = ["momentum [kg*m/s]"]
    _action_bounds = [[-20.0, 20.0]]

    def __init__(self, system_parameters_init = None, state_init = None, inputs_init = None):
        super().__init__(system_parameters_init, state_init, inputs_init)

    def _compute_state_dynamics(self, time, state, inputs):
        return
