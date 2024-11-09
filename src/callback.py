import pandas as pd
from regelum.callback import (
    HistoricalDataCallback,
    )


class GZHistoricalDataCallback(HistoricalDataCallback):
    def on_episode_done(self, scenario, episode_number, episodes_total, iteration_number, iterations_total):
        if episodes_total == 1:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}"
        else:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}_ep_{str(episode_number).zfill(5)}"
        try:
            self.dump_and_clear_data(identifier)
        except Exception as err:
            print("GZHistoricalDataCallback got Error:", err)

    def on_function_call(self, obj, method, output):
        if self.observation_components_naming is None:
            self.observation_components_naming = (
                [
                    f"observation_{i + 1}"
                    for i in range(obj.simulator.system.dim_observation)
                ]
                if obj.simulator.system.observation_naming is None
                else obj.simulator.system.observation_naming
            )

        if self.action_components_naming is None:
            self.action_components_naming = (
                [f"action_{i + 1}" for i in range(obj.simulator.system.dim_inputs)]
                if obj.simulator.system.inputs_naming is None
                else obj.simulator.system.inputs_naming
            )

        if self.state_components_naming is None:
            self.state_components_naming = (
                [f"state_{i + 1}" for i in range(obj.simulator.system.dim_state)]
                if obj.simulator.system.state_naming is None
                else obj.simulator.system.state_naming
            )

        if method == "post_compute_action":
            self.add_datum(
                {
                    **{
                        "time": output["time"],
                        "running_objective": output["running_objective"],
                        "current_value": output["current_value"],
                        "current_undiscounted_value": output[
                            "current_undiscounted_value"
                        ],
                        "episode_id": output["episode_id"],
                        "iteration_id": output["iteration_id"],
                        "task_name": output["task_name"],
                        "step_id": output["step_id"],
                        "phase": output["phase"],
                        "exploration": output["exploration"],
                    },
                    **dict(zip(self.action_components_naming, output["action"][0])),
                    **dict(
                        zip(self.state_components_naming, output["estimated_state"][0])
                    ),
                    # **dict(
                    #     zip(self.state_components_naming, output["estimated_state"][0])
                    # ),
                }
            )
        elif method == "dump_data_buffer":
            _, data_buffer = output
            self.data = pd.concat(
                [
                    data_buffer.to_pandas(
                        keys={
                            "time": float,
                            "running_objective": float,
                            "current_value": float,
                            "current_undiscounted_value": float,
                            "episode_id": int,
                            "iteration_id": int,
                        }
                    )
                ]
                + [
                    pd.DataFrame(
                        columns=columns,
                        data=np.array(
                            data_buffer.to_pandas([key]).values.tolist(),
                            dtype=float,
                        ).squeeze(),
                    )
                    for columns, key in [
                        (self.action_components_naming, "action"),
                        (self.state_components_naming, "estimated_state"),
                        # (self.state_components_naming, "estimated_state"),
                    ]
                ],
                axis=1,
            )