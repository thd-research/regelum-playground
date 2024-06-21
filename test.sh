python3.10 run.py \
            simulator=ros \
            initial_conditions=3wrobot_kin_customized \
            system=3wrobot_kin \
            scenario=mpc_scenario_customized \
            scenario.prediction_horizon=8 \
            scenario.prediction_step_size=10 \
            common.sampling_time=.1 \
            --interactive \
            --fps=10
