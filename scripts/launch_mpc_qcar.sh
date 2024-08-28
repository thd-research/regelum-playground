
python3.10 run.py \
            simulator=ros_qcar \
            system=qcar_kin \
            scenario=mpc_scenario_customized \
            scenario.prediction_horizon=15 \
            scenario.prediction_step_size=5 \
            common.sampling_time=.2 \
            simulator.time_final=50 \
            --interactive
