# python3.10 run_ros.py \
#            common.sampling_time=0.01 \
#            --interactive --fps=10 --jobs=1

python3.10 run.py \
        simulator=ros_qcar \
        scenario=stanley_scenario \
        policy=qcar_kin_stanley \
        initial_conditions=qcar_kin \
        common=qcar_kin \
        system=qcar_kin_with_trajectory \
        common.sampling_time=.1 \
        common.time_final=100 \
        --interactive --fps=20
