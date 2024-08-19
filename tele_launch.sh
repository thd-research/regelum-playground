# python3.10 run_ros.py \
#            common.sampling_time=0.01 \
#            --interactive --fps=10 --jobs=1

python3.10 run.py \
           simulator=ros \
           scenario=ros_scenario \
           policy=3wrobot_kin_nomial \
           initial_conditions=3wrobot_kin_customized \
           system=3wrobot_kin \
           common.sampling_time=.1 \
           --interactive --fps=10
