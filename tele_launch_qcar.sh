# python3.10 run_ros.py \
#            common.sampling_time=0.01 \
#            --interactive --fps=10 --jobs=1

python3.10 run.py \
           simulator=ros_qcar \
           scenario=ros_scenario \
           policy=qcar_kin_nominal \
           system=qcar_kin \
           common.sampling_time=.1 \
           --interactive --fps=10
