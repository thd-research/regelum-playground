# python3.10 run_ros.py \
#            common.sampling_time=0.01 \
#            --interactive --fps=10 --jobs=1



if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then
        python3.10 run.py \
                simulator=ros_qcar \
                scenario=ros_scenario \
                policy=qcar_kin_nominal \
                system=qcar_kin \
                common.sampling_time=.1 \
                --interactive --fps=10
    else
        python3.10 run.py \
                scenario=scenario \
                policy=qcar_kin_nominal \
                system=qcar_kin \
                common.sampling_time=.1 \
                --interactive --fps=20
fi
