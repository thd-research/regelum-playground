# python3.10 run.py \
#            policy=3wrobot_kin_min_grad_clf \
#            initial_conditions=3wrobot_kin \
#            system=3wrobot_kin \
#            common.sampling_time=0.01

python3.10 run.py \
           simulator=ros \
           policy=3wrobot_kin_min_grad_clf \
           initial_conditions=3wrobot_kin_customized \
           system=3wrobot_kin \
           common.sampling_time=0.02