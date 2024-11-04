#!/bin/bash
 
## define an array with three items ##
controllers=( sac td3 )
# controllers=( td3 )
tasks=( run_lf_sim.sh run_po_sim.sh run_rp_sim.sh )
reset_replay_buffer=( true false ) 

## get item count using ${arrayname[@]} ##
for c in "${controllers[@]}"; do
    for t in "${tasks[@]}"; do
        for r in "${reset_replay_buffer[@]}"; do
         
        echo "${c} - ${t}" ${r}
        source scripts/${c}/${t} ${r}
        sleep 10s
        # do something on $m #
        done

    done
done