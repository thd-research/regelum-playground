#!/bin/bash
 
## define an array with three items ##
controllers=( sac td3 )
# controllers=( td3 )
tasks=( run_po_sim.sh run_lf_sim.sh run_rp_sim.sh )
# tasks=( run_rp_sim.sh )
reset_replay_buffer=( false ) 
replay_buffer_size=( 16000 4000 ) 

## get item count using ${arrayname[@]} ##
for c in "${controllers[@]}"; do
    for t in "${tasks[@]}"; do
        for r in "${reset_replay_buffer[@]}"; do
            for s in "${replay_buffer_size[@]}"; do
         
                echo "${c} - ${t}" ${r} ${s}
                source scripts/${c}/${t} ${r} ${s}
                sleep 10s
            
            done
        done

    done
done