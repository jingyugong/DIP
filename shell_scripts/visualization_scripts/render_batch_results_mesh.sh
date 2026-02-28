RESULTS_PATTERN=~/gcode/RGBD/code/guided-motion-diffusion/save/trained_models/mixed_action2motion_control/generated_motions_ortho/multi_round_motion_in_prox/*/results.npy
for SCENE_RESULT in ${RESULTS_PATTERN} 
do
    echo ${SCENE_RESULT}
    python -m python_scripts.visualization.render_results_smplx_mesh --npy_path ${SCENE_RESULT} --sample_i 0 --rep_i 0
done
