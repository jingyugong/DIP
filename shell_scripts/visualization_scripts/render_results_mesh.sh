export DATASET="prox"
export METHOD_PATH="trained_models/mixed_action2motion_control/generated_motions_ortho/multi_round_motion_in" #"others_results/omnicontrol_results/results"
export DEMO_ID="MPH16+sit-bed_walk_sit-chair+0"
python -m python_scripts.visualization.render_results_smplx_mesh --npy_path /home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/${METHOD_PATH}_${DATASET}/${DEMO_ID}/results.npy --sample_i 0 --rep_i 0
