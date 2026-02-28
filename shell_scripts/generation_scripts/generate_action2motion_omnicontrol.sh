export ROUND=$1
case $ROUND in
    'single')
        #generate single-round in scene
        #python -m sample.generate_single_round_in_scene --model_path ./save/trained_models/mixed_action2motion_control/model001008525.pt --num_repetitions 1 --output_dir /home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/trained_models/mixed_action2motion_control/generated_motions/single_round_motion_in_scenes --use_scene_diffusion --single_round_demo_id "test_room_walk"
        ;;
    'multi')
        #generate multi-round in scene
        export DEMO_ID="room_0+sit-chair_sit-sofa+0"
        export DATASET="replica"
        python -m sample.generate_multi_round_in_scene --model_path ./save/trained_models/mixed_action2motion_control/model001008525.pt --num_repetitions 1 --output_dir /home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/others_results/omnicontrol_results/results_${DATASET} --guide_via_x_t --max_lasting_frames 10 --multi_round_demo_id ${DEMO_ID}
        ;;
    *)
        echo "invalid round"
        exit 1
        ;;
esac
