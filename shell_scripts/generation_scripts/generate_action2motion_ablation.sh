export ABLATION=$1
case ${ABLATION} in
    'inversion')
        export DEMO_ID="MPH8+walk_sit-bed_lie-bed+0"
        export DATASET="prox"
        #generate multi-round in scene without inversion
        python -m sample.generate_multi_round_in_scene --model_path ./save/trained_models/mixed_action2motion_control/model001008525.pt --num_repetitions 1 --output_dir /home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/trained_models/mixed_action2motion_control/generated_motions_direct_guide/multi_round_motion_in_${DATASET} --guide_via_x_t --use_scene_diffusion --inpainting --max_lasting_frames 10 --multi_round_demo_id ${DEMO_ID}
        ;;
    'inpainting')
        #generate multi-round in scene without inpainting 
        export TEST_SCENE_TYPE=shapenet_scene_test_lie
        python -m python_scripts.sample.generate_for_eval --model_path ./save/trained_models/mixed_action2motion_control/model001008525.pt --num_repetitions 10 --output_dir /home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/results_for_eval/mixed_action2motion_control_noinpainting/${TEST_SCENE_TYPE} --use_scene_diffusion --test_scene_type ${TEST_SCENE_TYPE}
        ;;
    *)
        echo "invalid ablation study"
        exit 1
        ;;
esac
