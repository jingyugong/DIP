export ROUND=$1
case $ROUND in
    'single')
        #generate single-round in scene
        python -m sample.generate_single_round_in_scene --model_path ./save/trained_models/mixed_action2motion_control/model001008525.pt --num_repetitions 1 --output_dir /home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/trained_models/mixed_action2motion_control/generated_motions/single_round_motion_in_scenes --use_scene_diffusion --single_round_demo_id "test_room_walk"
        ;;
    'multi')
        #generate multi-round in scene
        export DATASET="prox"
        #export DEMO_ID="BasementSittingBooth+sit-sofa_walk+0"
        for DEMO_ID in "MPH16+sit-bed_walk_sit-chair+0" "BasementSittingBooth+sit-sofa_walk+0" "MPH112+walk_sit-bed+0" "MPH11+walk_sit-sofa+0" "MPH11+walk_sit-sofa_lie-sofa+0" "MPH1Library+walk+0" "MPH8+walk_sit-bed+0" "MPH8+walk_sit-bed_lie-bed+0" "N0Sofa+walk_sit-sofa_walk+0" "N3Library+walk_sit-chair+0" "N3Office+sit-chair_walk+0" "N3OpenArea+lie-sofa_sit-sofa_walk_sit-chair+0" "N3OpenArea+sit-sofa_walk_sit-chair+0" "Werkraum+walk_sit-chair_walk+0"
        do
            python -m sample.generate_multi_round_in_scene --model_path ./save/trained_models/mixed_action2motion_control/model001008525.pt --num_repetitions 1 --output_dir /home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/trained_models/mixed_action2motion_control/generated_motions_ortho/multi_round_motion_in_${DATASET} --use_scene_diffusion --inpainting --max_lasting_frames 10 --multi_round_demo_id ${DEMO_ID}
        done
        ;;
    *)
        echo "invalid round"
        exit 1
        ;;
esac
