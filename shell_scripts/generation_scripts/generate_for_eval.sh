export TEST_SCENE_TYPE=shapenet_scene_test_sit
python -m python_scripts.sample.generate_for_eval --model_path ./save/trained_models/mixed_action2motion_control/model001008525.pt --num_repetitions 10 --output_dir ./save/results_for_eval/dip/${TEST_SCENE_TYPE} --use_scene_diffusion --inpainting --max_lasting_frames 10 --test_scene_type ${TEST_SCENE_TYPE}
