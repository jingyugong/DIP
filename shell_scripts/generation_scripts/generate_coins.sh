export DATASET="shapenet"
case $DATASET in
    'general')
        python -m python_scripts.sample.sample_pose_general --exp_name test --lr_posa 0.01 --max_step_body 100  --weight_penetration 100 --weight_pose 10 --weight_init 0.01  --weight_contact_semantic 1 --num_sample 8 --num_try 8  --visualize 1 --full_scene 1 --action 'sit on' --obj_path 'dataset/dimos_data/test_room/sofa.ply' --obj_category 'sofa' --obj_id 0 --scene_path 'dataset/dimos_data/test_room/room.ply' --scene_name 'test_room' --data_name 'general'
        ;;
    'shapenet')
        python -m python_scripts.sample.sample_pose_general --exp_name test --lr_posa 0.01 --max_step_body 100  --weight_penetration 100 --weight_pose 10 --weight_init 0.01  --weight_contact_semantic 1 --num_sample 32 --num_try 8  --visualize 1 --full_scene 1 --action 'lie on' --obj_path 'dataset/dimos_data/shapenet_real/Sofas/277231dcb7261ae4a9fe1734a6086750/model.obj' --obj_category 'sofa' --obj_id 0 --scene_path '' --scene_name 'test_room' --data_name 'shapenet'
        ;;
    'replica')
        python -m python_scripts.sample.sample_pose_general --exp_name test --lr_posa 0.01 --max_step_body 100  --weight_penetration 100 --weight_pose 10 --weight_init 0.01  --weight_contact_semantic 1 --num_sample 8 --num_try 8  --visualize 1 --full_scene 1 --action 'sit on' --obj_path '' --obj_category 'stool' --obj_id 1 --scene_path '' --scene_name 'room_0' --data_name 'replica'
        ;;
    'prox')
        python -m python_scripts.sample.sample_pose_two_stage --exp_name test --lr_posa 0.01 --max_step_body 100 --weight_penetration 100 --weight_pose 10 --weight_init 0.01  --weight_contact_semantic 1 --num_sample 16 --num_try 8  --visualize 1 --full_scene 1 --interaction "sit on-chair" --scene_name "Werkraum"
        ;;
    *)
        echo "invalid dataset"
        exit 1
        ;;
esac
