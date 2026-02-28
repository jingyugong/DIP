import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from tools.project_config_tools import host_device_count


interaction_dict = {
    'sit on':['chair', 'sofa', 'bed'],
    'stand on':['floor'],
    'lie on':['sofa', 'bed'],
}

sample_command_template = {
        "general":
            "python -m python_scripts.sample.sample_pose_general --exp_name test "
            "--lr_posa 0.01 --max_step_body 100  --weight_penetration 10 --weight_pose 10 --weight_init 0  --weight_contact_semantic 1 "
            "--num_sample $NUM --num_try 8  --visualize 1 --full_scene 1 "
            "--action '$ACTION' --obj_path 'dataset/dimos_data/test_room/sofa.ply' --obj_category '$OBJ' --obj_id $ID "
            "--scene_path '$PATH' --scene_name '$ROOM' --data_name 'general'",
        "replica":
            "CUDA_VISIBLE_DEVICES='$CUDA' python -m python_scripts.sample.sample_pose_general --exp_name test "
            "--lr_posa 0.01 --max_step_body 100 --weight_penetration 100 --weight_pose 10 --weight_init 0  --weight_contact_semantic 100 "
            "--num_sample $NUM --num_try 8  --visualize 1 --full_scene 1 --action '$ACTION' "
            "--obj_path '' --obj_category '$OBJ' --obj_id $ID --scene_path '' --scene_name '$ROOM' --data_name 'replica' ",
        "shapenet":
            "python -m python_scripts.sample.sample_pose_general --exp_name test "
            "--lr_posa 0.01 --max_step_body 100  --weight_penetration 10 --weight_pose 10 --weight_init 5  --weight_contact_semantic 10 "
            "--num_sample $NUM --num_try 8  --visualize 1 --full_scene 1 --action '$ACTION'  "
            "--obj_path $PATH --obj_category $OBJ --obj_id 0 "
            "--scene_path '' --scene_name 'test_room' --data_name 'shapenet'",
        "prox":
            "CUDA_VISIBLE_DEVICES='$CUDA' python -m python_scripts.sample.sample_pose_two_stage --exp_name test "
            "--lr_posa 0.01 --max_step_body 100 --weight_penetration 100 --weight_pose 10 --weight_init 0.01  --weight_contact_semantic 1 "
            "--num_sample $NUM --num_try 8  --visualize 1 --full_scene 1 --interaction '$ACTION-$OBJ' --scene_name $ROOM"
    }

THREAD_LIMITS = 6
INTERVAL = THREAD_LIMITS // host_device_count


def sample_for_replica(num:int):
    template = sample_command_template["replica"].replace("$NUM", str(num))
    from tools.project_config_tools import replica_dir, replica_room_list
    from tools.scene_tools import ReplicaSceneV2
    cnt = 0
    for room in tqdm(replica_room_list):
        room_command = template.replace("$ROOM", room)
        gen_pool = Pool(THREAD_LIMITS)
        scene = ReplicaSceneV2(room, replica_dir, False, False)
        for action, objects in interaction_dict.items():
            act_command = room_command.replace("$ACTION", action)
            for obj in objects:
                obj_command = act_command.replace("$OBJ", obj)
                for obj_id in range(len(scene.get_category_indices(obj))):
                    id_command = obj_command.replace("$ID", str(obj_id))
                    cuda_command = id_command.replace("$CUDA", str(cnt // INTERVAL))
                    cnt = (cnt + 1) % THREAD_LIMITS
                    gen_pool.apply_async(os.system, args=(cuda_command, ))
                    print("{%s} : generate {%s}-{%s}_{%s} in cuda {%d}" % (room, action, obj, obj_id, cnt // INTERVAL))
        gen_pool.close()
        gen_pool.join()


def sample_for_prox(num:int):
    template = sample_command_template["prox"].replace("$NUM", str(num))
    from tools.project_config_tools import prox_room_list
    cnt = 0
    for room in tqdm(prox_room_list):
        room_command = template.replace("$ROOM", room)
        gen_pool = Pool(THREAD_LIMITS)
        for action, objects in interaction_dict.items():
            act_command = room_command.replace("$ACTION", action)
            for obj in objects:
                obj_command = act_command.replace("$OBJ", obj)
                cuda_command = obj_command.replace("$CUDA", str(cnt // INTERVAL))
                print(cnt // INTERVAL)
                cnt = (cnt + 1) % THREAD_LIMITS
                gen_pool.apply_async(os.system, args=(cuda_command, ))
                print("generate for ", room, action, obj)
        gen_pool.close()
        gen_pool.join()


def sample_for_shapenet(num:int):
    template = sample_command_template["shapenet"].replace("$NUM", str(num))
    from tools.project_config_tools import shapenet_real_dir, shapenet_obj_list
    shapenet_real_dir = Path(shapenet_real_dir)
    cnt = 0
    for category in tqdm(shapenet_obj_list):
        gen_pool = Pool(THREAD_LIMITS)
        category_command = template.replace("$OBJ", category)
        for action, objects in interaction_dict.items():
            selected = any([obj.lower() in category.lower() for obj in objects])
            if not selected:
                continue
            action_command = category_command.replace("$ACTION", action)
            for shapenet_obj in shapenet_obj_list[category]:
                obj_path = shapenet_real_dir / category / shapenet_obj / 'model.obj'
                obj_command = action_command.replace("$PATH", str(obj_path))
                cuda_command = obj_command.replace("$CUDA", str(cnt // INTERVAL))
                print(cnt // INTERVAL)
                cnt = (cnt + 1) % THREAD_LIMITS
                gen_pool.apply_async(os.system, args=(cuda_command, ))
                print("generate for ", category, action, shapenet_obj)
        gen_pool.close()
        gen_pool.join()


def sample_for_general(num:int):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_datasets", default=["replica"], nargs="+")
    parser.add_argument("--num_sample", type=int ,default=8, help="the num of sample sequences for each action-object pair.")
    args = parser.parse_args()
    
    for dataset in args.process_datasets:
        sample_function = f"sample_for_{dataset}"
        globals()[sample_function](args.num_sample)
