import argparse
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tools.project_config_tools import host_device_count

command_template = "CUDA_VISIBLE_DEVICES=$CUDA python -m sample.generate_multi_round_in_scene "\
                   "--model_path $MODEL_PATH --num_repetitions 1 --output_dir $SAVE_PATH "\
                   "--use_scene_diffusion --inpainting --multi_round_demo_id $DEMO_ID --generate_mode eval --compile_model"


def get_replica_structure(required_action_obj_list:list = []):
    from tools.project_config_tools import replica_dir, replica_room_list
    replica_dir = Path(replica_dir)
    replica_structure = {}
    for room in replica_room_list:
        pose_root_dir = replica_dir / "gmd_poses" / room
        replica_structure[room] = {}
        for act_obj in required_action_obj_list:
            pair_candidates = list(pose_root_dir.glob(f"{act_obj}-*"))
            pair_candidates.sort()
            replica_structure[room][act_obj] = []
            for pair_idx, act_obj_id in enumerate(pair_candidates):
                pair_path = pose_root_dir / act_obj_id
                pkl_candidates = list(pair_path.glob(f"selected/*.pkl"))
                replica_structure[room][act_obj].append(len(pkl_candidates))
    return replica_structure


def get_replica_demo_id(required_pose_list:list = []):
    replica_structure = get_replica_structure(required_pose_list)
    demo_id_list = {}

    def generate_demo_id(room:str = "", idx:int = 0, action:str = ""):
        if idx == len(required_pose_list):
            demo_id_list[room].append(f"{room}+{action}+0")
        else:
            act_obj = required_pose_list[idx]
            for pair_id, pkl_num in enumerate(replica_structure[room][act_obj]):
                for pkl_id in range(pkl_num):
                    if action != "":
                        next_action = f"_{act_obj}-{pair_id}-{pkl_id}"
                    else:
                        next_action = f"{act_obj}-{pair_id}-{pkl_id}"
                    generate_demo_id(room, idx + 1, action + next_action)

    for room in replica_structure:
        demo_id_list[room] = []
        generate_demo_id(room, 0, "")

    return demo_id_list       
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./save/control/model001000000.pt")
    parser.add_argument("--save_dir", type=str, default="./save/control/")
    args = parser.parse_args()

    required_pose_lists = [
        ["sit-bed", "stand-floor", "sit-chair"], 
        ["sit-sofa", "stand-floor", "sit-chair"],
        ["sit-chair", "stand-floor", "sit-chair"],
        ["sit-chair", "stand-floor", "sit-sofa"],
        ["sit-sofa", "stand-floor", "sit-sofa"],
        ["sit-chair", "stand-floor", "sit-chair"],
        ["sit-chair", "stand-floor", "sit-sofa"],
        ["sit-bed", "stand-floor", "sit-bed"]
    ]

    demo_id = {}
    for required_pose_list in required_pose_lists:
        new_demo_id = get_replica_demo_id(required_pose_list)
        for k, v in new_demo_id.items():
            if k not in demo_id:
                demo_id[k] = []
            demo_id[k] += v

    selected_demo_id = []
    for room in demo_id:
        if not demo_id[room]:
            continue
        selected_demo_id += list(np.random.choice(demo_id[room], min(8, len(demo_id[room]))))
    
    print("select %d demo" % len(selected_demo_id))

    cmd = command_template.replace("$MODEL_PATH", args.model_path)
    pool = Pool(4)
    cnt = 0
    for demo_id in selected_demo_id:
        room = demo_id.split("+")[0]
        cur_cmd = cmd.replace("$CUDA", str(cnt))\
                     .replace("$SAVE_PATH", os.path.join(args.save_dir, "generated_motions", f"multi_round_motion_in_{room}"))\
                     .replace("$DEMO_ID", demo_id)
        cnt = (cnt + 1) % host_device_count
        pool.apply_async(os.system, args=(cur_cmd, ))
    pool.close()
    pool.join()

        

