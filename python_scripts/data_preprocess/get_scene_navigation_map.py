import argparse
import os
import glob
from typing import List
from tqdm import tqdm
from pathlib import Path
from tools.navigation_tools import get_navmesh


navmesh_dict = {
    "loose" : 0.2,
    "middle" : 0.12,
    "tight" : 0.06,
}


def process_prox_scene_navigation_map():
    from tools.project_config_tools import prox_data_dir
    from tools.navigation_tools import get_prox_floor_height
    prox_scene_dir = os.path.join(prox_data_dir, "proxs", "scenes")
    prox_navmesh_dir = os.path.join(prox_data_dir, "proxs", "gmd_navmeshes")
    all_scene_paths = sorted(glob.glob(os.path.join(prox_scene_dir, "*.ply")))
    all_scene_names = [os.path.basename(scene_path).split(".")[0] for scene_path in all_scene_paths]
    print(all_scene_paths)
    for scene_name, scene_path in tqdm(zip(all_scene_names, all_scene_paths)):
        scene_navmesh_dir = os.path.join(prox_navmesh_dir, scene_name)
        os.makedirs(scene_navmesh_dir, exist_ok=True)
        floor_height = get_prox_floor_height(os.path.join(prox_data_dir, "proxs", "scene_segmentation", scene_name+".pkl"), scene_path)
        for level, radius in navmesh_dict.items():
            navmesh_path = Path(os.path.join(scene_navmesh_dir, scene_name+f"_navmesh_{level}.ply"))
            if navmesh_path.exists():
                continue
            navmesh = get_navmesh(navmesh_path, scene_path, agent_radius=radius, floor_height=floor_height)
    return


def process_replica_scene_navigation_map():
    from tools.project_config_tools import replica_dir, replica_room_list
    from tools.navigation_tools import get_replica_floor_height
    replica_dir = Path(replica_dir)
    replica_navmesh_dir = replica_dir / "gmd_navmeshes"
    for scene in tqdm(replica_room_list):
        print(scene)
        scene_dir = replica_dir / scene
        scene_navmesh_dir = replica_navmesh_dir / scene
        floor_height = get_replica_floor_height(scene_dir)
        for level, radius in navmesh_dict.items():
            navmesh_path = Path(os.path.join(scene_navmesh_dir, scene+f"_navmesh_{level}.ply"))
            if navmesh_path.exists():
                continue
            navmesh = get_navmesh(navmesh_path, scene_dir / 'mesh.ply', agent_radius=radius, floor_height=floor_height)
    return

def process_shapenet_scene_navigation_map():
    from tools.project_config_tools import shapenet_real_dir, shapenet_obj_list
    for obj_category in tqdm(shapenet_obj_list.keys()):
        for obj_id in shapenet_obj_list[obj_category]:
            scene_mesh_path = Path(os.path.join(shapenet_real_dir, obj_category, obj_id, "scene_mesh.ply"))
            if not scene_mesh_path.exists():
                continue
            floor_height = 0.
            scene_navmesh_dir = Path(os.path.join(shapenet_real_dir, obj_category, obj_id))
            for level, radius in navmesh_dict.items():
                navmesh_path = Path(os.path.join(scene_navmesh_dir, f"navmesh_{level}.ply"))
                if navmesh_path.exists():
                    continue
                navmesh = get_navmesh(navmesh_path, scene_mesh_path, agent_radius=radius, floor_height=floor_height)
    return
            


def process_scene_navigation_map(process_datasets, nav_args):
    for dataset in process_datasets:
        function_name = f'process_{dataset}_scene_navigation_map'
        processer = globals()[function_name]
        processer(*nav_args[dataset])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_datasets", default=["prox", "replica"], nargs='+')
    args = parser.parse_args()
    process_datasets = args.process_datasets
    nav_args = {
        "replica":[],
        "prox":[],
        "shapenet":[]
    }
    process_scene_navigation_map(process_datasets, nav_args)
