import argparse
import json
from mesh_to_sdf import get_surface_point_cloud
import trimesh
import numpy as np
import os
import pickle
from pathlib import Path
from tqdm import tqdm
os.environ['PYOPENGL_PLATFORM'] = 'egl'


shapenet_to_zup = np.array(
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

zup_to_shapenet = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]]
)

def load_obj_mesh(obj_mesh_path:str, switch_to_zup:bool = False, floor_height = 0):
    obj_mesh = trimesh.load(obj_mesh_path, force='mesh')
    if switch_to_zup:
        obj_mesh.apply_transform(shapenet_to_zup)
    obj_mesh.vertices[:, 2] -= floor_height
    return obj_mesh

def list_shapenet(shapenet_dir:str):
    obj_categorys = {key:[] for key in os.listdir(shapenet_dir)}
    for obj_category in obj_categorys:
        category_path = os.path.join(shapenet_dir, obj_category)
        obj_categorys[obj_category] = os.listdir(category_path)
    return obj_categorys


def calc_instance_sdf(obj_mesh, voxel_resolution:int = 128, add_floor:bool = False, sample_method="scan"):
    scene_centroid = obj_mesh.bounding_box.centroid
    extents = obj_mesh.bounding_box.extents
    if add_floor:
        floor_mesh = trimesh.creation.box(extents=np.array([extents[0] + 2, extents[1] + 2, 0.5]),
                                              transform=np.array([[1.0, 0.0, 0.0, scene_centroid[0]],
                                                                  [0.0, 1.0, 0.0, scene_centroid[1]],
                                                                  [0.0, 0.0, 1.0, -0.25],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
        scene_mesh = obj_mesh + floor_mesh
        scene_extents = extents + np.array([2, 2, 1])
    else:
        scene_mesh = obj_mesh
        scene_extents = extents
    scene_scale = np.max(scene_extents) * 0.5
    scene_mesh.vertices -= scene_centroid
    scene_mesh.vertices /= scene_scale
    sign_method = 'normal'
    surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method=sample_method,
                                                  bounding_radius=3 ** 0.5,
                                                  scan_count=100,
                                                  scan_resolution=400, sample_point_count=10000000,
                                                  calculate_normals=(sign_method == 'normal'))

    sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth',
                                                             sample_count=11, pad=False,
                                                             check_result=False, return_gradients=True)
    print(sdf_grid.shape, gradient_grid.shape)
    object_sdf = {
        'grid': sdf_grid * scene_scale,
        'gradient_grid': gradient_grid,
        'dim': voxel_resolution,
        'centroid': scene_centroid,
        'scale': scene_scale,
    }
    scene_mesh.vertices *= scene_scale
    scene_mesh.vertices += scene_centroid
    return object_sdf, scene_mesh

def get_shapenet_sdf(sdf_dim:int = 128, 
                     add_floor:bool = True, 
                     sample_method:str = "scan"
):
    from tools.project_config_tools import shapenet_real_dir
    obj_categorys = list_shapenet(shapenet_real_dir)
    obj_categorys = {
    'Armchairs': ['9faefdf6814aaa975510d59f3ab1ed64',
        'cacb9133bc0ef01f7628281ecb18112',
        'ea918174759a2079e83221ad0d21775',],
    'L-Sofas': ['5cea034b028af000c2843529921f9ad7',],
    'Sofas': ['1dd6e32097b09cd6da5dde4c9576b854',
        '71fd7103997614db490ad276cd2af3a4',
        '277231dcb7261ae4a9fe1734a6086750',],
    'StraightChairs':['2ed17abd0ff67d4f71a782a4379556c7',
        '68dc37f167347d73ea46bea76c64cc3d',
        'd93760fda8d73aaece101336817a135f']
    }
    for obj_category in obj_categorys:
        for obj_name in tqdm(obj_categorys[obj_category]):
            print(f"Processing {obj_category}/{obj_name}...")
            model_path = Path(os.path.join(shapenet_real_dir, obj_category, obj_name))
            obj_mesh = load_obj_mesh(model_path / "model.obj", switch_to_zup=True, floor_height=0)
            obj_sdf_dict, scene_mesh = calc_instance_sdf(obj_mesh,
                                                         voxel_resolution=sdf_dim, 
                                                         add_floor=add_floor, 
                                                         sample_method=sample_method)
            obj_mesh.export(model_path / "model_zup.obj")
            obj_mesh.export(model_path / "model_zup.ply")
            scene_mesh.export(model_path / "scene_mesh.obj")
            scene_mesh.export(model_path / "scene_mesh.ply")
            with open(model_path / "sdf_gradient.pkl", "wb") as f:
                pickle.dump(obj_sdf_dict, f)
    return


def get_replica_sdf(sdf_dim:int = 256, 
                    add_floor:bool = False, 
                    sample_method:str = "sample"
):
    from tools.project_config_tools import replica_dir, replica_room_list
    replica_dir = Path(replica_dir)
    for scene in tqdm(replica_room_list):
        print(scene)
        scene_path = replica_dir / scene
        obj_mesh = load_obj_mesh(scene_path / "mesh.ply", switch_to_zup=False, floor_height=0)
        obj_sdf_dict, _ = calc_instance_sdf(obj_mesh,
                                            voxel_resolution=sdf_dim, 
                                            add_floor=add_floor, 
                                            sample_method=sample_method)
        np.save(scene_path / f"{scene}_sdf.npy", obj_sdf_dict['grid'])
        with open(scene_path / f"{scene}_sdf.json", "w") as f:
            obj_sdf_dict.pop("grid")
            obj_sdf_dict.pop("gradient_grid")
            for key, value in obj_sdf_dict.items():
                if isinstance(value, np.ndarray):
                    obj_sdf_dict[key] = list(value)
            json.dump(obj_sdf_dict, f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_datasets", default=["shapenet", "replica"], nargs='+')
    args = parser.parse_args()
    sdf_args = {
        "replica" : [256, False, "sample"],
        "shapenet" : [128, True, "scan"]
    }
    for dataset in args.process_datasets:
        processor_name = f"get_{dataset}_sdf"
        processor = globals()[processor_name]
        processor(*sdf_args[dataset])
