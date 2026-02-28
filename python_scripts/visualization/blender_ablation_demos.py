import bpy
import os
import glob
import json
from math import pi

subfigure_name = "ablation1-2"

render_quality = "high"
if subfigure_name in ["ablation1-1", "ablation1-2"]:
    demo_id = "MPH16+sit-bed_walk_sit-chair+0"
elif subfigure_name in ["ablation1-3", "ablation1-4"]:
    demo_id = "MPH8+walk_sit-bed_lie-bed+0"
dataset = "prox"
method = "direct_guide"
human_fmt = "obj"
project_dir = "/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion"
scene_name = demo_id.split("+")[0]
demos_render_config_file = project_dir + "/assets/demos_render_config.json"
if method == "ours":
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions/multi_round_motion_in_" + dataset + "/" +demo_id + "/sample00_rep00_" + human_fmt
elif method == "dimos":
    dir_name = project_dir + "/save/others_results/dimos_results/results_" + dataset + "/" +demo_id + "/meshes/body_meshes"
elif method == "omnicontrol":
    dir_name = project_dir + "/save/others_results/omnicontrol_results/results_" + dataset + "/" +demo_id + "/sample00_rep00_" + human_fmt
elif method == "direct_guide":
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions_direct_guide/multi_round_motion_in_" + dataset + "/" +demo_id + "/sample00_rep00_" + human_fmt

if dataset == "prox":
    if render_quality == "low":
        scene_path = project_dir + "/dataset/dimos_data/proxs/scenes_downsampled/" + scene_name + ".ply"
    else:
        scene_path = project_dir + "/dataset/dimos_data/proxs/scenes/" + scene_name + ".ply"
elif dataset == "replica":
    if render_quality == "low":
        scene_path = project_dir + "/dataset/dimos_data/replica/" + scene_name + "/mesh_downsampled.ply"
    else:
        scene_path = project_dir + "/dataset/dimos_data/replica/" + scene_name + "/mesh_woceil.ply"

if method == "direct_guide":
    fig_to_config = {
        "ablation1-1": {
            "light_xyz": [2, 3, 6],
            "cam_xyz": [3.4, -2, 2.5], 
            "cam_xyz_rot": [55, 0, 75],
            "idx1": 160,
            "idx2": 180
        },
        "ablation1-2": {
            "light_xyz": [2, 3, 6],
            "cam_xyz": [1.7, 0, 2.5], 
            "cam_xyz_rot": [40, 0, 120],
            "idx1": 200,
            "idx2": 220
        },
        "ablation1-3": {
            "light_xyz": [-3, 0, 6],
            "cam_xyz": [-1.9, -1.9, 3], 
            "cam_xyz_rot": [45, 0, 0],
            "idx1": 220,
            "idx2": 240
        },
        "ablation1-4": {
            "light_xyz": [-3, 0, 6],
            "cam_xyz": [-3.5, -1.3, 3], 
            "cam_xyz_rot": [45, 0, -45],
            "idx1": 290,
            "idx2": 310
        }
    }
elif method == "ours":
    fig_to_config = {
        "ablation1-1": {
            "light_xyz": [2, 3, 6],
            "cam_xyz": [3.4, -2, 2.5], 
            "cam_xyz_rot": [55, 0, 75],
            "idx1": 130,
            "idx2": 150
        },
        "ablation1-2": {
            "light_xyz": [2, 3, 6],
            "cam_xyz": [1.7, 0, 2.5], 
            "cam_xyz_rot": [40, 0, 120],
            "idx1": 170,
            "idx2": 190
        },
        "ablation1-3": {
            "light_xyz": [-3, 0, 6],
            "cam_xyz": [-1.9, -1.9, 3], 
            "cam_xyz_rot": [45, 0, 0],
            "idx1": 190,
            "idx2": 210
        },
        "ablation1-4": {
            "light_xyz": [-3, 0, 6],
            "cam_xyz": [-3.5, -1.3, 3], 
            "cam_xyz_rot": [45, 0, -45],
            "idx1": 290,
            "idx2": 310
        }
    }
def set_light_and_camera():
    #set light and camera
    with open(demos_render_config_file) as f:
        demos_render_config = json.load(f)
    light_position = fig_to_config[subfigure_name]["light_xyz"] 
    cam_position = fig_to_config[subfigure_name]["cam_xyz"]
    cam_rotation = fig_to_config[subfigure_name]["cam_xyz_rot"]
    bpy.data.objects['Light'].location[0] = light_position[0]
    bpy.data.objects['Light'].location[1] = light_position[1]
    bpy.data.objects['Light'].location[2] = light_position[2]
    bpy.data.objects['Light'].data.energy = 5000
    bpy.data.objects['Camera'].location[0] = cam_position[0]
    bpy.data.objects['Camera'].location[1] = cam_position[1]
    bpy.data.objects['Camera'].location[2] = cam_position[2]
    bpy.data.objects['Camera'].rotation_euler[0] = cam_rotation[0]*pi/180
    bpy.data.objects['Camera'].rotation_euler[1] = cam_rotation[1]*pi/180
    bpy.data.objects['Camera'].rotation_euler[2] = cam_rotation[2]*pi/180

def load_smoothness_body_meshes():
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    n_frames = len(all_files)
    n_frame_stride = 1
    frame_visible = [False] * n_frames
    idx1 = fig_to_config[subfigure_name]["idx1"]
    idx2 = fig_to_config[subfigure_name]["idx2"]
    for i in range(idx1, idx2, n_frame_stride):
        frame_visible[i] = True
    human_body_colors = [] 
    for i in range(n_frames):
        ratio = (idx2-i) / (idx2-idx1)
        color = (0.8 + 0.2 * ratio, 0.5 + 0.5 * ratio, ratio, 1.)
        human_body_colors.append(color)
    imported_models = []
    for i, file_name in enumerate(all_files):
        if not frame_visible[i]:
            continue
        if human_fmt == "obj":
            bpy.ops.wm.obj_import(filepath=dir_name+"/"+file_name, forward_axis='Y', up_axis='Z')
        elif human_fmt == "ply":
            bpy.ops.wm.ply_import(filepath=dir_name+"/"+file_name, directory=dir_name, files=[{"name":file_name, "name":file_name}])
        imported_model = bpy.context.selected_objects[0]
        mat = bpy.data.materials.new(f"{i:0>6}")
        mat.diffuse_color = human_body_colors[i]
        imported_model.active_material = mat
        for f in imported_model.data.polygons:
            f.use_smooth=True
        imported_models.append(imported_model)
    return

def load_scene():
    bpy.ops.wm.ply_import(filepath=scene_path, directory=os.path.dirname(scene_path), files=[{"name":os.path.basename(scene_path), "name":os.path.basename(scene_path)}])
    obj = bpy.context.selected_objects[0]
    bpy.ops.object.mode_set(mode='OBJECT')
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="Vertex_Color_Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    attribute = nodes.new(type='ShaderNodeAttribute')
    attribute.attribute_name = 'Col'
    mat.node_tree.links.new(bsdf.inputs['Base Color'], attribute.outputs['Color'])
    mat.node_tree.links.new(material_output.inputs['Surface'], bsdf.outputs['BSDF'])
    default_scene_collection = bpy.context.scene.collection
    default_scene_collection.objects.link(obj)
    bpy.context.view_layer.update()
    return obj

def set_render_config():
    render_path = os.path.join(project_dir, "save", "visualization_results", "visualization_figures_ablation", subfigure_name)
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    bpy.context.scene.render.filepath = os.path.join(render_path, method)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0 
    bpy.context.scene.render.fps = 1
    bpy.context.scene.render.film_transparent = True
    return

def render_image():
    bpy.ops.render.render(animation=True)
    return

def delete():
    collection = bpy.data.collections.get("Collection")
    for obj in collection.objects:
        if obj.name in ["Camera", "Light"]:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    return

def smoothness_ablation():
    set_light_and_camera()
    #load human mesh
    load_smoothness_body_meshes()
    #load scene mesh
    load_scene()
    #set render path
    set_render_config()
    #render image
    #render_image()
    #delete object except for camera and light
    #delete()

if __name__ == "__main__":
    smoothness_ablation()

