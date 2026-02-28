import bpy
import os
import glob
import json
from math import pi

render_quality = "high"
demo_id = "MPH16+sit-bed_walk_sit-chair+0"
dataset = "prox"
human_fmt = "obj"
project_dir = "/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion"
scene_name = demo_id.split("+")[0]

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

demos_render_config_file = project_dir + "/assets/demos_render_config.json"
with open(demos_render_config_file) as f:
    demos_render_config = json.load(f)
demos_render_config["MPH16+sit-bed_walk_sit-chair+0"]["light"] = [5, 5, 5] 
demos_render_config["MPH16+sit-bed_walk_sit-chair+0"]["cam_xyz"] = [3.5, 3.3, 1.3] 
demos_render_config["MPH16+sit-bed_walk_sit-chair+0"]["cam_xyz_rot"] = [75, 0, 130] 

color_list = [
    [0xff, 0x00, 0x00], #dark red
    [0xff, 0x20, 0x00], #red
    [0xff, 0x40, 0x00], #orange
    [0xff, 0xb0, 0x00], #yellow
    [0x40, 0xff, 0x00], #light green
    [0x00, 0xff, 0x00], #green
    [0x00, 0x40, 0xff], #blue
    [0x00, 0x00, 0xff], #dark blue
    [0x20, 0x00, 0xff], #purple
    [0x40, 0x00, 0xff], #dark purple
]

def demo_id_to_color(demo_id):
    color_idx = sum([ord(d) for d in demo_id]) % len(color_list)
    color = color_list[3]
    color = [color[0]/255., color[1]/255., color[2]/255.]
    return color

def fetch_motion_seq_colors(n_frames):
    start_color = (1., 1., 1., 1.)
    color_3d = demo_id_to_color(demo_id)
    target_color = (color_3d[0], color_3d[1], color_3d[2], 1.) 
    colors = []
    for i in range(n_frames):
        ratio = i / (n_frames - 1)
        color = (start_color[0] * (1 - ratio) + target_color[0] * ratio,
                 start_color[1] * (1 - ratio) + target_color[1] * ratio,
                 start_color[2] * (1 - ratio) + target_color[2] * ratio,
                 start_color[3] * (1 - ratio) + target_color[3] * ratio)
        colors.append(color)
    return colors

def set_light_and_camera():
    #set light and camera
    light_position = demos_render_config[demo_id]["light"]
    cam_position = demos_render_config[demo_id]["cam_xyz"]
    cam_rotation = demos_render_config[demo_id]["cam_xyz_rot"]
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

def load_init_motion_meshes():
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions/multi_round_motion_in_" + dataset + "/" + "multi_round_motion_in_MPH16_noised" + "/sample00_rep00_" + human_fmt
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    stride = 60
    all_files = [all_files[i] for i in range(len(all_files)-1-stride*2, len(all_files), stride)]
    n_frames = len(all_files)
    frame_visible = [True] * n_frames
    human_body_colors = fetch_motion_seq_colors(n_frames)
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

def load_intermediate_motion_meshes():
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions/multi_round_motion_in_" + dataset + "/" + "multi_round_motion_in_MPH16_intermediate" + "/sample00_rep00_" + human_fmt
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    stride = 60
    all_files = [all_files[i] for i in range(len(all_files)-1-stride*2, len(all_files), stride)]
    n_frames = len(all_files)
    frame_visible = [True] * n_frames
    human_body_colors = fetch_motion_seq_colors(n_frames)
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

def load_final_motion_meshes():
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions/multi_round_motion_in_" + dataset + "/" +demo_id + "/sample00_rep00_" + human_fmt
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    stride = 60
    all_files = [all_files[i] for i in range(len(all_files)-1-stride*2, len(all_files), stride)]
    n_frames = len(all_files)
    frame_visible = [True] * n_frames
    human_body_colors = fetch_motion_seq_colors(n_frames)
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

def set_render_config(subfigure_name):
    render_path = os.path.join(project_dir, "save", "visualization_results", "visualization_figures_intro")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    bpy.context.scene.render.filepath = os.path.join(render_path, subfigure_name)
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

def only_scene():
    set_light_and_camera()
    #load scene mesh
    load_scene()
    #set render path
    set_render_config("scene")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def init_motion():
    set_light_and_camera()
    #load human mesh
    load_init_motion_meshes()
    #load scene mesh
    load_scene()
    #set render path
    set_render_config("init")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def intermediate_motion():
    set_light_and_camera()
    #load human mesh
    load_intermediate_motion_meshes()
    #load scene mesh
    load_scene()
    #set render path
    set_render_config("intermediate")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def final_motion():
    set_light_and_camera()
    #load human mesh
    load_final_motion_meshes()
    #load scene mesh
    load_scene()
    #set render path
    set_render_config("final")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

if __name__ == "__main__":
    render_mode = "show"
    intermediate_motion()
