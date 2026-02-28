import bpy
import os
import glob
import json
from math import pi

render_quality = "high"
dataset = "prox"
human_fmt = "obj"
project_dir = "/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion"

demos_render_config_file = project_dir + "/assets/demos_render_config.json"
with open(demos_render_config_file) as f:
    demos_render_config = json.load(f)
render_config = {}
render_config["light"] = [4, -2, 5] 
render_config["cam_xyz"] = [4, -2, 2.5] 
render_config["cam_xyz_rot"] = [60, 0, 45] 

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

def demo_id_to_color():
    color = color_list[3]
    color = [color[0]/255., color[1]/255., color[2]/255.]
    return color

def fetch_motion_seq_colors(n_frames):
    start_color = (1., 1., 1., 1.)
    color_3d = demo_id_to_color()
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
    light_position = render_config["light"]
    cam_position = render_config["cam_xyz"]
    cam_rotation = render_config["cam_xyz_rot"]
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

def load_start_meshes():
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions/single_round_motion_controlnet/normal"
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    n_frames = len(all_files)
    frame_visible = [False] * n_frames
    frame_visible[0] = True
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
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions/single_round_motion_controlnet/intermediate/sample00_rep00_obj"
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    n_frames = len(all_files)
    frame_visible = [False] * n_frames
    for i in range(0, n_frames, 32):
        frame_visible[i] = True
    frame_visible[-1] = True
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
    dir_name = project_dir + "/save/trained_models/mixed_action2motion_control/generated_motions/single_round_motion_controlnet/normal"
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    n_frames = len(all_files)
    frame_visible = [False] * n_frames
    for i in range(0, n_frames, 32):
        frame_visible[i] = True
    frame_visible[-1] = True
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

def add_joint_sphere():
    material = bpy.data.materials.new(name="SphereMaterial")
    material.diffuse_color = (1, 0, 0, 1)
    points = [
        (0, 0, 0),
        (2, 2, 0)
    ]
    sphere_radius = 0.05
    for i, point in enumerate(points):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, location=point)
        sphere = bpy.context.object
        sphere.name = f"Sphere_{i:0>4}"
        if sphere.data.materials:
            sphere.data.materials[0] = material
        else:
            sphere.data.materials.append(material)
    return

def add_floor():
    material = bpy.data.materials.new(name="CubeMaterial")
    material.diffuse_color = (1, 1, 1, 1)
    points = [
        (1, 1, -0.9),
    ]
    for i, point in enumerate(points):
        bpy.ops.mesh.primitive_cube_add(scale=(2,2,0.01), location=point)
        cube = bpy.context.object
        cube.name = f"Floor_{i:0>4}"
        if cube.data.materials:
            cube.data.materials[0] = material
        else:
            cube.data.materials.append(material)
    return

def set_render_config(subfigure_name):
    render_path = os.path.join(project_dir, "save", "visualization_results", "visualization_figures_controlnet")
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

def start_pose():
    set_light_and_camera()
    #load human mesh
    load_start_meshes()
    #load floor
    add_floor()
    #set render path
    set_render_config("start")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def goal_joint():
    set_light_and_camera()
    #add joint sphere
    add_joint_sphere()
    #set render path
    set_render_config("joint")
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
    #load floor
    add_floor()
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
    #load floor
    add_floor()
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
    final_motion()
