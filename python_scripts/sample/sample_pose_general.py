import os
import pickle
import sys
sys.path.append(os.getcwd())
sys.path.append('./deps/coins')
sys.path.append('./deps/coins/interaction')

if "DISPLAY" not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


import numpy as np
import pytorch3d.transforms.rotation_conversions
import torch
import trimesh.util
import pylab
from tqdm import tqdm
import mesh_to_sdf

from transform_trainer import *
from interaction_trainer import LitInteraction
from pointnet2 import farthest_point_sample
from viz_util import *
from sample_pelvis import to_pointcloud, composition_sample
from sample_interaction import get_composition_mask
from deps.dimos.synthesize.get_scene import ReplicaScene, ShapenetScene, ObjectScene, GeneralScene, shapenet_to_zup
from tools.project_config_tools import replica_dir, shapenet_real_dir
from tools.coins_tools import rotation_matrix_vector_multiply, calc_interaction_loss, posa_optimize, params2numpy
from tools.scene_tools import ReplicaSceneV2

def two_stage_sample(scene_name, interaction_candidates, method='direct'):
    """
    Sample interactions in a given scene with the semantics of specified action-object pairs.
    When 'used_scene_names' and 'used_interactions' are set to 'all', this function iterates over all specified scenes and all action-object combinations and synthesize interactions for large scale evaluation.
    You can choose to synthesize a specific interaction in a specific scene by configuring 'used_scene_names' and 'used_interactions'.
    """
    scene = ReplicaSceneV2(scene_name, Path(replica_dir))
    for action in interaction_candidates:
        verb_ids = [action_names.index(action)]
        if len(verb_ids) < maximum_atomics:
            verb_ids = verb_ids + [-1] * (maximum_atomics - len(verb_ids))
        verb_ids = torch.tensor(verb_ids, device=device).unsqueeze(0)  # Bx2
        for obj_candidate in interaction_candidates[action]:
            category_idxs = scene.get_category_indices(obj_candidate['obj_category'])
            if len(category_idxs) == 0:
                print("no such category: ", obj_candidate['obj_category'])
                continue
            obj_id = category_idxs[min(int(obj_candidate['obj_id']), len(category_idxs) - 1)]
            object_mesh = scene.get_mesh_with_accessory(obj_id)
            obj_meshes = [object_mesh]
            object_name = scene_name + '_' + str(obj_id)
            combination_name = action + '_' + object_name

            object_points_scene = torch.from_numpy(np.stack(
                [np.concatenate(pointcloud, axis=1) for pointcloud in
                 to_pointcloud(obj_meshes, num_points=interaction_model.args.num_obj_points, sample_surface=False)],
                axis=0)).to(device).unsqueeze(0).expand(1, 2, -1, -1).clone()  # B x 2 * P * 9
            object_points_floor = object_points_scene.clone()
            offset = object_points_floor[:, :, :, :3].reshape(-1, 3).mean(axis=0)
            offset[2] = scene.get_floor_height()
            object_points_floor[:, :, :, :3] -= offset  # recenter around the objects, without this breaks on scannet scenes

            pelvis_batch_input = {
                'num_atomics': torch.ones(1, device=device),
                'object_pointclouds': object_points_floor,
                'verb_ids': verb_ids,
            }

            # sample pelvis and body
            samples = []
            for sample_idx in tqdm(range(args.num_sample)):
                if args.decode:
                    z_pelvis = z_pelvis_global[sample_idx]
                    z_body = z_body_global[sample_idx]
                else:
                    z_pelvis = np.random.randn(args.num_try, transform_model.args.latent_dim).astype(np.float32)
                    z_body = np.random.randn(args.num_try, interaction_model.args.latent_dim).astype(np.float32)

                best_loss = None
                best_sample = None
                for try_idx in range(args.num_try):
                    with torch.no_grad():
                        # sample pelvis frame
                        x, _ = transform_model.model.decode(pelvis_batch_input,
                                                            z_sample=torch.from_numpy(
                                                                z_pelvis[[try_idx], :]).to(device))
                        # x, _ = transform_model.model.sample(batch)
                        x = x.squeeze(1).detach()
                        x[:, 6:] += offset  # add floor height, back to PROX scene coords

                    if method in ['direct', 'optimization_after_get_body']:
                        with torch.no_grad():
                            # sample body
                            rotation = rot6d_to_mat(x[:, :6])
                            pelvis = x[:, 6:]
                            object_points_local = object_points_scene.clone()  # 1x2xPx9
                            object_points_local[:, :, :, :3] = object_points_local[:, :, :, :3] - pelvis.reshape(-1, 1, 1,
                                                                                                           3)
                            object_points_local[:, :, :, :3] = torch.matmul(object_points_local[: , :, :, :3],
                                                                      rotation.reshape(-1, 1, 3, 3))  # coord
                            object_points_local[:, :, :, 6:] = torch.matmul(object_points_local[: , :, :, 6:],
                                                                      rotation.reshape(-1, 1, 3, 3))  # normal
                            body_batch_input = {
                                'num_atomics': torch.ones(1, device=device),
                                'object_pointclouds': object_points_local,
                                'verb_ids': verb_ids,
                            }
                            bodies, _ = interaction_model.model.decode(body_batch_input, z_sample=torch.from_numpy(z_body[[try_idx], :]).to(device))
                            # bodies, _ = interaction_model.model.sample(batch)
                            bodies, contact = bodies[:, :, :3], bodies[:, :, 3:]
                            smplx_param, smplx_vertices = interaction_model.regress_smplx(bodies)
                            bodies = interaction_model.mesh.downsample(smplx_vertices)

                        if method == 'direct':
                            # transform back to scene coord frame
                            bodies = torch.matmul(bodies, rotation.transpose(1, 2))
                            bodies = bodies + pelvis.reshape(-1, 1, 3)
                            smplx_param['global_orient'] = rotation_matrix_vector_multiply(rotation, smplx_param['global_orient'])
                            smplx_param['transl'] = smplx_param['transl'] + pelvis
                            loss = calc_interaction_loss(bodies, contact, object_points_scene[:, :, :, :3],
                                                         scene, args)
                            loss = loss.item()
                        else:
                            loss, smplx_param, bodies = posa_optimize(smplx_param, contact, x, object_points_scene[:, :, :, :3], scene, body_model, global_body_mesh, args)

                    contact_semantic, contact_scene = contact[:, :, 0], contact[:, :, 1]
                    if np.isnan(loss):
                        loss = 2333  # result can be nan
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                        best_sample = {'body': bodies, 'contact': contact, 'smplx_param': smplx_param,
                                       'init_pelvis_frame': x}
                samples.append(best_sample)
                # visualize
                if args.visualize:
                    frame = create_frame(best_sample['init_pelvis_frame'][0])
                    if body_type == 'mesh':
                        bodies, contact = best_sample['body'], best_sample['contact'][:, :, args.contact_dimension]
                        colors = np.array([[0.8, 0.8, 0.8]] * bodies.shape[1])
                        if interaction_model.args.use_contact_feature:
                            colors[contact[0].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                        body_mesh = trimesh.Trimesh(
                            vertices=bodies[0].detach().cpu().numpy(),
                            faces=interaction_model.mesh.faces,
                            vertex_colors=colors,
                        )
                    else:
                        body_mesh = skeleton_to_mesh(bodies[0].detach().cpu().numpy(), color=np.array(color_map(sample_idx / args.sample_num)))
                    body_meshes = [frame, body_mesh]

                    # render synthesis results
                    shared_name = method + '_' + str(sample_idx)
                    base_name = shared_name  + '.png'
                    export_file = Path(args.save_dir, base_name)
                    export_file.parent.mkdir(exist_ok=True, parents=True)
                    img_collage = render_interaction_multview(body=trimesh.util.concatenate(body_meshes),
                                                              smooth_body=False,
                                                              obj_points_coord=object_points_scene[0, 0, :, :3].detach().cpu().numpy(),
                                                              static_scene=scene.mesh if args.full_scene else trimesh.util.concatenate(obj_meshes))
                    img_collage.save(str(export_file))
                    # base_name = combination_name + '_' + str(
                    #     sample_idx) + '_smooth.png'
                    # export_file = Path(args.save_dir, args.exp_name, method, action, scene_name, base_name)
                    # img_collage = render_interaction_multview(body=body_mesh,
                    #                                           obj_points_coord=object_points_scene[0, 0, :,
                    #                                                            :3].detach().cpu().numpy(),
                    #                                           static_scene=scene.mesh if args.full_scene else trimesh.util.concatenate(
                    #                                               obj_meshes))
                    # img_collage.save(str(export_file))
                    base_name = shared_name + '_body.png'
                    export_file = Path(args.save_dir, base_name)
                    img_collage = render_body_multview(body=body_mesh,)
                    img_collage.save(str(export_file))
                    # export ply mesh
                    # smplx_vertices = interaction_model.body_model(**best_sample['smplx_param']).vertices.detach().cpu().numpy()[0]
                    # body_mesh = trimesh.Trimesh(
                    #     vertices=smplx_vertices,
                    #     faces=interaction_model.mesh.meshes[0].faces,
                    #     vertex_colors=colors,
                    # )
                    # base_name = combination_name + '_' + str(
                    #     sample_idx) + '.ply'
                    # mesh_path = Path(args.save_dir, args.exp_name, method, action, scene_name, base_name)
                    # body_mesh.export(mesh_path)
                    # export smplx param
                    base_name = shared_name + '.pkl'
                    export_file = Path(args.save_dir, base_name)
                    with open(export_file, 'wb') as f:
                        pickle.dump({'smplx_param': params2numpy(best_sample['smplx_param']), 'action':action, 'combination':combination_name}, f)

            # write smplx results
            synthesis_results = [sample['smplx_param'] for sample in samples]
            result_path = Path(args.save_dir, method + '.pkl')
            result_path.parent.mkdir(exist_ok=True, parents=True)
            print(result_path)
            with open(result_path, 'wb') as result_file:
                pickle.dump(synthesis_results, result_file)


def shapenet_sample(interaction_candidates, method='direct'):
    """
    Sample interactions in a given scene with the semantics of specified action-object pairs.
    When 'used_scene_names' and 'used_interactions' are set to 'all', this function iterates over all specified scenes and all action-object combinations and synthesize interactions for large scale evaluation.
    You can choose to synthesize a specific interaction in a specific scene by configuring 'used_scene_names' and 'used_interactions'.
    """
    for action in interaction_candidates:
        verb_ids = [action_names.index(action)]
        if len(verb_ids) < maximum_atomics:
            verb_ids = verb_ids + [-1] * (maximum_atomics - len(verb_ids))
        verb_ids = torch.tensor(verb_ids, device=device).unsqueeze(0)  # Bx2
        for obj_candidate in interaction_candidates[action]:
            obj_path = Path(obj_candidate['obj_path'])
            scene = ShapenetScene(obj_path)
            scene.mesh.visual.vertex_colors = np.tile(np.array([200, 200, 200, 255]), (len(scene.mesh.vertices), 1))
            obj_category = obj_path.parent.parent.name
            obj_id = obj_path.parent.name
            object_mesh = scene.mesh
            obj_meshes = [object_mesh]
            object_name = obj_category + '_' + str(obj_id)
            combination_name = action + '_' + object_name

            object_points_scene = torch.from_numpy(np.stack(
                [np.concatenate(pointcloud, axis=1) for pointcloud in
                 to_pointcloud(obj_meshes, num_points=interaction_model.args.num_obj_points, sample_surface=True)],
                axis=0)).to(device).unsqueeze(0).expand(1, 2, -1, -1).clone()  # B x 2 * P * 9
            object_points_floor = object_points_scene.clone()
            offset = object_points_floor[:, :, :, :3].reshape(-1, 3).mean(axis=0)
            offset[2] = 0
            object_points_floor[:, :, :, :3] -= offset  # recenter around the objects, without this breaks on scannet scenes

            pelvis_batch_input = {
                'num_atomics': torch.ones(1, device=device),
                'object_pointclouds': object_points_floor,
                'verb_ids': verb_ids,
            }

            # sample pelvis and body
            samples = []
            for sample_idx in range(args.num_sample):
                if args.decode:
                    z_pelvis = z_pelvis_global[sample_idx]
                    z_body = z_body_global[sample_idx]
                else:
                    z_pelvis = np.random.randn(args.num_try, transform_model.args.latent_dim).astype(np.float32)
                    z_body = np.random.randn(args.num_try, interaction_model.args.latent_dim).astype(np.float32)

                best_loss = None
                best_sample = None
                for try_idx in range(args.num_try):
                    with torch.no_grad():
                        # sample pelvis frame
                        x, _ = transform_model.model.decode(pelvis_batch_input,
                                                            z_sample=torch.from_numpy(
                                                                z_pelvis[[try_idx], :]).to(device))
                        # x, _ = transform_model.model.sample(batch)
                        x = x.squeeze(1).detach()
                        x[:, 6:] += offset  # add floor height, back to PROX scene coords

                    if method in ['direct', 'optimization_after_get_body']:
                        with torch.no_grad():
                            # sample body
                            rotation = rot6d_to_mat(x[:, :6])
                            pelvis = x[:, 6:]
                            object_points_local = object_points_scene.clone()  # 1x2xPx9
                            object_points_local[:, :, :, :3] = object_points_local[:, :, :, :3] - pelvis.reshape(-1, 1, 1,
                                                                                                           3)
                            object_points_local[:, :, :, :3] = torch.matmul(object_points_local[: , :, :, :3],
                                                                      rotation.reshape(-1, 1, 3, 3))  # coord
                            object_points_local[:, :, :, 6:] = torch.matmul(object_points_local[: , :, :, 6:],
                                                                      rotation.reshape(-1, 1, 3, 3))  # normal
                            body_batch_input = {
                                'num_atomics': torch.ones(1, device=device),
                                'object_pointclouds': object_points_local,
                                'verb_ids': verb_ids,
                            }
                            bodies, _ = interaction_model.model.decode(body_batch_input, z_sample=torch.from_numpy(z_body[[try_idx], :]).to(device))
                            # bodies, _ = interaction_model.model.sample(batch)
                            bodies, contact = bodies[:, :, :3], bodies[:, :, 3:]
                            smplx_param, smplx_vertices = interaction_model.regress_smplx(bodies)
                            bodies = interaction_model.mesh.downsample(smplx_vertices)

                        if method == 'direct':
                            # transform back to scene coord frame
                            bodies = torch.matmul(bodies, rotation.transpose(1, 2))
                            bodies = bodies + pelvis.reshape(-1, 1, 3)
                            smplx_param['global_orient'] = rotation_matrix_vector_multiply(rotation, smplx_param['global_orient'])
                            smplx_param['transl'] = smplx_param['transl'] + pelvis
                            loss = calc_interaction_loss(bodies, contact, object_points_scene[:, :, :, :3],
                                                         scene, args)
                            loss = loss.item()
                        else:
                            loss, smplx_param, bodies = posa_optimize(smplx_param, contact, x, object_points_scene[:, :, :, :3], scene, body_model, global_body_mesh, args)

                    contact_semantic, contact_scene = contact[:, :, 0], contact[:, :, 1]
                    if np.isnan(loss):
                        loss = 2333  # result can be nan
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                        best_sample = {'body': bodies, 'contact': contact, 'smplx_param': smplx_param,
                                       'init_pelvis_frame': x}
                samples.append(best_sample)
                # visualize
                if args.visualize:
                    frame = create_frame(best_sample['init_pelvis_frame'][0])
                    if body_type == 'mesh':
                        bodies, contact = best_sample['body'], best_sample['contact'][:, :, args.contact_dimension]
                        colors = np.array([[0.8, 0.8, 0.8]] * bodies.shape[1])
                        if interaction_model.args.use_contact_feature:
                            colors[contact[0].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                        body_mesh = trimesh.Trimesh(
                            vertices=bodies[0].detach().cpu().numpy(),
                            faces=interaction_model.mesh.faces,
                            vertex_colors=colors,
                        )
                    else:
                        body_mesh = skeleton_to_mesh(bodies[0].detach().cpu().numpy(), color=np.array(color_map(sample_idx / args.sample_num)))
                    body_meshes = [frame, body_mesh]

                    # render synthesis results
                    shared_name = method + '_' + str(sample_idx)
                    base_name = shared_name + '.png'
                    export_file = Path(args.save_dir, base_name)
                    export_file.parent.mkdir(exist_ok=True, parents=True)
                    img_collage = render_interaction_multview(body=trimesh.util.concatenate(body_meshes),
                                                              smooth_body=False,
                                                              obj_points_coord=object_points_scene[0, 0, :, :3].detach().cpu().numpy(),
                                                              static_scene=scene.mesh if args.full_scene else trimesh.util.concatenate(obj_meshes))
                    img_collage.save(str(export_file))
                    # base_name = str(
                    #     sample_idx) + '_' + method + '_smooth.png'
                    # export_file = Path(args.save_dir, args.exp_name, method, action, combination_name, base_name)
                    # img_collage = render_interaction_multview(body=body_mesh,
                    #                                           obj_points_coord=object_points_scene[0, 0, :,
                    #                                                            :3].detach().cpu().numpy(),
                    #                                           static_scene=scene.mesh if args.full_scene else trimesh.util.concatenate(
                    #                                               obj_meshes))
                    # img_collage.save(str(export_file))
                    # base_name = combination_name + '_' + str(
                    #     sample_idx) + '_' + method + '_body.png'
                    # export_file = Path(args.save_dir, args.exp_name, method, action, base_name)
                    # img_collage = render_body_multview(body=body_mesh,)
                    # img_collage.save(str(export_file))
                    # # export ply mesh
                    # smplx_vertices = interaction_model.body_model(**best_sample['smplx_param']).vertices.detach().cpu().numpy()[0]
                    # body_mesh = trimesh.Trimesh(
                    #     vertices=smplx_vertices,
                    #     faces=interaction_model.mesh.meshes[0].faces,
                    #     vertex_colors=colors,
                    # )
                    # base_name = combination_name + '_' + str(
                    #     sample_idx) + '_' + method + '.ply'
                    # mesh_path = Path(args.save_dir, args.exp_name, method, action, base_name)
                    # body_mesh.export(mesh_path)

                    result_path = Path(args.save_dir, shared_name + '.pkl')
                    result = {
                        'smplx_param': params2numpy(best_sample['smplx_param']),
                        'obj_category': obj_category,
                        'obj_id': obj_id,
                        'action': action,
                    }
                    with open(result_path, 'wb') as result_file:
                        pickle.dump(result, result_file)

            # write smplx results
            synthesis_results = [sample['smplx_param'] for sample in samples]
            result_path = Path(args.save_dir, 'results.pkl')
            result_path.parent.mkdir(exist_ok=True, parents=True)
            print(result_path)
            with open(result_path, 'wb') as result_file:
                pickle.dump(synthesis_results, result_file)

def general_sample(interaction_candidates, method='direct'):
    """
    Sample interactions in a given scene with the semantics of specified action-object pairs.
    When 'used_scene_names' and 'used_interactions' are set to 'all', this function iterates over all specified scenes and all action-object combinations and synthesize interactions for large scale evaluation.
    You can choose to synthesize a specific interaction in a specific scene by configuring 'used_scene_names' and 'used_interactions'.
    """
    for action in interaction_candidates:
        verb_ids = [action_names.index(action)]
        if len(verb_ids) < maximum_atomics:
            verb_ids = verb_ids + [-1] * (maximum_atomics - len(verb_ids))
        verb_ids = torch.tensor(verb_ids, device=device).unsqueeze(0)  # Bx2
        for obj_candidate in interaction_candidates[action]:
            obj_category, obj_id = obj_candidate['obj_category'], obj_candidate['obj_id']
            scene = GeneralScene(obj_candidate['scene_path'], obj_candidate['scene_name'])
            object_mesh = trimesh.load(obj_candidate['obj_path'], force='mesh')
            if 'obj' in obj_candidate['obj_path'] or 'glb' in obj_candidate['obj_path']:
                object_mesh.apply_transform(shapenet_to_zup)
            object_mesh.visual.vertex_colors = np.tile(np.array([200, 200, 200, 255]), (len(object_mesh.vertices), 1))
            obj_meshes = [object_mesh]
            object_name = obj_category + '_' + str(obj_id)
            combination_name = action + '_' + object_name

            object_points_scene = torch.from_numpy(np.stack(
                [np.concatenate(pointcloud, axis=1) for pointcloud in
                 to_pointcloud(obj_meshes, num_points=interaction_model.args.num_obj_points, sample_surface=True)],
                axis=0)).to(device).unsqueeze(0).expand(1, 2, -1, -1).clone()  # B x 2 * P * 9
            object_points_floor = object_points_scene.clone()
            offset = object_points_floor[:, :, :, :3].reshape(-1, 3).mean(axis=0)
            offset[2] = 0
            object_points_floor[:, :, :, :3] -= offset  # recenter around the objects, without this breaks on scannet scenes

            pelvis_batch_input = {
                'num_atomics': torch.ones(1, device=device),
                'object_pointclouds': object_points_floor,
                'verb_ids': verb_ids,
            }
            # sample pelvis and body
            samples = []
            for sample_idx in range(args.num_sample):
                print('Sample_idx: ',sample_idx)
                if args.decode:
                    z_pelvis = z_pelvis_global[sample_idx]
                    z_body = z_body_global[sample_idx]
                else:
                    z_pelvis = np.random.randn(args.num_try, transform_model.args.latent_dim).astype(np.float32)
                    z_body = np.random.randn(args.num_try, interaction_model.args.latent_dim).astype(np.float32)

                best_loss = None
                best_sample = None
                for try_idx in range(args.num_try):
                    print('Try_idx: ',try_idx)
                    with torch.no_grad():
                        # sample pelvis frame
                        x, _ = transform_model.model.decode(pelvis_batch_input,
                                                            z_sample=torch.from_numpy(
                                                                z_pelvis[[try_idx], :]).to(device))
                        # x, _ = transform_model.model.sample(batch)
                        x = x.squeeze(1).detach()
                        x[:, 6:] += offset  # add floor height, back to PROX scene coords

                    if method in ['direct', 'optimization_after_get_body']:
                        with torch.no_grad():
                            # sample body
                            rotation = rot6d_to_mat(x[:, :6])
                            pelvis = x[:, 6:]
                            object_points_local = object_points_scene.clone()  # 1x2xPx9
                            object_points_local[:, :, :, :3] = object_points_local[:, :, :, :3] - pelvis.reshape(-1, 1, 1,
                                                                                                           3)
                            object_points_local[:, :, :, :3] = torch.matmul(object_points_local[: , :, :, :3],
                                                                      rotation.reshape(-1, 1, 3, 3))  # coord
                            object_points_local[:, :, :, 6:] = torch.matmul(object_points_local[: , :, :, 6:],
                                                                      rotation.reshape(-1, 1, 3, 3))  # normal
                            body_batch_input = {
                                'num_atomics': torch.ones(1, device=device),
                                'object_pointclouds': object_points_local,
                                'verb_ids': verb_ids,
                            }
                            bodies, _ = interaction_model.model.decode(body_batch_input, z_sample=torch.from_numpy(z_body[[try_idx], :]).to(device))
                            # bodies, _ = interaction_model.model.sample(batch)
                            bodies, contact = bodies[:, :, :3], bodies[:, :, 3:]
                            smplx_param, smplx_vertices = interaction_model.regress_smplx(bodies)
                            bodies = interaction_model.mesh.downsample(smplx_vertices)

                        if method == 'direct':
                            # transform back to scene coord frame
                            bodies = torch.matmul(bodies, rotation.transpose(1, 2))
                            bodies = bodies + pelvis.reshape(-1, 1, 3)
                            smplx_param['global_orient'] = rotation_matrix_vector_multiply(rotation, smplx_param['global_orient'])
                            smplx_param['transl'] = smplx_param['transl'] + pelvis
                            loss = calc_interaction_loss(bodies, contact, object_points_scene[:, :, :, :3],
                                                         scene, args)
                            loss = loss.item()
                        else:
                            loss, smplx_param, bodies = posa_optimize(smplx_param, contact, x, object_points_scene[:, :, :, :3], scene, body_model, global_body_mesh, args)

                    contact_semantic, contact_scene = contact[:, :, 0], contact[:, :, 1]
                    if np.isnan(loss):
                        loss = 2333  # result can be nan
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                        best_sample = {'body': bodies, 'contact': contact, 'smplx_param': smplx_param,
                                       'init_pelvis_frame': x}
                samples.append(best_sample)
                # visualize
                if args.visualize:
                    frame = create_frame(best_sample['init_pelvis_frame'][0])
                    if body_type == 'mesh':
                        bodies, contact = best_sample['body'], best_sample['contact'][:, :, args.contact_dimension]
                        colors = np.array([[0.8, 0.8, 0.8]] * bodies.shape[1])
                        if interaction_model.args.use_contact_feature:
                            colors[contact[0].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                        body_mesh = trimesh.Trimesh(
                            vertices=bodies[0].detach().cpu().numpy(),
                            faces=interaction_model.mesh.faces,
                            vertex_colors=colors,
                        )
                    else:
                        body_mesh = skeleton_to_mesh(bodies[0].detach().cpu().numpy(), color=np.array(color_map(sample_idx / args.sample_num)))
                    body_meshes = [frame, body_mesh]

                    # render synthesis results
                    base_name = str(sample_idx) + '_' + method + '.png'
                    export_file = Path(args.save_dir, args.exp_name, method, action, combination_name, base_name)
                    export_file.parent.mkdir(exist_ok=True, parents=True)
                    img_collage = render_interaction_multview(body=trimesh.util.concatenate(body_meshes),
                                                              smooth_body=False,
                                                              obj_points_coord=object_points_scene[0, 0, :, :3].detach().cpu().numpy(),
                                                              static_scene=scene.mesh if args.full_scene else trimesh.util.concatenate(obj_meshes))
                    img_collage.save(str(export_file))
                    # base_name = str(
                    #     sample_idx) + '_' + method + '_smooth.png'
                    # export_file = Path(args.save_dir, args.exp_name, method, action, combination_name, base_name)
                    # img_collage = render_interaction_multview(body=body_mesh,
                    #                                           obj_points_coord=object_points_scene[0, 0, :,
                    #                                                            :3].detach().cpu().numpy(),
                    #                                           static_scene=scene.mesh if args.full_scene else trimesh.util.concatenate(
                    #                                               obj_meshes))
                    # img_collage.save(str(export_file))
                    # base_name = combination_name + '_' + str(
                    #     sample_idx) + '_' + method + '_body.png'
                    # export_file = Path(args.save_dir, args.exp_name, method, action, base_name)
                    # img_collage = render_body_multview(body=body_mesh,)
                    # img_collage.save(str(export_file))
                    # # export ply mesh
                    # smplx_vertices = interaction_model.body_model(**best_sample['smplx_param']).vertices.detach().cpu().numpy()[0]
                    # body_mesh = trimesh.Trimesh(
                    #     vertices=smplx_vertices,
                    #     faces=interaction_model.mesh.meshes[0].faces,
                    #     vertex_colors=colors,
                    # )
                    # base_name = combination_name + '_' + str(
                    #     sample_idx) + '_' + method + '.ply'
                    # mesh_path = Path(args.save_dir, args.exp_name, method, action, base_name)
                    # body_mesh.export(mesh_path)

                    result_path = Path(args.save_dir, args.exp_name, method, action, combination_name, str(
                        sample_idx) + '.pkl')
                    result = {
                        'smplx_param': params2numpy(best_sample['smplx_param']),
                        'obj_category': obj_category,
                        'obj_id': obj_id,
                        'action': action,
                    }
                    with open(result_path, 'wb') as result_file:
                        pickle.dump(result, result_file)

            # write smplx results
            synthesis_results = [sample['smplx_param'] for sample in samples]
            result_path = Path(args.save_dir, args.exp_name, method, action, combination_name, 'results.pkl')
            result_path.parent.mkdir(exist_ok=True, parents=True)
            print(result_path)
            with open(result_path, 'wb') as result_file:
                pickle.dump(synthesis_results, result_file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--transform_checkpoint", type=str, default='pelvis.ckpt')
    parser.add_argument("--interaction_checkpoint", type=str, default='body.ckpt')
    parser.add_argument("--save_dir", type=str, default="two_stage")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--scene_name", type=str, default="test")
    parser.add_argument("--data_name", type=str, default="general")
    parser.add_argument("--interaction", type=str, default="all")
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--num_sample", type=int, default=32)
    parser.add_argument("--num_try", type=int, default=10)
    parser.add_argument("--decode", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=1)
    parser.add_argument("--full_scene", type=int, default=0)
    parser.add_argument("--composition", type=int, default=0)
    parser.add_argument("--contact_dimension", type=int, default=1, help="0:semantic, 1:scene")

    parser.add_argument("--lr_pelvis", type=float, default=0.1)
    parser.add_argument("--weight_prob_pelvis", type=float, default=0.5)
    parser.add_argument("--max_step_pelvis", type=int, default=100)

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_body", type=float, default=1e-2)
    parser.add_argument("--lr_posa", type=float, default=1.0)
    parser.add_argument("--weight_prob_body", type=float, default=0.)
    parser.add_argument("--weight_init", type=float, default=0)
    parser.add_argument("--weight_z", type=float, default=10)
    parser.add_argument("--weight_contact_semantic", type=float, default=1.0)
    parser.add_argument("--weight_contact_scene", type=float, default=0.0)
    parser.add_argument("--weight_penetration", type=float, default=10.0)
    parser.add_argument("--weight_pose", type=float, default=100.0)
    parser.add_argument("--opt_pose", type=int, default=1)
    parser.add_argument("--max_step_body", type=int, default=100)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--annealing", type=int, default=0)

    parser.add_argument("--obj_path", type=str, default='data/test_room/sofa.ply')
    parser.add_argument("--obj_category", type=str, default='sofa')
    parser.add_argument("--obj_id", type=str, default='0')
    parser.add_argument("--scene_path", type=str, default='data/test_room/room.ply')
    parser.add_argument("--action", type=str, default='sit on')
    args = parser.parse_args()
    if args.data_name == 'shapenet':
        obj_category, obj_id = args.obj_path.split('/')[-3:-1]
        args.save_dir = Path(shapenet_real_dir) / obj_category / obj_id / args.action.split(' ')[0] / 'candidates'
    elif args.data_name == 'replica':
        args.save_dir = Path(replica_dir) / 'gmd_poses' / args.scene_name / (args.action.split(' ')[0] + '-' + args.obj_category + '-' + args.obj_id) / 'candidates'
    else:
        args.save_dir = Path('save/coins_results') / args.save_dir / args.scene_name

    device = torch.device('cuda')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    transform_model = LitTransformNet.load_from_checkpoint(checkpoint_folder / args.transform_checkpoint).to(device)
    interaction_model = LitInteraction.load_from_checkpoint(checkpoint_folder / args.interaction_checkpoint).to(device)
    interaction_model.eval()
    transform_model.eval()
    body_model = interaction_model.body_model = smplx.create(smplx_model_folder, model_type='smplx',
                                                   gender='neutral', ext='npz',
                                                   num_pca_comps=num_pca_comps, batch_size=1).to(device)
    global_body_mesh = interaction_model.mesh
    body_type = interaction_model.args.body_type
    color_map = pylab.get_cmap('gist_yarg')

    torch.manual_seed(233)
    np.random.seed(233)
    z_pelvis_global = np.random.randn(args.num_sample, args.num_try, transform_model.args.latent_dim).astype(np.float32)
    z_body_global = np.random.randn(args.num_sample, args.num_try, interaction_model.args.latent_dim).astype(np.float32)


    interaction_candidates = {
        args.action: [{'obj_category': args.obj_category,
                       'obj_id': args.obj_id,
                       'obj_path': args.obj_path,
                       'scene_path': args.scene_path,
                       'scene_name': args.scene_name,
                       }],
    }
    if args.data_name == 'general':
        general_sample(method='optimization_after_get_body', interaction_candidates=interaction_candidates)
    elif args.data_name == 'shapenet':
        shapenet_sample(method='optimization_after_get_body', interaction_candidates=interaction_candidates)
    elif args.data_name == 'replica':
        two_stage_sample(scene_name=args.scene_name, interaction_candidates=interaction_candidates, method='optimization_after_get_body')
    else:
        raise ValueError
