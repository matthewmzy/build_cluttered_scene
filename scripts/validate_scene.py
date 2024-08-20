import os
import sys
import random
import argparse
from datetime import datetime
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from isaacgym import gymapi

import trimesh as tm

# 获取当前脚本文件的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 将项目根目录添加到sys.path中，这样可以导入utils包
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation as R

from utils.hand_model_lite import HandModelMJCFLite
from utils.hand_model import HandModel

import torch
import transforms3d
import pytorch3d

parser = argparse.ArgumentParser()
parser.add_argument('--scene_file',type=str, default="scenes_2_.npy")
parser.add_argument('--index',type=int,default=0)
parser.add_argument('--vis_all',action='store_true')
parser.add_argument('--vis_val',action='store_true')
parser.add_argument('--save_result',action='store_true')
parser.add_argument('--data_path',type=str,default='data/dataset')

args=parser.parse_args()

class Scene():
    """
    in this class, we define a classic scene, where multiple objects
    are layed randomly on a table. The hand model was also defined
    in this class
    """
    def __init__(self, scene_dict):
        self.scene_dict = scene_dict
        self.code_list = self.scene_dict['code_list']
        self.device = 'cuda:0'
        self.joint_names = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
        self.translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
        self.rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
        self.use_visual_mesh = False
        self.mesh_path_root = 'data/meshdata'
        self.num_samples = 4096
        self.table_size = [0.8,0.8,0.5]

        self.target_translation = [0, 0, 1.5]
     

    def build_hand(self, code):
        OW_pos = self.scene_dict[code][:3]
        OW_rot = self.scene_dict[code][3:]
        OW_rot_mat = R.from_quat(OW_rot).as_matrix()
        # OW_rot_mat = transforms3d.quaternions.quat2mat(OW_rot)
        grasp_dict_path = os.path.join(args.data_path,'graspnet-'+code+'.npy')
        grasp_data = np.load(grasp_dict_path, allow_pickle=True)
        batch_size = grasp_data.shape[0]
        if batch_size==0:
            return None,0,None
        # hand_file = "mjcf/shadow_hand_vis.xml" if self.use_visual_mesh else "mjcf/shadow_hand_wrist_free.xml"
        # hand_model_lite = HandModelMJCFLite(hand_file,"mjcf/meshes")
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=self.device
        )
        hand_state = []
        # scale_tensor = []
        for index in range(batch_size):
            qpos = grasp_data[index]['qpos']
            HO_pos = [qpos[name] for name in self.translation_names]
            HO_rot_mat = np.array(transforms3d.euler.euler2mat(
                *[qpos[name] for name in self.rot_names]))
            
            OW_pos = np.array(OW_pos)
            HO_pos = np.array(HO_pos).reshape((3, 1))
            HW_pos = OW_pos+np.dot(OW_rot_mat, HO_pos).reshape((3,))

            HW_rot = OW_rot_mat@HO_rot_mat
            rot = HW_rot[:, :2].T.ravel().tolist()
            hand_pose = torch.tensor(HW_pos.tolist() + rot + [qpos[name]
                                    for name in self.joint_names], dtype=torch.float, device=self.device)
            hand_state.append(hand_pose)
        hand_state = torch.stack(hand_state).to(self.device)
        # print(hand_state.shape)
        hand_model.set_parameters(hand_state)

        return hand_model, batch_size
        
    def get_scene_pc(self, spetial_code):
        tm_scene = tm.Scene()
        meshes = []
        table_mesh = tm.primitives.Box(self.table_size)
        table_mesh.apply_translation([0,0,self.table_size[2]/2])
        for code in self.code_list:
            mesh_path = os.path.join(self.mesh_path_root,'graspnet-'+code,'coacd','decomposed.obj')
            pos = np.array(self.scene_dict[code][:3])
            rot = R.from_quat(self.scene_dict[code][3:]).as_matrix()
            # scale = 10
            mesh = tm.load(mesh_path,force='mesh')
            transform = np.eye(4)
            transform[:3,:3] = rot
            transform[:3,3] = pos
            # mesh.apply_scale(scale)
            mesh.apply_transform(transform)
            if code == spetial_code:
                mesh.visual.vertex_colors = np.array([0, 0, 255, 255], dtype=np.uint8)
                origin_mesh = tm.load(mesh_path,force='mesh')
                target_obj_mesh = origin_mesh.copy().apply_translation(self.target_translation)
            else:
                mesh.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
            tm_scene.add_geometry(mesh)
            meshes.append(mesh)
        combined_mesh = tm.util.concatenate(meshes)
        # tm_scene.show()
        # combined_mesh.show()
        vertices = torch.tensor(combined_mesh.vertices, dtype=torch.float, device=self.device)
        faces = torch.tensor(combined_mesh.faces, dtype=torch.float, device=self.device)
        mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
        obj_surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]

        x = torch.arange(-self.table_size[0]/2,self.table_size[0]/2,0.01)
        y = torch.arange(-self.table_size[1]/2,self.table_size[1]/2,0.01)
        xn = x.shape[0]
        yn = y.shape[0]
        x = x.repeat(yn).to(self.device)
        y = y.repeat_interleave(xn).to(self.device)
        z = torch.full_like(x,self.table_size[2], device=self.device)
        table_surface_points = torch.stack([x,y,z],dim=1)

        surface_points = torch.cat([obj_surface_points,table_surface_points],dim=0)

        combined_mesh = tm.util.concatenate([combined_mesh,table_mesh])

        return surface_points, combined_mesh
    
    def build_target_hand(self, code, idx):
        grasp_dict_path = os.path.join(args.data_path,'graspnet-'+code+'.npy')
        grasp_data = np.load(grasp_dict_path, allow_pickle=True)
        batch_size = grasp_data.shape[0]
        if batch_size==0:
            return None
        
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=self.device
        )

        qpos = grasp_data[idx]['qpos']
        HO_pos = [qpos[name] + self.target_translation[i] for i, name in enumerate(self.translation_names)]
        HO_rot_mat = np.array(transforms3d.euler.euler2mat(
            *[qpos[name] for name in self.rot_names]))

        rot = HO_rot_mat[:, :2].T.ravel().tolist()
        hand_pose = torch.tensor(HO_pos + rot + [qpos[name]
                                for name in self.joint_names], dtype=torch.float, device=self.device).unsqueeze(0)

        hand_model.set_parameters(hand_pose)
        target_hand_mesh = hand_model.get_trimesh_data(0)
        target_hand_mesh.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        return target_hand_mesh


    def validate_grasp(self, code, vis_all, vis_val, pen_thre = 0.002):
        hand_model, batch_size = self.build_hand(code)
        if batch_size==0:
            return None,0
        
        scene_pc, scene_mesh = self.get_scene_pc(code)

        scene_pc_tensor = scene_pc.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        dis = hand_model.cal_distance(scene_pc_tensor)
        valid_dis = (dis < pen_thre).all(dim=1)
        valid_grasp = torch.nonzero(valid_dis).ravel().tolist()

        if vis_all:
            for idx in range(batch_size):
                print(f"{idx} ---- {'valid' if idx in valid_grasp else 'invalid'}")
                hand_mesh = hand_model.get_trimesh_data(idx)
                target_hand_mesh = self.build_target_hand(code,idx)
                hand_mesh.visual.vertex_colors = np.array([0, 255, 0, 255], dtype=np.uint8)
                (hand_mesh+scene_mesh).show()
        elif vis_val and len(valid_grasp)!=0:
            for idx in valid_grasp:
                print(str(idx)+' ---- valid')
                hand_mesh = hand_model.get_trimesh_data(idx)
                # target_hand_mesh = self.build_target_hand(code,idx)
                hand_mesh.visual.vertex_colors = np.array([0, 255, 0, 255], dtype=np.uint8)
                (hand_mesh+scene_mesh).show()

        return valid_grasp, batch_size

scene_file = os.path.join("data/visualized_data",args.scene_file)
scene = Scene(np.load(scene_file,allow_pickle=True)[args.index])

random.shuffle(scene.code_list)

# # visualize point cloud 
# print(">>>>begin")
# pc, mesh = scene.get_scene_pc(None)
# meshes = [mesh]
# for p in pc:
#     sphere = tm.creation.icosphere(radius=0.002, subdivisions=2)
#     sphere.apply_translation(p.tolist())
#     sphere.visual.face_colors = [0, 255, 0, 255]
#     meshes.append(sphere)
# tm.Scene(meshes).show()
# print(">>>>end")

valid_index_dict = {}

if args.save_result:
    with open("data/result",mode='a',encoding='utf-8') as file:
        file.write(f"{datetime.now()}\n")
        file.write(f"{scene_file}:\n")
for code in scene.code_list:

    valid_grasp, batch_size = scene.validate_grasp(code, vis_all = args.vis_all, vis_val = args.vis_val)
    valid_index_dict[code]=valid_grasp

    print(f">>> code {code} ---- valid_grasp: {valid_grasp}, batch_size: {batch_size}")
    if args.save_result:
        with open("data/result",mode='a',encoding='utf-8') as file:
            if batch_size==0:
                file.write(f"{scene_file[-13:-4]} code {code} has no valid grasps, let alone in scene\n")
            else:
                file.write(f"{scene_file[-13:-4]} code {code} {len(valid_grasp)}/{batch_size} : {valid_grasp}\n")

    
# acquire gym interface
gym = gymapi.acquire_gym()

device = args.device

# configure sim
sim_params = gymapi.SimParams()

sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0., -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

sim_params.use_gpu_pipeline = False

# set ground plane parameters
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# get all object codes
obj_code = scene.code_list

# options for loading table asset
table_size = gymapi.Vec3(0.6,0.6,0.4)
table_asset_options = gymapi.AssetOptions()
table_asset_options.fix_base_link = True

num_per_row = 6
spacing = 0.4
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 4* spacing)

# options for loading 8 object assets
obj_asset_root = "data/meshdata"
obj_asset_files = {}
for code in obj_code:
    obj_asset_files['graspnet-' + code] = 'graspnet-' + code + "/coacd/coacd_simple.urdf"

obj_asset_options = gymapi.AssetOptions()
obj_asset_options.override_com = True
obj_asset_options.override_inertia = True
obj_asset_options.density = 1000
obj_asset_options.max_angular_velocity = 0.1
obj_asset_options.max_linear_velocity = 0.1

# options for loading hand asset
hand_asset_root = "open_ai_assets"
hand_asset_file = "hand/shadow_hand.xml"
hand_asset_options = gymapi.AssetOptions()
hand_asset_options.disable_gravity = True
hand_asset_options.fix_base_link = True
hand_asset_options.collapse_fixed_joints = True


def _create_sim():
    # create sim    
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # position the camera
    cam_pos = gymapi.Vec3(13.0, 7.0, 10.0)
    cam_target = gymapi.Vec3(0.0, 7.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    table_asset = gym.create_box(sim, table_size.x, table_size.y, table_size.z, table_asset_options)
    obj_assets_all = {}
    for code in obj_asset_files.keys():
        obj_assets_all[code[9:12]] = gym.load_asset(sim, obj_asset_root, obj_asset_files[code], obj_asset_options)
    hand_asset = gym.load_asset(sim, hand_asset_root, hand_asset_file, hand_asset_options)

    return sim,viewer,table_asset,obj_assets_all,hand_asset

for code in obj_code:

    grasp_data = np.load(os.path.join(args.data_path,'graspnet-'+code+'.npy'),allow_pickle=True) # grasp poses
    grasp_num = len(valid_index_dict[code])

    # cache useful handles
    envs = []
    obj_idxs = []
    hand_idxs = []
    hand_handles = []
    obj_handles = []

    sim,viewer,table_asset,obj_assets_all,hand_asset=_create_sim()

    for index in range(grasp_num):
        
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_size.z)
        table_handle = gym.create_actor(env, table_asset, table_pose, "table", index, 0)

        # set table restitution
        shape_props = gym.get_actor_rigid_shape_properties(env, table_handle)
        shape_props[0].restitution = 0
        gym.set_actor_rigid_shape_properties(env, table_handle, shape_props)

        # add actor(all objects)
        pose = gymapi.Transform()
        for idx,cd in enumerate(obj_code):

            pose.p = gymapi.Vec3(*scene.scene_dict[cd][:3])
            pose.r = gymapi.Quat(*scene.scene_dict[cd][-4:])
            obj_handle = gym.create_actor(env, obj_assets_all[cd], pose, f"env_{index}_obj_{idx}_{cd}", index, 0)
            color = gymapi.Vec3(np.random.uniform(0, 0.1), np.random.uniform(0, 0.1), np.random.uniform(0, 0.1))
            if cd == code:
                gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0.8,0)) 
            else:
                gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)            

            obj_shape_props = gym.get_actor_rigid_shape_properties(
                env, obj_handle)

            for i in range(len(obj_shape_props)):
                obj_shape_props[i].friction = 3.
            gym.set_actor_rigid_shape_properties(env, obj_handle,
                                                    obj_shape_props)

            # set coefficience of restitution
            shape_props = gym.get_actor_rigid_shape_properties(env, obj_handle)
            for sp_prop in shape_props:
                sp_prop.restitution = 0
            gym.set_actor_rigid_shape_properties(env, obj_handle, shape_props)

            # set mass
            rb_props = gym.get_actor_rigid_body_properties(env, obj_handle)
            for rb_prop in rb_props:
                rb_prop.mass = 10
            gym.set_actor_rigid_body_properties(env, obj_handle, rb_props)
            
            # get global index of object in rigid body state tensor
            obj_idx = gym.get_actor_rigid_body_index(env, obj_handle, 0, gymapi.DOMAIN_SIM)
            obj_idxs.append(obj_idx)
            obj_handles.append(obj_handle)

        # add hand
        qpos = grasp_data[valid_index_dict[code][index]]
        rot = [qpos[name] for name in scene.rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        translation = np.array([qpos[name] for name in scene.translation_names])
        hand_pose = np.array([qpos[name] for name in scene.joint_names])

        pose = gymapi.Transform()
        pose.r = gymapi.Quat


    while not gym.query_viewer_has_closed(viewer):
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)








    
