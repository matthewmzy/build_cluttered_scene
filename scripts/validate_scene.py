import os
import sys
import random
import argparse
from datetime import datetime

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
parser.add_argument('--scene_file',type=str, default="scenes_1_.npy")
parser.add_argument('--index',type=int,default=0)
parser.add_argument('--visualize',action='store_true')
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
        self.num_samples = 2000
        self.table_size = [1.,1.,0.5]

        self.target_translation = [0, 0, 1.5]
     

    def build_hand(self, code):
        OW_pos = self.scene_dict[code][:3]
        OW_rot = self.scene_dict[code][3:]
        # OW_rot_mat = R.from_quat(OW_rot).as_matrix()
        OW_rot_mat = transforms3d.quaternions.quat2mat(OW_rot)
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
            HO_pos = np.array(HO_pos).reshape((3,1))
            HW_pos = OW_pos+np.dot(OW_rot_mat,HO_pos).reshape((3,))

            HW_rot = OW_rot_mat@HO_rot_mat
            rot = HW_rot[:, :2].T.ravel().tolist()
            hand_pose = torch.tensor(HW_pos.tolist() + rot + [qpos[name]
                                    for name in self.joint_names], dtype=torch.float, device=self.device)
            hand_state.append(hand_pose)
        hand_state = torch.stack(hand_state).to(self.device)
        # print(hand_state.shape)
        hand_model.set_parameters(hand_state)

        target_hand_mesh = self.build_target_hand(code)

        return hand_model, batch_size, target_hand_mesh
        
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

        x = torch.rand(self.num_samples, device=self.device)*self.table_size[0]-self.table_size[0]/2
        y = torch.rand(self.num_samples, device=self.device)*self.table_size[1]-self.table_size[1]/2
        z = torch.full_like(x,self.table_size[2], device=self.device)
        table_surface_points = torch.stack([x,y,z],dim=1)

        surface_points = torch.cat([obj_surface_points,table_surface_points],dim=0)

        combined_mesh = tm.util.concatenate([combined_mesh,table_mesh,target_obj_mesh])

        return surface_points, combined_mesh
    
    def build_target_hand(self, code):
        grasp_dict_path = os.path.join('data/dataset','graspnet-'+code+'.npy')
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

        qpos = grasp_data[0]['qpos']
        HO_pos = [qpos[name] + self.target_translation[i] for i, name in enumerate(self.translation_names)]
        HO_rot_mat = np.array(transforms3d.euler.euler2mat(
            *[qpos[name] for name in self.rot_names]))

        rot = HO_rot_mat[:, :2].T.ravel().tolist()
        hand_pose = torch.tensor(HO_pos + rot + [qpos[name]
                                for name in self.joint_names], dtype=torch.float, device=self.device).unsqueeze(0)

        hand_model.set_parameters(hand_pose)
        target_hand_mesh = hand_model.get_trimesh_data(0)

        return target_hand_mesh


    def validate_grasp(self, code, visualize, pen_thre = 0.001):
        hand_model, batch_size, target_hand_mesh = self.build_hand(code)
        if batch_size==0:
            return None,0
        hand_mesh = hand_model.get_trimesh_data(0)

        scene_pc, scene_mesh = self.get_scene_pc(code)

        if visualize:
            hand_mesh.visual.vertex_colors = np.array([0, 255, 0, 255], dtype=np.uint8)
            (hand_mesh+scene_mesh+target_hand_mesh).show()
        scene_pc_tensor = scene_pc.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        dis = hand_model.cal_distance(scene_pc_tensor)
        valid_dis = (dis < pen_thre).all(dim=1)
        valid_grasp = torch.nonzero(valid_dis).squeeze()
        return valid_grasp, batch_size

scene_file = os.path.join("data/visualized_data",args.scene_file)
scene = Scene(np.load(scene_file,allow_pickle=True)[args.index])

random.shuffle(scene.code_list)

if args.save_result:
    with open("data/result",mode='a',encoding='utf-8') as file:
        file.write(f"{datetime.now()}\n")
        file.write(f"{scene_file}:\n")
for code in scene.code_list:
    if code in ['077']:
        continue
    # print(code)
    # try:
    valid_grasp, batch_size = scene.validate_grasp(code,visualize= args.visualize)
    # except FileNotFoundError:
    #     continue
    # else:
    print(f">>> valid_grasp: {valid_grasp}, batch_size: {batch_size}")
    # TODO: valid_grasp might be int?
    if args.save_result:
        with open("data/result",mode='a',encoding='utf-8') as file:
            if batch_size==0 or (not isinstance(valid_grasp.tolist(), list)):
                file.write(f"{scene_file[-13:-4]} code {code} has no valid grasps, let alone in scene\n")
            else:
                valid_grasp = valid_grasp.tolist()                
                file.write(f"{scene_file[-13:-4]} code {code} {len(valid_grasp)}/{batch_size} : {valid_grasp}\n")






    
