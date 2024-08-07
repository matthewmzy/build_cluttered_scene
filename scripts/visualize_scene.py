import os
from isaacgym import gymapi
import numpy as np
import random

import argparse
import time


# np.random.seed(543)


parser = argparse.ArgumentParser(description="visulize options")

parser.add_argument('--data_root', type=str, default='isaacgym_simu/data/scene_data')
parser.add_argument('--data_file', type=str, default='scenes_1.npy')
parser.add_argument('--save_root', type=str, default='isaacgym_simu/data/visualized_data')
parser.add_argument('--save_file', type=str, default='scenes_1_.npy')
parser.add_argument('--save_result', action='store_true')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=234)

args = parser.parse_args()

random.seed(args.seed)

data_file = os.path.join(args.data_root, args.data_file)
save_file = os.path.join(args.save_root, args.save_file)
save_result = args.save_result

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
obj_code = os.listdir("isaacgym_simu/data/meshdata")

# options for loading 8 object assets
obj_asset_root = "isaacgym_simu/data/meshdata"
obj_asset_files = {}
for code in obj_code:
    obj_asset_files[code] = code + "/coacd/coacd.urdf"

obj_asset_options = gymapi.AssetOptions()
obj_asset_options.override_com = True
obj_asset_options.override_inertia = True
obj_asset_options.density = 10000
obj_asset_options.max_angular_velocity = 0.01
obj_asset_options.max_linear_velocity = 0.01

# options for loading table asset
table_size = gymapi.Vec3(1.,1.,0.5)
table_asset_options = gymapi.AssetOptions()
table_asset_options.fix_base_link = True

num_per_row = 6
spacing = 1.3
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 4* spacing)
    
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

    obj_assets_all = {}
    for code in obj_asset_files.keys():
        obj_assets_all[code[9:12]] = gym.load_asset(sim, obj_asset_root, obj_asset_files[code], obj_asset_options)
    table_asset = gym.create_box(sim, table_size.x, table_size.y, table_size.z, table_asset_options)

    return sim,viewer,obj_assets_all,table_asset

data_dict = np.load(data_file,allow_pickle=True)

scene_num = len(data_dict)

# cache useful handles
envs = []
obj_idxs = []
hand_idxs = []
hand_handles = []
bound_handles = []
obj_handles = []

sim,viewer,obj_assets_all,table_asset=_create_sim()

for index in range(scene_num):
    data = data_dict[index]

    obj_assets = []
    for code in data['code_list']:
        obj_assets.append(obj_assets_all[code])
    
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
    for idx,code in enumerate(data['code_list']):

        pose.p = gymapi.Vec3(*data[code][:3])
        pose.r = gymapi.Quat(*data[code][-4:])
        obj_handle = gym.create_actor(env, obj_assets[idx], pose, f"env_{index}_obj_{idx}_{code}", index, 0)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
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

def check_state(state):
    x = state['pose']['p']['x']
    y = state['pose']['p']['y']
    z = state['pose']['p']['z']
    rx = state['pose']['r']['x']
    ry = state['pose']['r']['y']
    rz = state['pose']['r']['z']
    rw = state['pose']['r']['w']
    vx = state['vel']['linear']['x']
    vy = state['vel']['linear']['y']
    vz = state['vel']['linear']['z']
    ax = state['vel']['angular']['x']
    ay = state['vel']['angular']['y']
    az = state['vel']['angular']['z']
    flag_p = (x>-table_size.x*2/3) and (x<table_size.x*2/3) and (y>-table_size.y*2/3) \
        and (y<table_size.y*2/3) and (z>table_size.z) and (z<table_size.z*2)
    flag_v = vx**2+vy**2+vz**2 < 0.01
    flag_a = ax**2+ay**2+az**2 < 0.1
    return flag_p and flag_v and flag_a,[x,y,z,rx,ry,rz,rw]

save_lst = []
start_time = time.time()
flag = [False for i in range(20)]

while not gym.query_viewer_has_closed(viewer):

    n = round((time.time()-start_time))/5
    if n in [float(r) for r in range(5)] and flag[int(n)]==False:
        flag[int(n)]=True
        cnt = 0
        for idx in range(len(envs)):
            env = envs[idx]
            for i in range(len(data_dict[idx]['code_list'])):
                handle = obj_handles[cnt]
                cnt += 1
                state = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_ALL)
                state['vel']['linear']['x']=0
                state['vel']['linear']['y']=0
                state['vel']['linear']['z']=0
                state['vel']['angular']['x']=0
                state['vel']['angular']['y']=0
                state['vel']['angular']['z']=0
                gym.set_actor_rigid_body_states(env, handle, state, gymapi.STATE_ALL)                
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    if time.time()-start_time > 40:
        cnt = 0
        for idx in range(len(envs)):
            env = envs[idx]
            result_dict = {}
            result_dict['code_list']=[]

            for i in range(len(data_dict[idx]['code_list'])):
                handle = obj_handles[cnt]
                cnt += 1
                state = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_ALL)
                name = gym.get_actor_name(env, handle)
                reasonable, pose_list = check_state(state[0])
                if reasonable:
                    name = gym.get_actor_name(env, handle)
                    _,env_index,_,indice_in_env,code = name.split('_')
                    result_dict['code_list'].append(code)
                    result_dict[code]=pose_list
            print(result_dict['code_list'])
            if len(result_dict['code_list'])>7:
                save_lst.append(result_dict)
        if save_result:
            time.sleep(3)
            assert cnt == len(obj_handles)
            break

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

if save_result:
    np.save(save_file,save_lst,allow_pickle=True)

