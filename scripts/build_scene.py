import os
from isaacgym import gymapi, gymutil
import random
import math
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='build scene options')

parser.add_argument('--gui', action='store_false')
parser.add_argument('--result_path', type=str, default="isaacgym_simu/data/scene_data")
parser.add_argument('--result_file', type=str, default="scenes_3.npy")
parser.add_argument('--mesh_path', type=str, default="isaacgym_simu/data/meshdata")
parser.add_argument('--total_batch', type=int, default=192)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dense', action='store_true')

args = parser.parse_args()

# acquire gym interface
gym = gymapi.acquire_gym()

# define result path
result_path = args.result_path
save_lst = []

# set torch device
device = 'cuda:0'#args.sim_device if args.use_gpu_pipeline else 'cpu'

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
obj_code = os.listdir(args.mesh_path)
bad_code = ['graspnet-027',
            'graspnet-018',
            'graspnet-043']
for code in bad_code:
    obj_code.remove(code)

# options for loading 8 object assets
obj_asset_root = args.mesh_path
obj_asset_files = [f"{code}/coacd/coacd_simple.urdf" for code in obj_code]

obj_asset_options = gymapi.AssetOptions()
obj_asset_options.override_com = True
obj_asset_options.override_inertia = True
# obj_asset_options.density = 10000
# obj_asset_options.vhacd_enabled = True
obj_asset_options.max_angular_velocity = 0.2
obj_asset_options.max_linear_velocity = 2

# options for loading hand asset
hand_asset_root = "isaacgym_simu/open_ai_assets"
hand_asset_file = "hand/shadow_hand.xml"

hand_asset_options = gymapi.AssetOptions()
hand_asset_options.disable_gravity = True
hand_asset_options.fix_base_link = True
hand_asset_options.collapse_fixed_joints = True

# options for loading table asset
table_size = gymapi.Vec3(0.6,0.6,0.4)
table_asset_options = gymapi.AssetOptions()
table_asset_options.fix_base_link = True

# options for loading assisting box boundary
w = table_size.x / 4.5
t = w/3 if args.dense else w/4
h = 1.
bound_asset_options = gymapi.AssetOptions()
bound_asset_options.fix_base_link = True

# set up the env grid
total_batch = args.total_batch # grasp_dict.shape[0]
batch_size = args.batch_size
num_per_row = int(math.sqrt(batch_size))
spacing = 0.8
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 4* spacing)
safe_gap = 0.25
sample_num = 10

# define how many objects are reasonable in a scene and unzip pose
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
    flag_v = vx**2+vy**2+vz**2 < 0.5
    flag_a = ax**2+ay**2+az**2 < 8
    return flag_p and flag_v and flag_a,[x,y,z,rx,ry,rz,rw]
    

def _create_sim():
    # create sim    
    
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    gym.add_ground(sim, plane_params)

    # create viewer
    if args.gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(6.0, 5.0, 6.0)
        cam_target = gymapi.Vec3(0.0, 5.0, 6.0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    else:
        viewer = None

    obj_assets_all = [gym.load_asset(sim, obj_asset_root, file, obj_asset_options) for file in obj_asset_files]
    hand_asset = gym.load_asset(sim, hand_asset_root, hand_asset_file, hand_asset_options)
    table_asset = gym.create_box(sim, table_size.x, table_size.y, table_size.z, table_asset_options)
    bound_assets = [gym.create_box(sim, 2*w, t, h, bound_asset_options),
                   gym.create_box(sim, 2*w, t, h, bound_asset_options),
                   gym.create_box(sim, t, 2*w, h, bound_asset_options),
                   gym.create_box(sim, t, 2*w, h, bound_asset_options),
                   gym.create_box(sim, 2*w, 2*w, t, bound_asset_options)]

    return sim,viewer,obj_assets_all,hand_asset,table_asset,bound_assets

# view in batches
offset = 0
for batch in range(total_batch//batch_size):
    offset_ = min(offset + batch_size, total_batch)
    print(f"batch {batch+1} : adding {offset_-offset} envs to sim")

    # cache useful handles
    envs = []
    obj_idxs = []
    hand_idxs = []
    hand_handles = []
    bound_handles = []
    obj_handles = []

    sim,viewer,obj_assets_all,hand_asset,table_asset,bound_assets=_create_sim()

    # add a batch of envs
    for index in range(offset, offset_):

        scene_objs_idx = random.sample(range(len(obj_code)),k=sample_num)
        scene_objs_code = [obj_code[idx]for idx in scene_objs_idx]

        obj_assets = [obj_assets_all[idx]for idx in scene_objs_idx]
        
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

        # add bound
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(0.0, w+t/2, h/2)
        bound_handles.append(gym.create_actor(env, bound_assets[0], box_pose, "bound_1", index, 0))
        box_pose.p = gymapi.Vec3(0.0, -w-t/2, h/2)
        bound_handles.append(gym.create_actor(env, bound_assets[1], box_pose, "bound_2", index, 0))
        box_pose.p = gymapi.Vec3(w+t/2, 0.0, h/2)
        bound_handles.append(gym.create_actor(env, bound_assets[2], box_pose, "bound_3", index, 0))
        box_pose.p = gymapi.Vec3(-w-t/2, 0.0, h/2)
        bound_handles.append(gym.create_actor(env, bound_assets[3], box_pose, "bound_4", index, 0))
        box_pose.p = gymapi.Vec3(0.0, 0.0, h+t/2)
        bound_handles.append(gym.create_actor(env, bound_assets[4], box_pose, "bound_5", index, 0))

        # add actor(all objects)
        pose = gymapi.Transform()
        for idx in range(sample_num):
            pose.p = gymapi.Vec3(20, 20, table_size.z+(2+idx)*safe_gap)
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            obj_handle = gym.create_actor(env, obj_assets[idx], pose, f"env_{index}_obj_{idx}_{scene_objs_code[idx][-3:]}", index, 0)
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

    # simulate a batch of envs
    thred = 0.1
    start_time = time.time()
    flag = [False for i in range(sample_num)]

    while True: 
        # check object states and save results
        n = round((time.time()-start_time))/10
        if n in [float(r) for r in range(sample_num)] and flag[int(n)]==False:
            flag[int(n)]=True
            print('>>>>>'+str(n))
            for idx in range(len(envs)):
                env = envs[idx]
                for handle in obj_handles[idx*sample_num:(idx+1)*sample_num]:
                    state = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_ALL)
                    state['vel']['linear']['x']=0
                    state['vel']['linear']['y']=0
                    state['vel']['linear']['z']=0
                    state['vel']['angular']['x']=0
                    state['vel']['angular']['y']=0
                    state['vel']['angular']['z']=0
                    gym.set_actor_rigid_body_states(env, handle, state, gymapi.STATE_ALL)


                handle = obj_handles[idx*sample_num+int(n)]
                p = 8 if args.dense else 5
                state = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_ALL)
                state['pose']['p']['x']=random.uniform(-table_size.x/p,table_size.x/p)
                state['pose']['p']['y']=random.uniform(-table_size.y/p,table_size.y/p)
                state['pose']['p']['z']=table_size.z+0.4
                state['vel']['linear']['x']=0
                state['vel']['linear']['y']=0
                state['vel']['linear']['z']=0
                state['vel']['angular']['x']=0
                state['vel']['angular']['y']=0
                state['vel']['angular']['z']=0                
                gym.set_actor_rigid_body_states(env, handle, state, gymapi.STATE_ALL)

        if time.time()-start_time > 20+sample_num*10:
            for idx in range(len(envs)):
                env = envs[idx]
                result_dict = {}
                result_dict['code_list']=[]
                
                for i in range(sample_num):
                    handle = obj_handles[idx*sample_num+i]
                    state = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_ALL)
                    name = gym.get_actor_name(env, handle)
                    reasonable, pose_list = check_state(state[0])
                    if reasonable:
                        name = gym.get_actor_name(env, handle)
                        _,env_index,_,indice_in_env,code = name.split('_')
                        result_dict['code_list'].append(code)
                        result_dict[code]=pose_list
                if len(result_dict['code_list'])>8:
                    save_lst.append(result_dict)
            break
        if args.gui and gym.query_viewer_has_closed(viewer):
            break
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if args.gui:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)


    
    print(f"batch {batch+1} Done")
    if args.gui:
        print("viewer is closing")
        # time.sleep(5)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    offset = offset_

np.save(os.path.join(args.result_path,args.result_file),save_lst,allow_pickle=True)

