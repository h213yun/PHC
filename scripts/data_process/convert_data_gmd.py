# from ast import Try
import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
from scipy import interpolate
import glob
import os
import sys
import pdb
import os.path as osp
import argparse

sys.path.append(os.getcwd())

# from smpl_sim.utils.config_utils.copycat_config import Config as CC_Config
# from smpl_sim.khrylib.utils import get_body_qposaddr
from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
# from smpl_sim.smpllib.smpl_robot import Robot
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
# from typing import List, Optional
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    6D rotation representation to rotation matrix.
    Args:
        d6: (*, 6) tensor
    Returns:
        (*, 3, 3) rotation matrices
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def fps_20_to_30(mdm_jts):
    """
    Interpolate motion from 20 FPS to 30 FPS (1.5x frames)
    Args:
        mdm_jts: (N, num_joints, 3) array
    Returns:
        interpolated array: (N*1.5, num_joints, 3)
    """
    jts = []
    N = mdm_jts.shape[0]
    num_joints = mdm_jts.shape[1]
    
    for i in range(num_joints):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x)
        f_y = interpolate.interp1d(x, int_y)
        f_z = interpolate.interp1d(x, int_z)
        new_x = f_x(np.linspace(0, N-1, int(N * 1.5)))
        new_y = f_y(np.linspace(0, N-1, int(N * 1.5)))
        new_z = f_z(np.linspace(0, N-1, int(N * 1.5)))
        jts.append(np.stack([new_x, new_y, new_z], axis=1))
    jts = np.stack(jts, axis=1)
    return jts

def interpolate_translation(trans, target_length):
    """
    Interpolate translation data
    Args:
        trans: (N, 3) array
        target_length: target number of frames
    Returns:
        interpolated array: (target_length, 3)
    """
    N = trans.shape[0]
    x = np.arange(0, N)
    
    f_x = interpolate.interp1d(x, trans[:, 0])
    f_y = interpolate.interp1d(x, trans[:, 1])
    f_z = interpolate.interp1d(x, trans[:, 2])
    
    new_x = f_x(np.linspace(0, N-1, target_length))
    new_y = f_y(np.linspace(0, N-1, target_length))
    new_z = f_z(np.linspace(0, N-1, target_length))
    
    return np.stack([new_x, new_y, new_z], axis=1)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert GMD motion data to Isaac format')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input .npy file path (e.g., gmd_walk_guidance_smpl_2_motion.npy)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output .pkl file path (e.g., gmd_isaac_walk_guidance.pkl)')
    parser.add_argument('--smpl_humanoid_xml', type=str, 
                        default="phc/data/assets/mjcf/smpl_humanoid_1.xml",
                        help='Path to SMPL humanoid XML file')
    parser.add_argument('--data_dir', type=str, default="data/smpl",
                        help='SMPL data directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    print(robot_cfg)

    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir=args.data_dir,
    )

    print(f"Loading input file: {args.input}")
    smpl_motion_data = np.load(args.input, allow_pickle=True).item()

    print("Checking data structure...")
    print("thetas shape:", smpl_motion_data['thetas'].shape)
    print("motion shape:", smpl_motion_data['motion'].shape)
    print("root_translation shape:", smpl_motion_data['root_translation'].shape)

    num_frames = smpl_motion_data['motion'].shape[2]  # 120
    num_samples = 1 

    # 6D rotation to axis-angle
    thetas_6d = torch.from_numpy(smpl_motion_data['thetas'])  # (24, 6, 120)
    thetas_6d = thetas_6d.permute(2, 0, 1)  # (120, 24, 6)

    # 6D -> rotation matrix -> axis-angle
    rot_matrices = rotation_6d_to_matrix(thetas_6d)  # (120, 24, 3, 3)
    rot_matrices_flat = rot_matrices.reshape(-1, 3, 3)  # (120*24, 3, 3)

    # scipy Rotation
    rotations = sRot.from_matrix(rot_matrices_flat.numpy())
    pose_aa = rotations.as_rotvec()  # (120*24, 3)
    pose_aa = pose_aa.reshape(num_frames, 24, 3)  # (120, 24, 3)

    # Euler angles
    pose_euler = rotations.as_euler('XYZ', degrees=True)  # (120*24, 3)
    pose_euler = pose_euler.reshape(num_frames, 24, 3)  # (120, 24, 3)

    # root_translation
    root_trans = smpl_motion_data['root_translation'].T  # (120, 3)

    # === 20 FPS to 30 FPS interpolation ===
    print(f"Original frames: {pose_euler.shape[0]}")
    pose_euler = fps_20_to_30(pose_euler)  # (120, 24, 3) -> (180, 24, 3)
    root_trans = interpolate_translation(root_trans, pose_euler.shape[0])  # (120, 3) -> (180, 3)
    print(f"Interpolated frames: {pose_euler.shape[0]} (20 FPS -> 30 FPS)")

    # res_data 
    res_data = {
        'json_file': {
            'thetas': [pose_euler], 
            'root_translation': [root_trans]
        }
    }

    print(f"Converted pose_euler shape: {pose_euler.shape}")
    print(f"Converted root_trans shape: {root_trans.shape}")
    print(f"Number of samples: {len(res_data['json_file']['thetas'])}")

    amass_data = {}
    for i in range(len(res_data['json_file']['thetas'])):
        pose_euler = np.array(res_data['json_file']['thetas'])[i].reshape(-1, 24, 3)
        B = pose_euler.shape[0]
        trans = np.array(res_data['json_file']['root_translation'])[i]
        pose_aa = sRot.from_euler('XYZ', pose_euler.reshape(-1, 3), degrees=True).as_rotvec().reshape(B, 72)

        transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
        new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
        pose_aa[:, :3] = new_root

        trans = trans.dot(transform.as_matrix().T)
        trans[:, 2] = trans[:, 2] - (trans[0, 2] - 0.92)

        amass_data[f"{i}"] = {"pose_aa": pose_aa, "trans": trans, 'beta': np.zeros(10)}

    double = False

    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    amass_full_motion_dict = {}
    for key_name in tqdm(amass_data.keys()):
        key_name_dump = key_name
        smpl_data_entry = amass_data[key_name]
        file_name = f"data/amass/singles/{key_name}.npy"
        B = smpl_data_entry['pose_aa'].shape[0]

        start, end = 0, 0

        pose_aa = smpl_data_entry['pose_aa'].copy()[start:]
        root_trans = smpl_data_entry['trans'].copy()[start:]
        B = pose_aa.shape[0]

        beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
        if len(beta.shape) == 2:
            beta = beta[0]

        gender = smpl_data_entry.get("gender", "neutral")
        fps = 30.0  # Interpolated to 30 FPS

        if isinstance(gender, np.ndarray):
            gender = gender.item()

        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        if gender == "neutral":
            gender_number = [0]
        elif gender == "male":
            gender_number = [1]
        elif gender == "female":
            gender_number = [2]
        else:
            import ipdb
            ipdb.set_trace()
            raise Exception("Gender Not Supported!!")

        smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
        pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

        num = 1
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

        gender_number, beta[:], gender = [0], 0, "neutral"
        print("using neutral model")

        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(args.smpl_humanoid_xml)
        skeleton_tree = SkeletonTree.from_mjcf(args.smpl_humanoid_xml)

        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)

            print("############### filtering!!! ###############")
            import scipy.ndimage.filters as filters
            from smpl_sim.utils.transform_utils import quat_correct
            root_trans_offset = filters.gaussian_filter1d(root_trans_offset, 3, axis=0, mode="nearest")
            root_trans_offset = torch.from_numpy(root_trans_offset)
            pose_quat_global = np.stack([quat_correct(pose_quat_global[:, i]) for i in range(pose_quat_global.shape[1])], axis=1)

            filtered_quats = filters.gaussian_filter1d(pose_quat_global, 2, axis=0, mode="nearest")
            pose_quat_global = filtered_quats / np.linalg.norm(filtered_quats, axis=-1)[..., None]
            print("############### filtering!!! ###############")
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()

        new_motion_out = {}
        new_motion_out['pose_quat_global'] = pose_quat_global
        new_motion_out['pose_quat'] = pose_quat
        new_motion_out['trans_orig'] = root_trans
        new_motion_out['root_trans_offset'] = root_trans_offset
        new_motion_out['beta'] = beta
        new_motion_out['gender'] = gender
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['fps'] = fps
        amass_full_motion_dict[key_name_dump] = new_motion_out

    print(f"\nSaving output to: {args.output}")
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    joblib.dump(amass_full_motion_dict, args.output)
    print(f"Successfully saved to {args.output}")
    print(f"Motion FPS: 30.0 (interpolated from 20 FPS)")

if __name__ == "__main__":
    main()