import numpy as np
import pyceres
import cv2
from typing import Dict, List, Tuple

def prepare_ba_data(pts3d, R, t, res_imgs, kpts, K):
    cam_params = {}
    pts = []
    p2d_meas = []
    
    for cam_idx in res_imgs:
        if cam_idx not in R or cam_idx not in t:
            continue
            
        R_mat = R[cam_idx]
        t_vec = t[cam_idx]
        
        theta = np.arccos((np.trace(R_mat) - 1) / 2)
        if theta > 0:
            omega = theta / (2 * np.sin(theta)) * np.array([
                R_mat[2, 1] - R_mat[1, 2],
                R_mat[0, 2] - R_mat[2, 0],
                R_mat[1, 0] - R_mat[0, 1]
            ])
        else:
            omega = np.zeros(3)
        
        cam_params[cam_idx] = (omega, t_vec.flatten())
    
    for pt_idx, pt3d in enumerate(pts3d):
        pts.append(pt3d.point3d.flatten())
        
        for cam_idx, kpt_idx in pt3d.source_2dpt_idxs.items():
            if cam_idx in cam_params:
                kpt = kpts[cam_idx][kpt_idx]
                p2d_meas.append((pt_idx, cam_idx, kpt.pt[0], kpt.pt[1]))
    
    return cam_params, pts, p2d_meas

def optimize_bundle_adjustment(pts3d, R, t, res_imgs, kpts, K, ftol=1e-3):
    print(f"\nStarting bundle adjustment with {len(pts3d)} points and {len(res_imgs)} cameras...")
    
    cam_params, pts, p2d_meas = prepare_ba_data(pts3d, R, t, res_imgs, kpts, K)
    
    if not cam_params:
        print("No valid cameras found for optimization")
        return pts3d, R, t
    
    print(f"Number of cameras: {len(cam_params)}")
    print(f"Number of points: {len(pts)}")
    print(f"Number of measurements: {len(p2d_meas)}")
    
    try:
        prob = pyceres.Problem()
        
        for pt_idx, cam_idx, x_obs, y_obs in p2d_meas:
            cost_fn = ReprojectionError(
                x_obs, y_obs,
                focal_x=K[0, 0],
                focal_y=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2]
            )
            
            omega, t_vec = cam_params[cam_idx]
            
            prob.AddResidualBlock(
                cost_fn,
                None,
                omega,
                t_vec,
                pts[pt_idx]
            )
        
        opts = pyceres.SolverOptions()
        opts.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        opts.minimizer_progress_to_stdout = True
        opts.function_tolerance = ftol
        opts.max_num_iterations = 200
        
        summary = pyceres.Solve(opts, prob)
        print(summary.BriefReport())
        
        for cam_idx, (omega, t_vec) in cam_params.items():
            theta = np.linalg.norm(omega)
            if theta > 0:
                omega_norm = omega / theta
                omega_hat = np.array([
                    [0, -omega_norm[2], omega_norm[1]],
                    [omega_norm[2], 0, -omega_norm[0]],
                    [-omega_norm[1], omega_norm[0], 0]
                ])
                R_mat = np.eye(3) + np.sin(theta) * omega_hat + (1 - np.cos(theta)) * (omega_hat @ omega_hat)
            else:
                R_mat = np.eye(3)
            
            R[cam_idx] = R_mat
            t[cam_idx] = t_vec.reshape(3, 1)
        
        for i, pt3d in enumerate(pts3d):
            pt3d.point3d = pts[i].reshape(1, 3)
        
        print(f"\nOptimization complete:")
        print(f"Optimized {len(cam_params)} camera poses")
        print(f"Optimized {len(pts)} points")
        
    except Exception as e:
        print(f"Warning: Bundle adjustment failed with error: {str(e)}")
        print("Continuing with original points and poses...")
    
    return pts3d, R, t

class ReprojectionError:
    def __init__(self, x_obs, y_obs, focal_x, focal_y, cx, cy):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.fx = focal_x
        self.fy = focal_y
        self.cx = cx
        self.cy = cy
    
    def __call__(self, cam_rot, cam_trans, pt):
        theta = np.linalg.norm(cam_rot)
        if theta > 0:
            omega = cam_rot / theta
            omega_hat = np.array([
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0]
            ])
            R = np.eye(3) + np.sin(theta) * omega_hat + (1 - np.cos(theta)) * (omega_hat @ omega_hat)
        else:
            R = np.eye(3)
        
        p_cam = R @ pt + cam_trans
        
        if p_cam[2] <= 0:
            return (self.x_obs, self.y_obs)
        
        x_proj = self.fx * p_cam[0] / p_cam[2] + self.cx
        y_proj = self.fy * p_cam[1] / p_cam[2] + self.cy
        
        return (x_proj - self.x_obs, y_proj - self.y_obs)

def do_BA(pts3d, R, t, res_imgs, kpts, K, ftol=1e-3):
    return optimize_bundle_adjustment(pts3d, R, t, res_imgs, kpts, K, ftol)