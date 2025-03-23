import numpy as np
import cv2
from numba import jit
import random


class Point3D_with_views:
    def __init__(self, p3d, src_2d_idx, col=None):
        self.point3d = p3d
        self.source_2dpt_idxs = src_2d_idx
        self.color = col

@jit(nopython=True)
def calculate_reproj_errors_fast(proj_pts, pts_2d):
    n = len(proj_pts)
    delta = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        delta[i,0] = abs(proj_pts[i,0] - pts_2d[i,0])
        delta[i,1] = abs(proj_pts[i,1] - pts_2d[i,1])
    
    avg_delta = np.mean(delta)
    return avg_delta, delta

@jit(nopython=True)
def test_reproj_pnp_points_fast(pts3d, pts2d, R_new, t_new, K, rep_thresh):
    n = len(pts3d)
    errs = np.zeros((n, 2), dtype=np.float64)
    inl = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        Xw = np.zeros((3, 1), dtype=np.float64)
        Xw[0,0] = pts3d[i,0]
        Xw[1,0] = pts3d[i,1]
        Xw[2,0] = pts3d[i,2]
        
        Xr = np.dot(R_new, Xw)
        Xc = np.zeros((3, 1), dtype=np.float64)
        Xc[0,0] = Xr[0,0] + t_new[0,0]
        Xc[1,0] = Xr[1,0] + t_new[1,0]
        Xc[2,0] = Xr[2,0] + t_new[2,0]
        
        x = np.dot(K, Xc)
        x_norm = x[2,0]
        if abs(x_norm) > 1e-10:
            x_proj = np.zeros(2, dtype=np.float64)
            x_proj[0] = x[0,0] / x_norm
            x_proj[1] = x[1,0] / x_norm
            
            errs[i,0] = x_proj[0] - pts2d[i,0]
            errs[i,1] = x_proj[1] - pts2d[i,1]
            
            if abs(errs[i,0]) <= rep_thresh and abs(errs[i,1]) <= rep_thresh:
                inl[i] = 1
    
    avg_err = np.mean(np.abs(errs))
    perc_inl = np.sum(inl) / float(n)
    
    return errs, inl, avg_err, perc_inl

def calculate_reproj_errors(proj_pts, pts_2d):
    proj_pts = np.array(proj_pts)
    pts_2d = np.array(pts_2d)
    return calculate_reproj_errors_fast(proj_pts, pts_2d)

def test_reproj_pnp_points(p3d_pnp, p2d_pnp, R_new, t_new, K, rep_thresh=5):
    pts3d = np.array([p[0] for p in p3d_pnp])
    pts2d = np.array(p2d_pnp)
    errs, inl, avg_err, perc_inl = test_reproj_pnp_points_fast(pts3d, pts2d, R_new, t_new, K, rep_thresh)
    
    errs_list = errs.tolist()
    proj_pts = []
    return errs_list, proj_pts, avg_err, perc_inl

@jit(nopython=True)
def triangulate_points_fast(P_l, P_r, kpts_i, kpts_j):
    n_pts = kpts_i.shape[1]
    pts_4d = np.zeros((4, n_pts), dtype=np.float64)
    
    for i in range(n_pts):
        A = np.zeros((4, 4), dtype=np.float64)
        A[0] = kpts_i[1,i] * P_l[2] - P_l[1]
        A[1] = P_l[0] - kpts_i[0,i] * P_l[2]
        A[2] = kpts_j[1,i] * P_r[2] - P_r[1]
        A[3] = P_r[0] - kpts_j[0,i] * P_r[2]
        
        AtA = np.dot(A.T, A)
        evals, evecs = np.linalg.eigh(AtA)
        pts_4d[:,i] = evecs[:,0]
    
    return pts_4d

def triangulate_points_and_reproject(R_l, t_l, R_r, t_r, K, pts3d, idx1, idx2, kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs, color_images=None, reproject=True):
    print(f"Triangulating: {len(kpts_i)} points.")
    P_l = np.dot(K, np.hstack((R_l, t_l)))
    P_r = np.dot(K, np.hstack((R_r, t_r)))

    kpts_i = np.squeeze(kpts_i)
    kpts_i = kpts_i.transpose()
    kpts_i = kpts_i.reshape(2,-1)
    kpts_j = np.squeeze(kpts_j)
    kpts_j = kpts_j.transpose()
    kpts_j = kpts_j.reshape(2,-1)

    pt_4d_hom = triangulate_points_fast(P_l, P_r, kpts_i, kpts_j)
    pt_4d_hom = pt_4d_hom.T
    pts_3D = cv2.convertPointsFromHomogeneous(pt_4d_hom)

    for i in range(kpts_i.shape[1]):
        src_2d_idx = {idx1:kpts_i_idxs[i], idx2:kpts_j_idxs[i]}
        col = None
        if color_images is not None:
            x, y = int(kpts_i[0,i]), int(kpts_i[1,i])
            col = color_images[idx1][y, x].tolist()
        pt = Point3D_with_views(pts_3D[i], src_2d_idx, col)
        pts3d.append(pt)

    if reproject:
        kpts_i = kpts_i.transpose()
        kpts_j = kpts_j.transpose()
        rvec_l, _ = cv2.Rodrigues(R_l)
        rvec_r, _ = cv2.Rodrigues(R_r)
        proj_pts_l, _ = cv2.projectPoints(pts_3D, rvec_l, t_l, K, distCoeffs=np.array([]))
        proj_pts_r, _ = cv2.projectPoints(pts_3D, rvec_r, t_r, K, distCoeffs=np.array([]))
        
        avg_err_l, delta_l = calculate_reproj_errors_fast(np.squeeze(proj_pts_l), kpts_i)
        avg_err_r, delta_r = calculate_reproj_errors_fast(np.squeeze(proj_pts_r), kpts_j)
        
        print(f"Average reprojection error for just-triangulated points on image {idx1} is:", avg_err_l, "pixels.")
        print(f"Average reprojection error for just-triangulated points on image {idx2} is:", avg_err_r, "pixels.")
        errs = list(zip(delta_l.tolist(), delta_r.tolist()))
        return pts3d, errs, avg_err_l, avg_err_r

    return pts3d

def best_img_pair(adj, mtch, kpts, K, top_x_perc=0.2):
    n_mtch = []

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                n_mtch.append(len(mtch[i][j]))

    n_mtch = sorted(n_mtch, reverse=True)
    min_mtch_idx = int(len(n_mtch)*top_x_perc)
    min_mtch = n_mtch[min_mtch_idx]
    best_R = 0
    best_pair = None

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                if len(mtch[i][j]) > min_mtch:
                    k1, k2, k1_idx, k2_idx = get_aligned_kpts(i, j, kpts, mtch)
                    E, _ = cv2.findEssentialMat(k1, k2, K, cv2.FM_RANSAC, 0.999, 1.0)
                    pts, R1, t1, mask = cv2.recoverPose(E, k1, k2, K)
                    rvec, _ = cv2.Rodrigues(R1)
                    rot_angle = abs(rvec[0]) +abs(rvec[1]) + abs(rvec[2])
                    if (rot_angle > best_R or best_pair == None) and pts == len(k1):
                        best_R = rot_angle
                        best_pair = (i,j)

    return best_pair

def get_aligned_kpts(i, j, kpts, mtch, mask=None):
    if mask is None:
        mask = np.ones(len(mtch[i][j]))

    k1, k1_idx, k2, k2_idx = [], [], [], []
    for k in range(len(mtch[i][j])):
        if mask[k] == 0: continue
        k1.append(kpts[i][mtch[i][j][k].queryIdx].pt)
        k1_idx.append(mtch[i][j][k].queryIdx)
        k2.append(kpts[j][mtch[i][j][k].trainIdx].pt)
        k2_idx.append(mtch[i][j][k].trainIdx)
    k1 = np.array(k1)
    k2 = np.array(k2)
    k1 = np.expand_dims(k1, axis=1)
    k2 = np.expand_dims(k2, axis=1)

    return k1, k2, k1_idx, k2_idx

def initialize_reconstruction(kpts, mtch, K, idx1, idx2):
    k1, k2, k1_idx, k2_idx = get_aligned_kpts(idx1, idx2, kpts, mtch)
    E, _ = cv2.findEssentialMat(k1, k2, K, cv2.FM_RANSAC, 0.999, 1.0)
    pts, R1, t1, mask = cv2.recoverPose(E, k1, k2, K)
    assert abs(np.linalg.det(R1)) - 1 < 1e-7

    R0 = np.eye(3, 3)
    t0 = np.zeros((3, 1))

    pts3d = []
    pts3d = triangulate_points_and_reproject(R0, t0, R1, t1, K, pts3d, idx1, idx2, k1, k2, k1_idx, k2_idx, reproject=False)

    return R0, t0, R1, t1, pts3d

def get_idxs_in_correct_order(idx1, idx2):
    if idx1 < idx2: return idx1, idx2
    else: return idx2, idx1

def images_adjacent(i, j, adj):
    if adj[i][j] == 1 or adj[j][i] == 1:
        return True
    else:
        return False

def has_resected_pair(unres_idx, res_imgs, adj):
    for idx in res_imgs:
        if adj[unres_idx][idx] == 1 or adj[idx][unres_idx] == 1:
            return True
    return False

def has_unresected_pair(res_idx, unres_imgs, adj):
    for idx in unres_imgs:
        if adj[res_idx][idx] == 1 or adj[idx][res_idx] == 1:
            return True
    return False

def next_img_pair_to_grow_reconstruction(n_imgs, init_pair, res_imgs, unres_imgs, adj):
    if len(unres_imgs) == 0: raise ValueError('Should not check next image to resect if all have been resected already!')
    straddle = False
    if init_pair[1] - init_pair[0] > n_imgs/2 : straddle = True

    init_arc = init_pair[1] - init_pair[0] + 1

    if len(res_imgs) < init_arc:
        if straddle == False: idx = res_imgs[-2] + 1
        else: idx = res_imgs[-1] + 1
        while True:
            if idx not in res_imgs:
                prepend = True
                unres_idx = idx
                res_idx = random.choice(res_imgs)
                return res_idx, unres_idx, prepend
            idx = idx + 1 % n_imgs

    extensions = len(res_imgs) - init_arc
    if straddle == True:
        if extensions % 2 == 0:
            unres_idx = (init_pair[0] + int(extensions/2) + 1) % n_imgs
            res_idx = (unres_idx - 1) % n_imgs
        else:
            unres_idx = (init_pair[1] - int(extensions/2) - 1) % n_imgs
            res_idx = (unres_idx + 1) % n_imgs
    else:
        if extensions % 2 == 0:
            unres_idx = (init_pair[1] + int(extensions/2) + 1) % n_imgs
            res_idx = (unres_idx - 1) % n_imgs
        else:
            unres_idx = (init_pair[0] - int(extensions/2) - 1) % n_imgs
            res_idx = (unres_idx + 1) % n_imgs

    prepend = False
    return res_idx, unres_idx, prepend

def check_and_get_unresected_point(res_kpt_idx, mtch, res_idx, unres_idx):
    if res_idx < unres_idx:
        if res_kpt_idx == mtch.queryIdx:
            unres_kpt_idx = mtch.trainIdx
            success = True
            return unres_kpt_idx, success
        else:
            return None, False
    elif unres_idx < res_idx:
        if res_kpt_idx == mtch.trainIdx:
            unres_kpt_idx = mtch.queryIdx
            success = True
            return unres_kpt_idx, success
        else:
            return None, False

def get_correspondences_for_pnp(res_idx, unres_idx, pts3d, mtch, kpts):
    idx1, idx2 = get_idxs_in_correct_order(res_idx, unres_idx)
    tri_stat = np.ones(len(mtch[idx1][idx2]))
    p3d_pnp = []
    p2d_pnp = []
    for pt3d in pts3d:
        if res_idx not in pt3d.source_2dpt_idxs: continue
        res_kpt_idx = pt3d.source_2dpt_idxs[res_idx]
        for k in range(len(mtch[idx1][idx2])):
            unres_kpt_idx, success = check_and_get_unresected_point(res_kpt_idx, mtch[idx1][idx2][k], res_idx, unres_idx)
            if not success: continue
            pt3d.source_2dpt_idxs[unres_idx] = unres_kpt_idx
            p3d_pnp.append(pt3d.point3d)
            p2d_pnp.append(kpts[unres_idx][unres_kpt_idx].pt)
            tri_stat[k] = 0

    return pts3d, p3d_pnp, p2d_pnp, tri_stat

def do_pnp(p3d_pnp, p2d_pnp, K, iters=200, rep_thresh=5):
    lst_p3d = p3d_pnp
    lst_p2d = p2d_pnp
    p3d_pnp = np.squeeze(np.array(p3d_pnp))
    p2d_pnp = np.expand_dims(np.squeeze(np.array(p2d_pnp)), axis=1)
    n_pts = len(p3d_pnp)

    highest_inl = 0
    for i in range(iters):
        pt_idxs = np.random.choice(n_pts, 6, replace=False)
        pts3 = np.array([p3d_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        pts2 = np.array([p2d_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        _, rvec, tvec = cv2.solvePnP(pts3, pts2, K, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        pnp_errs, proj_pts, avg_err, perc_inl = test_reproj_pnp_points(lst_p3d, lst_p2d, R, tvec, K, rep_thresh=rep_thresh)
        if highest_inl < perc_inl:
            highest_inl = perc_inl
            best_R = R
            best_tvec = tvec
    R = best_R
    tvec = best_tvec
    print('rvec:', rvec,'\n\ntvec:', tvec)

    return R, tvec

def prep_for_reproj(img_idx, pts3d, kpts):
    pts_3d = []
    pts_2d = []
    pt3d_idxs = []
    i = 0
    for pt3d in pts3d:
        if img_idx in pt3d.source_2dpt_idxs.keys():
            pt3d_idxs.append(i)
            pts_3d.append(pt3d.point3d)
            kpt_idx = pt3d.source_2dpt_idxs[img_idx]
            pts_2d.append(kpts[img_idx][kpt_idx].pt)
        i += 1

    return np.array(pts_3d), np.array(pts_2d), pt3d_idxs

def get_reproj_errors(img_idx, pts3d, R, t, K, kpts, distCoeffs=np.array([])):
    try:
        pts_3d, pts_2d, pt3d_idxs = prep_for_reproj(img_idx, pts3d, kpts)
        if len(pts_3d) == 0:
            print(f"Warning: No 3D points found for image {img_idx}")
            return [], [], 0, []
            
        rvec, _ = cv2.Rodrigues(R)
        proj_pts, _ = cv2.projectPoints(pts_3d, rvec, t, K, distCoeffs=distCoeffs)
        proj_pts = np.squeeze(proj_pts)
        
        avg_err, errs = calculate_reproj_errors_fast(proj_pts, pts_2d)
        return pts_3d, pts_2d, avg_err, errs
    except Exception as e:
        print(f"Warning: Error calculating reprojection errors for image {img_idx}: {str(e)}")
        return [], [], 0, []
