import cv2
import numpy as np
import os

def find_features(n_imgs, img_path, img_ext, K):
    imgs = []
    col_imgs = []
    kpts = []
    desc = []
    sift = cv2.SIFT_create()
    
    for i in range(n_imgs):
        img_re = f'{img_path}/{i:02d}{img_ext}'
        # print(os.path.exists(img_re))
        img = cv2.imread(img_re, cv2.IMREAD_GRAYSCALE)
        col_img = cv2.imread(img_re, cv2.IMREAD_COLOR)
        col_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
        
        imgs.append(img)
        col_imgs.append(col_img)
        kp, des = sift.detectAndCompute(imgs[-1], None)
        kpts.append(kp)
        desc.append(des)
    return imgs, col_imgs, kpts, desc, K


def find_matches(matcher, kpts, desc, ratio=0.7):
    mtch = []
    n = len(kpts)
    for i in range(n):
        mtch.append([])
        for j in range(n):
            if j <= i: mtch[i].append(None)
            else:
                m_list = []
                m = matcher.knnMatch(desc[i], desc[j], k=2)
                for k in range(len(m)):
                    try:
                        if m[k][0].distance < ratio*m[k][1].distance:
                            m_list.append(m[k][0])
                    except:
                        continue
                mtch[i].append(m_list)
    return mtch

def remove_outliers(mtch, kpts):
    for i in range(len(mtch)):
        for j in range(len(mtch[i])):
            if j <= i: continue
            if len(mtch[i][j]) < 20:
                mtch[i][j] = []
                continue
            pts_i = []
            pts_j = []
            for k in range(len(mtch[i][j])):
                pts_i.append(kpts[i][mtch[i][j][k].queryIdx].pt)
                pts_j.append(kpts[j][mtch[i][j][k].trainIdx].pt)
            pts_i = np.int32(pts_i)
            pts_j = np.int32(pts_j)
            F, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, ransacReprojThreshold=3)
            if np.linalg.det(F) > 1e-7: raise ValueError(f"Bad F_mat between images: {i}, {j}. Determinant: {np.linalg.det(F)}")
            mtch[i][j] = np.array(mtch[i][j])
            if mask is None:
                mtch[i][j] = []
                continue
            mtch[i][j] = mtch[i][j][mask.ravel() == 1]
            mtch[i][j] = list(mtch[i][j])

            if len(mtch[i][j]) < 20:
                mtch[i][j] = []
                continue

    return mtch

def num_matches(mtch):
    n = 0
    for i in range(len(mtch)):
        for j in range(len(mtch[i])):
            if j <= i: continue
            n += len(mtch[i][j])
    return n

def print_num_img_pairs(mtch):
    n_pairs = 0
    n_total = 0
    for i in range(len(mtch)):
        for j in range(len(mtch[i])):
            if j <= i: continue
            n_total += 1
            if len(mtch[i][j]) > 0: n_pairs += 1
    print(f"Number of img pairs is {n_pairs} out of possible {n_total}")

def create_img_adjacency_matrix(n_imgs, mtch):
    n = len(mtch)
    adj = np.zeros((n, n), dtype=np.uint8)
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            if mtch[i][j]:
                adj[i,j] = 1
                pairs.append((i,j))
    
    return adj, pairs