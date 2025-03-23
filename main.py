import cv2
import numpy as np
import open3d as o3d
import matplotlib as mpl
import ceres_bundle_adjustment as b
import matching as m
import reconstruction as r

class SfMReconstructor:
    def __init__(self, n_imgs, img_path, img_ext, K, ba_chk=None):
        self.n_imgs = n_imgs
        self.img_path = img_path
        self.img_ext = img_ext
        self.ba_chk = ba_chk or [3,4,5,6] + [int(6*(1.34**i)) for i in range(25)]
        
        self.imgs = None
        self.col_imgs = None
        self.kpts = None
        self.desc = None
        self.K = K
        self.mtch = None
        self.adj = None
        self.pair_lst = None
        
        self.pts3d = None
        self.R = {}
        self.t = {}
        self.res_imgs = []
        self.unres_imgs = []
        self.bp = None
        
        mpl.rcParams['figure.dpi'] = 200
    
    def load_and_match_features(self):
        print(f"\nLoading and matching features for {self.n_imgs} images from {self.img_path} dataset...")
        
        self.imgs, self.col_imgs, self.kpts, self.desc, self.K = m.find_features(self.n_imgs, 
                                                                                 img_path=self.img_path, 
                                                                                 img_ext=self.img_ext,
                                                                                 K=self.K)
        
        matcher = cv2.BFMatcher(cv2.NORM_L1)
        self.mtch = m.find_matches(matcher, self.kpts, self.desc)
        print('Number of matches before outlier removal:', m.num_matches(self.mtch))
        m.print_num_img_pairs(self.mtch)
        
        self.mtch = m.remove_outliers(self.mtch, self.kpts)
        print("\nAfter outlier removal:")
        m.print_num_img_pairs(self.mtch)
        
        self.adj, self.pair_lst = m.create_img_adjacency_matrix(self.n_imgs, self.mtch)
    
    def initialize_reconstruction(self):
        print("\nInitializing reconstruction...")
        
        self.bp = r.best_img_pair(self.adj, self.mtch, self.kpts, self.K, top_x_perc=0.2)
        R0, t0, R1, t1, self.pts3d = r.initialize_reconstruction(self.kpts, self.mtch, self.K, self.bp[0], self.bp[1])
        
        self.R = {self.bp[0]: R0, self.bp[1]: R1}
        self.t = {self.bp[0]: t0, self.bp[1]: t1}
        
        self.res_imgs = [self.bp[0], self.bp[1]]
        self.unres_imgs = [i for i in range(len(self.imgs)) if i not in self.res_imgs]
        print('Initial image pair:', self.res_imgs)
    
    def calculate_reprojection_errors(self):
        av = 0
        valid = 0
        
        for im in self.res_imgs:
            if im in self.R and im in self.t:
                try:
                    p3d, p2d, err, errs = r.get_reproj_errors(im, self.pts3d, self.R[im], self.t[im], self.K, self.kpts)
                    print(f'Average reprojection error on image {im} is {err} pixels')
                    av += err
                    valid += 1
                except Exception as e:
                    print(f"Warning: Could not calculate reprojection error for image {im}: {str(e)}")
        
        if valid > 0:
            av = av / valid
            print(f'Average reprojection error across {valid} resected images is {av} pixels')
        else:
            print("Warning: Could not calculate average reprojection error")
        
        return av, valid
    
    def grow_reconstruction(self):
        print("\nGrowing reconstruction...")
        
        while self.unres_imgs:
            res_idx, unres_idx, pre = r.next_img_pair_to_grow_reconstruction(self.n_imgs, self.bp, self.res_imgs, self.unres_imgs, self.adj)
            
            self.pts3d, p3d_pnp, p2d_pnp, tri_stat = r.get_correspondences_for_pnp(res_idx, unres_idx, self.pts3d, self.mtch, self.kpts)
            
            if len(p3d_pnp) < 12:
                print(f"{len(p3d_pnp)} is too few correspondences for PnP. Skipping imgs resected:{res_idx} and unresected:{unres_idx}")
                continue
            
            print(f"Unresected image: {unres_idx}, resected: {res_idx}")
            R_new, t_new = r.do_pnp(p3d_pnp, p2d_pnp, self.K)
            
            self.R[unres_idx] = R_new
            self.t[unres_idx] = t_new
            
            if pre:
                self.res_imgs.insert(0, unres_idx)
            else:
                self.res_imgs.append(unres_idx)
            self.unres_imgs.remove(unres_idx)
            
            pnp_errs, proj_pts, avg_err, perc_in = r.test_reproj_pnp_points(p3d_pnp, p2d_pnp, R_new, t_new, self.K)
            print(f"Average error of reprojecting points used to resect image {unres_idx} back onto it is: {avg_err}")
            print(f"Fraction of PnP inliers: {perc_in} num pts used in PnP: {len(pnp_errs)}")
            
            self._triangulate_new_points(res_idx, unres_idx, tri_stat)
            
            self._check_and_run_bundle_adjustment(perc_in, avg_tri_err_l=0, avg_tri_err_r=0)
            
            self.calculate_reprojection_errors()
    
    def _triangulate_new_points(self, res_idx, unres_idx, tri_stat):
        if res_idx < unres_idx:
            k1, k2, k1_idx, k2_idx = r.get_aligned_kpts(res_idx, unres_idx, self.kpts, self.mtch, mask=tri_stat)
            if np.sum(tri_stat) > 0:
                self.pts3d, tri_errs, err_l, err_r = r.triangulate_points_and_reproject(
                    self.R[res_idx], self.t[res_idx],
                    self.R[unres_idx], self.t[unres_idx],
                    self.K, self.pts3d, res_idx, unres_idx,
                    k1, k2, k1_idx, k2_idx,
                    color_images=self.col_imgs, reproject=True)
        else:
            k1, k2, k1_idx, k2_idx = r.get_aligned_kpts(unres_idx, res_idx, self.kpts, self.mtch, mask=tri_stat)
            if np.sum(tri_stat) > 0:
                self.pts3d, tri_errs, err_l, err_r = r.triangulate_points_and_reproject(
                    self.R[unres_idx], self.t[unres_idx],
                    self.R[res_idx], self.t[res_idx],
                    self.K, self.pts3d, unres_idx, res_idx,
                    k1, k2, k1_idx, k2_idx,
                    color_images=self.col_imgs, reproject=True)
    
    def _check_and_run_bundle_adjustment(self, perc_in, avg_tri_err_l, avg_tri_err_r):
        if 0.8 < perc_in < 0.95 or 5 < avg_tri_err_l < 10 or 5 < avg_tri_err_r < 10:
            self.pts3d, self.R, self.t = b.do_BA(self.pts3d, self.R, self.t, self.res_imgs, self.kpts, self.K, ftol=1e0)
        
        if (len(self.res_imgs) in self.ba_chk or not self.unres_imgs or perc_in <= 0.8 or avg_tri_err_l >= 10 or avg_tri_err_r >= 10):
            self.pts3d, self.R, self.t = b.do_BA(self.pts3d, self.R, self.t, self.res_imgs, self.kpts, self.K, ftol=1e-1)
    
    def create_point_cloud(self):
        print("\nCreating point cloud...")
        
        pts = []
        cols = []
        for pt in self.pts3d:
            pts.append(pt.point3d[0])
            if pt.color is not None:
                cols.append([c/255.0 for c in pt.color])
            else:
                cols.append([0.5, 0.5, 0.5])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        
        return pcd
    
    def remove_outliers(self, pcd, nb=50, std=2.0):
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
        return pcd.select_by_index(ind)
    
    def run_reconstruction(self):
        self.load_and_match_features()
        self.initialize_reconstruction()
        self.grow_reconstruction()
        
        pcd = self.create_point_cloud()
        pcd_clean = self.remove_outliers(pcd)
        
        o3d.io.write_point_cloud("reconstruction_colored.ply", pcd_clean)
        o3d.visualization.draw_geometries([pcd_clean])
        
        return pcd_clean

def main():
    img_path = "./datasets/templeRing/"
    K = np.matrix('1520.40 0.00 302.32; 0.00 1525.90 246.87; 0.00 0.00 1.00')

    rec = SfMReconstructor(n_imgs=46, img_path=img_path, 
                           img_ext=".png", K=K)
    pcd = rec.run_reconstruction()
    
    print("\nReconstruction complete!")
    print(f"Final number of points: {len(pcd.points)}")
    print(f"Final number of cameras: {len(rec.res_imgs)}")

if __name__ == "__main__":
    main()