import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
import pdb
import imageio
import cv2
cv2.setNumThreads(0)

from scipy.interpolate import CubicSpline

class DocDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a DocInstanceDataset."""

    def __init__(self,
                train_cameras,
                data_root,
                data_dir,
                img_res,
                tex_res,
                scan_id=0,
                cam_file=None,
                imp_map=False
                ):

        # self.instance_dir = os.path.join('/media/hilab/HiLabData/Sagnik/idr/input/pami/', data_dir)
        # self.instance_dir = os.path.join('/media/hilab/HiLabData/Sagnik/idr/input/real/', data_dir)
        self.instance_dir = os.path.join(data_root, data_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.tex_res=tex_res
        self.imp_map_flag=imp_map

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.train_cameras = train_cameras

        self.image_dir = os.path.join(self.instance_dir,'images')
        self.mask_dir = os.path.join(self.instance_dir,'masks')

        self._coord_trans_world = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.cam_file=os.path.join(self.instance_dir,'poses_bounds.npy')
        
        poses ,bds, imgs, hrimgs, msks=self._load_data(
            width=img_res[1], height=img_res[0])
        
        self.n_images=poses.shape[0]
        # pdb.set_trace()
        self.intrinsics_all = []
        self.pose_all = []
        self.object_masks=[]
        self.imp_map=[]
        for i in range(self.n_images):
            pose = np.eye(4, dtype=np.float32)
            K = np.eye(3, dtype=np.float32)
            pose[:3,:]=poses[i,:,:4]
            # pose[:3,:3]=pose[:3,:3].transpose()
            pose = np.matmul(self._coord_trans_world,pose) 
            # pose = np.linalg.inv(pose)
            K[0,0]=poses[i,2,4]
            K[1,1]=poses[i,2,4]
            # pdb.set_trace()
            K[0,2]=poses[i,1,4]/2.0
            K[1,2]=poses[i,0,4]/2.0
            self.intrinsics_all.append(torch.from_numpy(K).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            
            # get importance sampling + preprocess masks
            msk = msks[i].reshape(-1)
            self.object_masks.append(torch.from_numpy(msk).bool())
            if self.imp_map_flag:
                impmap=self.get_impmap(1-msks[i])
                self.imp_map.append(torch.from_numpy(impmap))
            else:
                self.imp_map.append(torch.ones(msk.shape))
            
        # pdb.set_trace()
        self.rgb_images = imgs
        self.rgb_hrimages = hrimgs
        # self.object_masks = msks
        texpath=os.path.join(self.instance_dir,'GT.PNG')
        tex = rend_util.load_rgb(texpath, resize=self.tex_res)
        self.tex=torch.from_numpy(tex).float()
        
    def get_impmap(self,object_mask):
        kernel = np.ones((15,15),np.uint8)
        erosion = cv2.erode(object_mask.copy().astype(np.uint8),kernel,iterations = 3)
        dist=1-erosion
        dist = cv2.distanceTransform(dist.astype(np.uint8), cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist=1-dist
        # pdb.set_trace()
        dist[dist<=0.8]=0.3
        dist[(dist>0.8) & (dist<1)]=10.0
        dist[dist==1]=0.3
        # dist=dist/dist.sum()
        dist=dist.reshape(-1)
        return dist
    
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        uv_pts=torch.zeros_like(uv)
        uv_pts[:,0]=uv[:,0]/(self.img_res[1]-1)
        uv_pts[:,1]=uv[:,1]/(self.img_res[0]-1)
        # pdb.set_trace()

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "uv_inp": uv_pts,
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "rgbhr": self.rgb_hrimages[idx],
            "tex": self.tex,
            "uv_pts": uv_pts,
            "imp_map": self.imp_map[idx],
        }

        # pdb.set_trace()
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]
            sample['uv_inp']=uv_pts[self.sampling_idx, :]
            ground_truth["uv_pts"]= uv_pts[self.sampling_idx, :]
            ground_truth["imp_map"]=self.imp_map[idx][self.sampling_idx]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))
        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get colmap pose 
        # pdb.set_trace()
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in self.pose_all], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

    def _minify(self, factors=[], resolutions=[]):
        # pdb.set_trace()
        needtoload = False
        for r in factors:
            imgdir = os.path.join(self.instance_dir, "images_{}".format(r))
            mskdir = os.path.join(self.instance_dir, "masks_{}".format(r))
            if not os.path.exists(imgdir):
                needtoload = True
        for r in resolutions:
            imgdir = os.path.join(self.instance_dir, "images_{}x{}".format(r[1], r[0]))
            mskdir = os.path.join(self.instance_dir, "masks_{}x{}".format(r[1], r[0]))
            if not os.path.exists(imgdir):
                needtoload = True
        if not needtoload:
            return

        from subprocess import check_output

        imgdir = self.image_dir
        imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
        imgs = [
            f
            for f in imgs
            if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
        ]
        imgdir_orig = imgdir
        
        mskdir = self.mask_dir
        msks = [os.path.join(mskdir, f) for f in sorted(os.listdir(mskdir))]
        msks = [
            f
            for f in msks
            if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
        ]
        mskdir_orig = mskdir

        wd = os.getcwd()

        for r in factors + resolutions:
            if isinstance(r, int):
                name = "images_{}".format(r)
                imgresizearg = "{}%".format(100.0 / r)
            else:
                name = "images_{}x{}".format(r[1], r[0])
                imgresizearg = "{}x{}".format(r[1], r[0])
            imgdir = os.path.join(self.instance_dir, name)
            
            if os.path.exists(imgdir):
                continue
            
            if isinstance(r, int):
                name = "masks_{}".format(r)
                imgresizearg = "{}%".format(100.0 / r)
            else:
                name = "masks_{}x{}".format(r[1], r[0])
                imgresizearg = "{}x{}".format(r[1], r[0])
            mskdir = os.path.join(self.instance_dir, name)
            
            if os.path.exists(mskdir):
                continue

            print("Minifying", r, self.instance_dir)

            os.makedirs(imgdir)
            os.makedirs(mskdir)
            
            check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)
            check_output("cp {}/* {}".format(mskdir_orig, mskdir), shell=True)

            # processing images
            ext = imgs[0].split(".")[-1]
            args = " ".join(
                ["mogrify", "-resize", imgresizearg, "-format", "png", "*.{}".format(ext)]
            )
            print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)

            if ext != "png":
                check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
                print("Removed image duplicates")
                
            # process masks
            ext = msks[0].split(".")[-1]
            args = " ".join(
                ["mogrify", "-resize", mskresizearg, "-format", "png", "*.{}".format(ext)]
            )
            print(args)
            os.chdir(mskdir)
            check_output(args, shell=True)
            os.chdir(wd)

            if ext != "png":
                check_output("rm {}/*.{}".format(mskdir, ext), shell=True)
                print("Removed mask duplicates")
            print("Done")


    def _load_data(self, factor=None, width=None, height=None, load_imgs=True):

        poses_arr = np.load(os.path.join(self.instance_dir, "poses_bounds.npy"))
        # pdb.set_trace()
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])

        img0 = [
            os.path.join(self.instance_dir, "images", f)
            for f in sorted(os.listdir(self.image_dir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ][0]
        sh = imageio.imread(img0).shape

        sfx = ""

        if factor is not None:
            sfx = "_{}".format(factor)
            self._minify(factors=[factor])
            factor = factor
        elif height is not None:
            factor = sh[0] / float(height)
            width = int(round(sh[1] / factor))
            # pdb.set_trace()
            self._minify(resolutions=[[height, width]])
            sfx = "_{}x{}".format(width, height)
        elif width is not None:
            factor = sh[1] / float(width)
            height = int(round(sh[0] / factor))
            self._minify(resolutions=[[height, width]])
            sfx = "_{}x{}".format(width, height)
        else:
            factor = 1

        imgdir = os.path.join(self.instance_dir, "images" + sfx)
        hrimgdir = os.path.join(self.instance_dir, "images")
        if not os.path.exists(imgdir):
            print(imgdir, "does not exist, returning")
            return

        imgfiles = [
            os.path.join(imgdir, f)
            for f in sorted(os.listdir(imgdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]

        hrimgfiles = [
            os.path.join(hrimgdir, f)
            for f in sorted(os.listdir(hrimgdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]
        
        mskdir = os.path.join(self.instance_dir, "masks" + sfx)
        if not os.path.exists(mskdir):
            print(mskdir, "does not exist, returning")
            return

        mskfiles = [
            os.path.join(mskdir, f)
            for f in sorted(os.listdir(mskdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]
        if poses.shape[-1] != len(imgfiles):
            print(
                "Mismatch between imgs {} and poses {} !!!!".format(
                    len(imgfiles), poses.shape[-1]
                )
            )
            return
        
        # pdb.set_trace()
        sh = imageio.imread(imgfiles[0]).shape
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:, :]], 1)
        # pdb.set_trace()
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        # #recenter
        poses = recenter_poses(poses)
        #spherify
        poses, _ , bds = spherify_poses(poses, bds)
        
        if not load_imgs:
            return poses, bds
        
        imgs=[]
        hrimgs=[]
        msks=[]
        for rgbpath in imgfiles:
            rgb = rend_util.load_rgb(rgbpath)
            # pdb.set_trace()
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            imgs.append(torch.from_numpy(rgb).float())

        for rgbpath in hrimgfiles:
            rgb = rend_util.load_rgb(rgbpath)
            # pdb.set_trace()
            # rgb = rgb.reshape(3, -1).transpose(1, 0)
            hrimgs.append(torch.from_numpy(rgb).float())
            
        for mskpath in mskfiles:
            msk = rend_util.load_mask(mskpath)
            # msk = msk.reshape(-1)
            # msks.append(torch.from_numpy(msk).bool())
            msks.append(msk)
            
        print("Loaded image and poses data")
        return poses, bds, imgs, hrimgs, msks
    


class DocDatasetEval(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a DocInstanceDataset."""

    def __init__(self,
                train_cameras,
                data_dir,
                img_res,
                tex_res,
                scan_id=0,
                cam_file=None,
                mode='dtu', # sample camera pose based on DTU
                imp_map=False
                ):

        # self.instance_dir = os.path.join('/media/hilab/HiLabData/Sagnik/idr/input/pami/', data_dir)
        self.instance_dir = os.path.join('/media/hilab/HiLabData/Sagnik/idr/input/real/', data_dir)
        # self.instance_dir = os.path.join('/media/hilab/HiLabData/Sagnik/idr/input/ocr_set/data', data_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.tex_res=tex_res
        self.imp_map_flag=imp_map

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        self.image_dir = os.path.join(self.instance_dir,'images')
        self.mask_dir = os.path.join(self.instance_dir,'masks')

        self._coord_trans_world = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        self.cam_file=os.path.join(self.instance_dir,'poses_bounds.npy')
        
        poses ,bds, imgs, hrimgs, msks=self._load_data(
            width=img_res[1], height=img_res[0])
        
        if mode=='dtu':
            t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
            pose_quat = torch.tensor(
                [
                    [0.9698, 0.2121, 0.1203, -0.0039],
                    [0.7020, 0.1578, 0.4525, 0.5268],
                    [0.6766, 0.3176, 0.5179, 0.4161],
                    [0.9085, 0.4020, 0.1139, -0.0025],
                    [0.9698, 0.2121, 0.1203, -0.0039],
                ]
            )
            n_inter = 90 // 5
            num_views = n_inter * 5
            t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
            # scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)
            scales = np.array([0.5, 0.5, 0.5, 0.5, 0.5]).astype(np.float32)

            s_new = CubicSpline(t_in, scales, bc_type="periodic")
            s_new = s_new(t_out)

            q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
            q_new = q_new(t_out)
            q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
            q_new = torch.from_numpy(q_new).float()
            
            self.intrinsics_all=[]
            self.render_poses = []
            self.object_masks=[]
            for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
                new_q = new_q.unsqueeze(0)
                R = rend_util.quat_to_rot(new_q)
                t = R[:, :, 2] * scale
                new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
                new_pose[:, :3, :3] = R
                new_pose[:, :3, 3] = t
                # new_pose = torch.inverse(new_pose)
                self.render_poses.append(new_pose)
                K = np.eye(3, dtype=np.float32)
                K[0,0]=poses[0,2,4]
                K[1,1]=poses[0,2,4]
                K[0,2]=poses[0,1,4]/2.0
                K[1,2]=poses[0,0,4]/2.0
                self.intrinsics_all.append(torch.from_numpy(K).float())
                msk = msks[0].reshape(-1)
                # msk = np.ones(msk.shape)
                self.object_masks.append(torch.from_numpy(msk).bool())
            self.render_poses = torch.cat(self.render_poses, dim=0)
            self.n_images=self.render_poses.shape[0]
            
        else:
            radius = 1
            # pdb.set_trace()
            # Use 360 pose sequence from NeRF
            # self.render_poses = torch.stack(
            #     [
            #         rend_util.pose_spherical(angle, 100, radius)
            #         for angle in np.linspace(-180, 180, 100 + 1)[:-1]
            #     ],
            #     0,
            # )  # (NV, 4, 4)
            self.render_poses = torch.stack( # left to right
                [
                    rend_util.pose_spherical(-60, elev, radius)
                    for elev in np.linspace(60, 120, 100 + 1)[:-1]
                ],
                0,
            )
            self.render_poses = torch.stack( # top to down
                [
                    rend_util.pose_spherical(30, elev, radius)
                    for elev in np.linspace(60, 120, 100 + 1)[:-1]
                ],
                0,
            )

            self.n_images=self.render_poses.shape[0]
            self.intrinsics_all=[]
            self.object_masks=[]
            
            for i in range(self.n_images):
                K = np.eye(3, dtype=np.float32)
                K[0,0]=poses[0,2,4]
                K[1,1]=poses[0,2,4]
                K[0,2]=poses[0,1,4]/2.0
                K[1,2]=poses[0,0,4]/2.0
                self.intrinsics_all.append(torch.from_numpy(K).float())
                msk = msks[0].reshape(-1)
                msk = np.ones(msk.shape)
                self.object_masks.append(torch.from_numpy(msk).bool())
        
        
        # self.object_masks = msks
        texpath=os.path.join(self.instance_dir,'GT.PNG')
        tex = rend_util.load_rgb(texpath, resize=self.tex_res)
        self.tex=torch.from_numpy(tex).float()
    
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        # uv_pts=torch.zeros_like(uv)
        # uv_pts[:,0]=uv[:,0]/(self.img_res[1]-1)
        # uv_pts[:,1]=uv[:,1]/(self.img_res[0]-1)
        # pdb.set_trace()

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose":self.render_poses[idx]
        }

        return idx, sample

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))
        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get colmap pose 
        # pdb.set_trace()
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in self.pose_all], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

    def _minify(self, factors=[], resolutions=[]):
        # pdb.set_trace()
        needtoload = False
        for r in factors:
            imgdir = os.path.join(self.instance_dir, "images_{}".format(r))
            mskdir = os.path.join(self.instance_dir, "masks_{}".format(r))
            if not os.path.exists(imgdir):
                needtoload = True
        for r in resolutions:
            imgdir = os.path.join(self.instance_dir, "images_{}x{}".format(r[1], r[0]))
            mskdir = os.path.join(self.instance_dir, "masks_{}x{}".format(r[1], r[0]))
            if not os.path.exists(imgdir):
                needtoload = True
        if not needtoload:
            return

        from subprocess import check_output

        imgdir = self.image_dir
        imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
        imgs = [
            f
            for f in imgs
            if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
        ]
        imgdir_orig = imgdir
        
        mskdir = self.mask_dir
        msks = [os.path.join(mskdir, f) for f in sorted(os.listdir(mskdir))]
        msks = [
            f
            for f in msks
            if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
        ]
        mskdir_orig = mskdir

        wd = os.getcwd()

        for r in factors + resolutions:
            if isinstance(r, int):
                name = "images_{}".format(r)
                imgresizearg = "{}%".format(100.0 / r)
            else:
                name = "images_{}x{}".format(r[1], r[0])
                imgresizearg = "{}x{}".format(r[1], r[0])
            imgdir = os.path.join(self.instance_dir, name)
            
            if os.path.exists(imgdir):
                continue
            
            if isinstance(r, int):
                name = "masks_{}".format(r)
                imgresizearg = "{}%".format(100.0 / r)
            else:
                name = "masks_{}x{}".format(r[1], r[0])
                imgresizearg = "{}x{}".format(r[1], r[0])
            mskdir = os.path.join(self.instance_dir, name)
            
            if os.path.exists(mskdir):
                continue

            print("Minifying", r, self.instance_dir)

            os.makedirs(imgdir)
            os.makedirs(mskdir)
            
            check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)
            check_output("cp {}/* {}".format(mskdir_orig, mskdir), shell=True)

            # processing images
            ext = imgs[0].split(".")[-1]
            args = " ".join(
                ["mogrify", "-resize", imgresizearg, "-format", "png", "*.{}".format(ext)]
            )
            print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)

            if ext != "png":
                check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
                print("Removed image duplicates")
                
            # process masks
            ext = msks[0].split(".")[-1]
            args = " ".join(
                ["mogrify", "-resize", mskresizearg, "-format", "png", "*.{}".format(ext)]
            )
            print(args)
            os.chdir(mskdir)
            check_output(args, shell=True)
            os.chdir(wd)

            if ext != "png":
                check_output("rm {}/*.{}".format(mskdir, ext), shell=True)
                print("Removed mask duplicates")
            print("Done")


    def _load_data(self, factor=None, width=None, height=None, load_imgs=True):

        poses_arr = np.load(os.path.join(self.instance_dir, "poses_bounds.npy"))
        # pdb.set_trace()
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])

        img0 = [
            os.path.join(self.instance_dir, "images", f)
            for f in sorted(os.listdir(self.image_dir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ][0]
        sh = imageio.imread(img0).shape

        sfx = ""

        if factor is not None:
            sfx = "_{}".format(factor)
            self._minify(factors=[factor])
            factor = factor
        elif height is not None:
            factor = sh[0] / float(height)
            width = int(round(sh[1] / factor))
            # pdb.set_trace()
            self._minify(resolutions=[[height, width]])
            sfx = "_{}x{}".format(width, height)
        elif width is not None:
            factor = sh[1] / float(width)
            height = int(round(sh[0] / factor))
            self._minify(resolutions=[[height, width]])
            sfx = "_{}x{}".format(width, height)
        else:
            factor = 1

        imgdir = os.path.join(self.instance_dir, "images" + sfx)
        hrimgdir = os.path.join(self.instance_dir, "images")
        if not os.path.exists(imgdir):
            print(imgdir, "does not exist, returning")
            return

        imgfiles = [
            os.path.join(imgdir, f)
            for f in sorted(os.listdir(imgdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]

        hrimgfiles = [
            os.path.join(hrimgdir, f)
            for f in sorted(os.listdir(hrimgdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]
        
        mskdir = os.path.join(self.instance_dir, "masks" + sfx)
        if not os.path.exists(mskdir):
            print(mskdir, "does not exist, returning")
            return

        mskfiles = [
            os.path.join(mskdir, f)
            for f in sorted(os.listdir(mskdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]
        if poses.shape[-1] != len(imgfiles):
            print(
                "Mismatch between imgs {} and poses {} !!!!".format(
                    len(imgfiles), poses.shape[-1]
                )
            )
            return
        
        # pdb.set_trace()
        sh = imageio.imread(imgfiles[0]).shape
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:, :]], 1)
        # pdb.set_trace()
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        # #recenter
        poses = recenter_poses(poses)
        #spherify
        poses, _ , bds = spherify_poses(poses, bds)
        
        if not load_imgs:
            return poses, bds
        
        imgs=[]
        hrimgs=[]
        msks=[]
        for rgbpath in imgfiles:
            rgb = rend_util.load_rgb(rgbpath)
            # pdb.set_trace()
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            imgs.append(torch.from_numpy(rgb).float())

        for rgbpath in hrimgfiles:
            rgb = rend_util.load_rgb(rgbpath)
            # pdb.set_trace()
            # rgb = rgb.reshape(3, -1).transpose(1, 0)
            hrimgs.append(torch.from_numpy(rgb).float())
            
        for mskpath in mskfiles:
            msk = rend_util.load_mask(mskpath)
            # msk = msk.reshape(-1)
            # msks.append(torch.from_numpy(msk).bool())
            msks.append(msk)
            
        print("Loaded image and poses data")
        return poses, bds, imgs, hrimgs, msks





def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(x):
    return x / np.linalg.norm(x)

def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

def spherify_poses(poses, bds):
    def add_row_to_homogenize_transform(p):
        r"""Add the last row to homogenize 3 x 4 transformation matrices."""
        return np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
        )

    # p34_to_44 = lambda p: np.concatenate(
    #     [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    # )

    p34_to_44 = add_row_to_homogenize_transform

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds