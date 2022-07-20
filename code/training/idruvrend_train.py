import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch

import utils.general as utils
import utils.plots as plt
import pdb

class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = kwargs['train_cameras']
        initmode=kwargs['initmode']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        # print (self.expname)
        print (kwargs['conf'])
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
                                                                                          **dataset_conf)

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            # collate_fn=self.train_dataset.collate_fn,
                                                            # pin_memory=True,
                                                            num_workers=8
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                            shuffle=True,
                                                            # collate_fn=self.train_dataset.collate_fn,
                                                            # pin_memory=True
                                                            )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.grad_clip = self.conf.get_float('train.grad_clip')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        self.start_iter = 0 
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            # this is for loading pretrained model
            if timestamp=='pretrained':
                uvfwd_saved_model_state = torch.load("../exps/doc3d_uvprior/latest_uvfwd2.pth")
                uvfwd_saved_model_state["model_state_dict"]=utils.update_state_dict(uvfwd_saved_model_state["model_state_dict"],'forward_network.lin','lin' )
                self.model.forward_network.load_state_dict(uvfwd_saved_model_state["model_state_dict"], strict=True)
            else:
                if trainstep=='step1':
                    impl_saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
                    impl_saved_model_state["model_state_dict"]=utils.update_state_dict(impl_saved_model_state["model_state_dict"],'implicit_network.lin','lin' )
                    self.model.implicit_network.load_state_dict(impl_saved_model_state["model_state_dict"], strict=False)
                    
                    uvfwd_saved_model_state = torch.load("../exps/doc3d_uvprior/latest_uvfwd2.pth")
                    uvfwd_saved_model_state["model_state_dict"]=utils.update_state_dict(uvfwd_saved_model_state["model_state_dict"],'forward_network.lin','lin' )
                    self.model.forward_network.load_state_dict(uvfwd_saved_model_state["model_state_dict"], strict=False)
                    
                    uvbwd_saved_model_state = torch.load('../exps/doc3d_3dfield/1.pth')
                    uvbwd_saved_model_state["model_state_dict"]=utils.update_state_dict(uvbwd_saved_model_state["model_state_dict"],'backward_network.lin','lin' )
                    self.model.backward_network.load_state_dict(uvbwd_saved_model_state["model_state_dict"], strict=False)
                else:
                    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
                    self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
                    self.start_epoch = saved_model_state.get('epoch',0)
                    self.start_iter = saved_model_state.get('iter',0)


                
            
            optim_params_path=os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth")
            if os.path.exists(optim_params_path):
                data = torch.load(optim_params_path)
                # self.fboptimizer.load_state_dict(data["optimizer_state_dict"])
                self.optimizer.load_state_dict(data["optimizer_state_dict"])
            else: 
                print('Optimizer parameters not loaded..!')
             
            sched_params_path=os.path.join(old_checkpnts_dir, 'SchedulerParameters', str(kwargs['checkpoint']) + ".pth")
            if os.path.exists(sched_params_path):
                data = torch.load(sched_params_path)
                self.scheduler.load_state_dict(data["scheduler_state_dict"])
            else:
                print('Scheduler parameters not loaded..!')
            # pdb.set_trace()
            if self.train_cameras and timestamp!='pretrained':
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')
        self.model_conf = self.conf.get_config('model')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        self.grad_clip_milestones = self.conf.get_list('train.grad_clip_milestones', default=[])
        self.grad_clip_factor = self.conf.get_list('train.grad_clip_factor', default=[0.0])
        for acc in self.alpha_milestones:
            if self.start_iter > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor
        self.unfreeze_fwd=self.conf.get_int('train.unfreeze_fwd')
        self.freeze_sdf=self.conf.get_bool('train.freeze_sdf')
                
        self.bwd_wt_milestones = self.conf.get_list('train.bwd_wt_milestones', default=[])
        self.fwd_wt_milestones = self.conf.get_list('train.fwd_wt_milestones', default=[])
        # self.bwd_wt_factor = self.conf.get_float('train.bwd_wt_factor', default=0.0)
        self.bwd_wt_factor = self.conf.get_list('train.bwd_wt_factor', default=[0.0])
        # self.fwd_wt_factor = self.conf.get_float('train.fwd_wt_factor', default=0.0)
        self.fwd_wt_factor = self.conf.get_list('train.fwd_wt_factor', default=[0.0])
        for acc in self.bwd_wt_milestones:
            if self.start_iter > acc:
                ix=self.bwd_wt_milestones.index(acc)
                self.loss.pt_weight = self.bwd_wt_factor[ix]
        
        for acc in self.fwd_wt_milestones:
            if self.start_iter > acc:
                ix=self.fwd_wt_milestones.index(acc)
                self.loss.uv_weight = self.fwd_wt_factor[ix]

        if self.start_iter>self.grad_clip_milestones[-1]:
            self.grad_clip= self.grad_clip_factor[-1]
            print ('grad_clip={}'.format(self.grad_clip))
        else:
            # pdb.set_trace()
            for a in range(len(self.grad_clip_milestones)-1):
                acc1=self.grad_clip_milestones[a]
                acc2=self.grad_clip_milestones[a+1]
                if (self.start_iter >= acc1) and (self.start_iter < acc2):
                    self.grad_clip= self.grad_clip_factor[a]
                    print ('grad_clip={}'.format(self.grad_clip))
                    break
                elif self.start_iter >= acc2:
                    self.grad_clip= self.grad_clip_factor[a+1]
                    print ('grad_clip={}'.format(self.grad_clip))
                    break

    def save_checkpoints(self, epoch, itert):
        torch.save(
            {"iter":itert, "epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"iter":itert, "epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"iter":itert, "epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        # torch.save(
        #     {"epoch": epoch, "optimizer_state_dict": self.fboptimizer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + "_fb.pth"))
        torch.save(
            {"iter":itert, "epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        # torch.save(
        #     {"epoch": epoch, "optimizer_state_dict": self.fboptimizer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest_fb.pth"))

        torch.save(
            {"iter":itert, "epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        # torch.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.fbscheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + "_fb.pth"))
        torch.save(
            {"iter":itert, "epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))
        # torch.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.fbscheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest_fb.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))
    # pdb.set_trace()
    def run(self):
        print("training...")
        itert=self.start_iter
        for epoch in range(self.start_epoch, self.nepochs + 1):  
            
            if epoch % 100 == 0:
                self.save_checkpoints(epoch, itert)

            if epoch % self.plot_freq == 0:
                self.model.eval()
                if self.train_cameras:
                    self.pose_vecs.eval()
                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["uv_inp"] = model_input["uv_inp"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()
                # pdb.set_trace()
                split = utils.split_input(model_input, self.total_pixels)
                res = []
                for sx in split:
                    out = self.model(sx)
                    # pdb.set_trace()
                    res.append({
                        'points': out['points'].detach(),
                        'rgb_values': out['rgb_values'].detach(),
                        'normals' : out['normals'].detach(),
                        'network_object_mask': out['network_object_mask'].detach(),
                        'object_mask': out['object_mask'].detach(),
                        'uv_points': out['uv_points'].detach()
                    })

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                points= model_outputs['points']
                mask= model_outputs['network_object_mask'] & model_outputs['object_mask']
                norm_points=points.clone()
                norm_dsp=points[mask].clone()
                # dsp=points[mask].clone()
                # pdb.set_trace()
                if self.model_conf.get_bool('normalize'):
                    # normalize 
                    centroid_x=norm_dsp[:,0].mean()
                    centroid_y=norm_dsp[:,1].mean()
                    centroid_z=norm_dsp[:,2].mean()
                    # translate to (0,0,0)
                    norm_dsp[:,0]=norm_dsp[:,0]-centroid_x
                    norm_dsp[:,1]=norm_dsp[:,1]-centroid_y
                    norm_dsp[:,2]=norm_dsp[:,2]-centroid_z
                    #furthest point from (0,0,0)
                    furthest_distance = torch.max(torch.sqrt(torch.sum(abs(norm_dsp.clone())**2,dim=-1)))
                    norm_dsp /= furthest_distance
                    
                # if self.model_conf.get_bool('fixed_transform'):
                #     #data1
                #     '''
                #     R=torch.Tensor([[[ 0.8698,  0.4921, -0.0368],
                #             [-0.4924,  0.8607, -0.1292],
                #             [-0.0319,  0.1306,  0.9909]]]).to(norm_dsp.device)
                #     T=torch.Tensor([[ 0.0150, -0.0067, -0.0920]]).to(norm_dsp.device)
                #     S=torch.Tensor([[3.2347]]).to(norm_dsp.device)
                #     '''
                #     #data12
                #     R=torch.Tensor([[[ 0.8640,  0.4948, -0.0936],
                #                     [-0.5017,  0.8616, -0.0765],
                #                     [ 0.0427,  0.1131,  0.9927]]]).to(norm_dsp.device)
                #     R90=torch.Tensor([[[ 0.,  -1., 0.],
                #                     [1.,  0., 0.],
                #                     [ 0.,  0., 1.]]]).to(norm_dsp.device)
                #     T=torch.Tensor([[-0.0148,  0.0171, -0.1117]]).to(norm_dsp.device)
                #     S=torch.Tensor([[-3.2347]]).to(norm_dsp.device)
                #     # pdb.set_trace()
                    
                #     norm_dsp=torch.bmm(norm_dsp.unsqueeze(0),R)
                #     norm_dsp=torch.bmm(norm_dsp,R90)
                #     norm_dsp=norm_dsp+T
                #     norm_dsp=norm_dsp*S
                    
                # if self.model_conf.get_bool('fixed_transform') or self.model_conf.get_bool('normalize'):
                    norm_points[mask]=norm_dsp.squeeze()
                    uvmodel_input={
                        'points':points.cpu().unsqueeze(0),
                        'norm_points':norm_points.cpu().unsqueeze(0),
                        'mask':mask.cpu().unsqueeze(0),
                        'uv':model_input["uv"].cpu(),
                        'pose': model_input["pose"],
                        'intrinsics':model_input["intrinsics"]
                    }
                    split_uv_inp = utils.split_input_uvinfer(uvmodel_input, self.total_pixels)
                    uvres = []
                    for sx in split_uv_inp:
                        out = self.model.infer_fwd_bwd(sx)
                        uvres.append({
                            'uv_points': out['uv_points'].detach(),
                            'rgb': out['rgb'].detach(),
                            'normals': out['normals'].detach(),
                            # 'uv2surface_points':out['uv2surface_points'].detach()
                        })
                    # pdb.set_trace()
                    uvmodel_outputs = utils.merge_output(uvres, self.total_pixels, batch_size)
                    model_outputs['uv_points']=uvmodel_outputs['uv_points']
                    model_outputs['rgb_values']=uvmodel_outputs['rgb']
                    model_outputs['normals']=uvmodel_outputs['normals']
                
                # pdb.set_trace()
                plt.plot(self.model,
                        indices,
                        model_outputs,
                        model_input['pose'],
                        ground_truth['rgb'],
                        self.plots_dir,
                        epoch,
                        self.img_res,
                        **self.plot_conf
                        )
                
                uv_eval=model_outputs['uv_points']
                # rgb_gt= ground_truth['rgb']
                plt.plot_uv_pred(uv_eval.unsqueeze(0), self.plots_dir, epoch,1,self.img_res)
                plt.plot_warp_tex(ground_truth['tex'], uv_eval.unsqueeze(0), self.plots_dir, epoch,1,self.img_res)

                self.model.train()
                if self.train_cameras:
                    self.pose_vecs.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)
            # self.model.implicit_network.eval()
            if epoch<self.unfreeze_fwd:
                self.model.forward_network.eval()
            if self.freeze_sdf:
                self.model.implicit_network.eval()
           
            dsp_list=[]
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                itert+=1
                
                #update training params
                if itert in self.alpha_milestones:
                    self.loss.alpha = self.loss.alpha * self.alpha_factor
                
                if itert in self.bwd_wt_milestones:
                    ix=self.bwd_wt_milestones.index(itert)
                    self.loss.pt_weight = self.bwd_wt_factor[ix]
                    print('bwd weight: {}'.format(self.loss.pt_weight))
                    
                if itert in self.fwd_wt_milestones:
                    ix=self.fwd_wt_milestones.index(itert)
                    self.loss.uv_weight = self.fwd_wt_factor[ix]
                    print('fwd weight: {}'.format(self.loss.uv_weight))
                    
                # pdb.set_trace()
                
                if itert>self.grad_clip_milestones[-1]:
                    self.grad_clip= self.grad_clip_factor[-1]
                else:
                    for a in range(len(self.grad_clip_milestones)-1):
                        acc1=self.grad_clip_milestones[a]
                        acc2=self.grad_clip_milestones[a+1]
                        if (itert >= acc1) and (itert < acc2):
                            self.grad_clip= self.grad_clip_factor[a]
                            # print ('grad_clip={}'.format(self.grad_clip))
                            break
                        elif self.start_iter >= acc2:
                            self.grad_clip= self.grad_clip_factor[a+1]
                            # print ('grad_clip={}'.format(self.grad_clip))
                            break
                
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["uv_inp"] = model_input["uv_inp"].cuda()
                # pdb.set_trace()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()
                    

                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)
                
                # save
                # dsp=model_outputs['diff_surface_points']
                # dsp_list.append(dsp.detach().cpu().numpy())
                loss = loss_output['loss']

                self.optimizer.zero_grad()
                # self.fboptimizer.zero_grad()
                if self.train_cameras:
                    self.optimizer_cam.zero_grad()

                loss.backward()
                if self.grad_clip>0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.implicit_network.parameters(), self.grad_clip) # use for better training stability, must for initial round of training without isometry
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) # use for better training stability, use higher clip for isometry training
                self.optimizer.step()
                
                # self.fboptimizer.step()
                if self.train_cameras:
                    self.optimizer_cam.step()
                
                if data_index%5 ==0:
                    utils.print_log(self.expname, epoch, data_index, self.n_batches,loss_output,self.loss.alpha,self.scheduler.get_lr())
                    
                # scheduler per iteration
                self.scheduler.step()
