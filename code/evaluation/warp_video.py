import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math
import pdb
import cv2
import tqdm
import imageio

import utils.general as utils
import utils.plots as plt
from utils import rend_util
import torch.nn.functional as F
import matplotlib.pyplot as matplt

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    plot_conf = conf.get_config('plot')
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_cameras = kwargs['eval_cameras']
    eval_rendering = kwargs['eval_rendering']
    texpath = kwargs['texpath']
    prefwd_model = kwargs['prefwd_model']
    res=kwargs['resolution']
    hres=kwargs['hres']


    expname = conf.get_string('train.expname') + kwargs['expname']
    data_folder= conf.get_string('dataset.data_dir')
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname, data_folder)
    utils.mkdir_ifnotexists(evaldir)

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(eval_cameras, **dataset_conf)

    # settings for camera optimization
    # scale_mat = eval_dataset.get_scale_mat()

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    collate_fn=eval_dataset.collate_fn
                                                    )
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res
    if hres:
        tex_res = (res,res)
    else:
        tex_res = eval_dataset.tex_res

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
    epoch = saved_model_state['epoch'] 

    if prefwd_model is not None:
        print ("Lodaing pre-trained uv-forward model...")
        uvfwd_saved_model_state = torch.load(prefwd_model)
        uvfwd_saved_model_state["model_state_dict"]=utils.update_state_dict(uvfwd_saved_model_state["model_state_dict"],'forward_network.lin','lin' )
        model.forward_network.load_state_dict(uvfwd_saved_model_state["model_state_dict"], strict=True)
    

    ####################################################################################################################
    print("evaluating...")

    model.eval()

    tex = rend_util.load_rgb(texpath, resize=tex_res)
    tex=torch.from_numpy(tex).float()


    images_dir = '{0}/warp_tex-ep{1}-ts{2}'.format(evaldir, epoch,timestamp)
    utils.mkdir_ifnotexists(images_dir)

    psnrs = []
    frames=[]
    # xxx=0
    for data_index, (indices, model_input) in tqdm.tqdm(enumerate(eval_dataloader)):
        # xxx+=1
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()
        model_input['pose'] = model_input['pose'].cuda()

        split = utils.split_input(model_input, total_pixels)
        res = []
        for s in split:
            out = model(s)
            res.append({
                'uv_points': out['uv_points'].detach(),
                'object_mask': out['object_mask'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'points': out['points'].detach(),
            })
            
        batch_size = 1
        model_outputs = utils.merge_output(res, total_pixels, batch_size)
        
        # plt.plot_(model,
        # indices,
        # model_outputs,
        # model_input['pose'],
        # images_dir,
        # 0,
        # img_res,
        # **plot_conf
        # )
        
        uv_eval=model_outputs['uv_points']
        plt.plot_warp_tex(tex.unsqueeze(0), uv_eval.unsqueeze(0), images_dir, indices[0],1,img_res, resize=True, blur=True)        
        img=plt.warp_tex(tex.unsqueeze(0), uv_eval.unsqueeze(0), images_dir, indices[0],1,img_res, resize=True, blur=True)
        frames.append(img)
        # # pdb.set_trace()
        # if xxx==10:
        #     break
    frames=np.stack(frames, axis=0)
    print("Writing video")
    vid_name = "360vid"
    vid_path = os.path.join(images_dir, vid_name + ".mp4")
    imageio.mimwrite(
        vid_path, frames.astype(np.uint8), fps=20, quality=8
    )
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')
    parser.add_argument('--eval_cameras', default=False, action="store_true", help='If set, evaluate camera accuracy of trained cameras.')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')
    parser.add_argument('--hres', default=False, action="store_true", help='HR warped images.')
    parser.add_argument('--texture', default=None, type=str, help='Specifies a texture to warp')
    parser.add_argument('--prefwd_model', default=None, type=str, help='Specifies a fwd model to load from')
    
    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_cameras=opt.eval_cameras,
             eval_rendering=opt.eval_rendering,
             texpath=opt.texture,
             prefwd_model=opt.prefwd_model,
             hres=opt.hres
             )
