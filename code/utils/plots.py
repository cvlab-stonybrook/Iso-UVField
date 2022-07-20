import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util
import pdb
import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   

def plot(model, indices, model_outputs ,pose, rgb_gt, path, epoch, img_res, plot_nimgs, max_depth, resolution):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)
    norm_eval = model_outputs['normals']
    norm_eval = norm_eval.reshape(batch_size, num_samples, 3)
    uv_eval=None
    if model_outputs.get('uv_points',None) is not None:
        uv_eval = model_outputs['uv_points']
        uv_eval = uv_eval.reshape(batch_size, num_samples, 2)

    depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = depth.reshape(batch_size, num_samples, 1)
    network_object_mask = network_object_mask.reshape(batch_size,-1)

    cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)
    
    # pdb.set_trace()
    # plot rendered images
    plot_images(rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)
    
    # plot rendered norm
    plot_norms(norm_eval, path, epoch, plot_nimgs, img_res)

    # # plot depth maps
    # plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)
    
    # plot uv maps
    if uv_eval is not None:
        plot_uv_pred(uv_eval, path, epoch, plot_nimgs, img_res)
        
    '''
    data = []

    # plot surface
    surface_traces = get_surface_trace(path=path,
                                        epoch=epoch,
                                        sdf=lambda x: model.implicit_network(x)[:, 0],
                                        resolution=resolution
                                        )
    data.append(surface_traces[0])

    # plot cameras locations
    for i, loc, dir in zip(indices, cam_loc, cam_dir):
        data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))

    for i, p, m in zip(indices, points, network_object_mask):
        p = p[m]
        sampling_idx = torch.randperm(p.shape[0])[:2048]
        p = p[sampling_idx, :]

        val = model.implicit_network(p)
        caption = ["sdf: {0} ".format(v[0].item()) for v in val]

        data.append(get_3D_scatter_trace(p, name='intersection_points_{0}'.format(i), caption=caption))

    fig = go.Figure(data=data)
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                        yaxis=dict(range=[-3, 3], autorange=False),
                        zaxis=dict(range=[-3, 3], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    filename = '{0}/surface_{1}.html'.format(path, epoch)
    offline.plot(fig, filename=filename, auto_open=False)
    '''



def plot_(model, indices, model_outputs ,pose, path, epoch, img_res, plot_nimgs, max_depth, resolution):
    # arrange data to plot
    # batch_size, num_samples, _ = rgb_gt.shape
    num_samples=448*448
    batch_size=1

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    # rgb_eval = model_outputs['rgb_values']
    # rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)
    uv_eval=None
    if model_outputs.get('uv_points',None) is not None:
        uv_eval = model_outputs['uv_points']
        uv_eval = uv_eval.reshape(batch_size, num_samples, 2)

    depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = depth.reshape(batch_size, num_samples, 1)
    network_object_mask = network_object_mask.reshape(batch_size,-1)

    cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)
    
    # pdb.set_trace()
    # plot rendered images
    # plot_images(rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # # plot depth maps
    # plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)
    
    # plot uv maps
    if uv_eval is not None:
        plot_uv_pred(uv_eval, path, epoch, plot_nimgs, img_res)
        
    
    data = []

    # plot surface
    surface_traces = get_surface_trace(path=path,
                                        epoch=epoch,
                                        sdf=lambda x: model.implicit_network(x)[:, 0],
                                        resolution=resolution
                                        )
    data.append(surface_traces[0])

    # plot cameras locations
    for i, loc, dir in zip(indices, cam_loc, cam_dir):
        data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))

    for i, p, m in zip(indices, points, network_object_mask):
        p = p[m]
        sampling_idx = torch.randperm(p.shape[0])[:2048]
        p = p[sampling_idx, :]

        val = model.implicit_network(p)
        caption = ["sdf: {0} ".format(v[0].item()) for v in val]

        data.append(get_3D_scatter_trace(p, name='intersection_points_{0}'.format(i), caption=caption))

    fig = go.Figure(data=data)
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                        yaxis=dict(range=[-3, 3], autorange=False),
                        zaxis=dict(range=[-3, 3], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    filename = '{0}/surface_{1}.html'.format(path, epoch)
    offline.plot(fig, filename=filename, auto_open=False)
    




def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, resolution=100, return_mesh=False):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_high_res_mesh(sdf, resolution=100):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=0,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_grid_uniform(resolution):
    x = np.linspace(-1.0, 1.0, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, epoch))

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = (ground_true.cuda() + 1.) / 2.
    rgb_points = (rgb_points + 1. ) / 2.

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))
    
def plot_norms(norm_points, path, epoch, plot_nrow, img_res):
    # pdb.set_trace()
    norm_plot = lin2img(norm_points, img_res)

    tensor = torchvision.utils.make_grid(norm_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    # pdb.set_trace()
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/norm_{1}.png'.format(path, epoch))
    
def plot_uv_img(rgb_gt,uv_points, path, epoch, plot_nrow, img_res):
    # pdb.set_trace()
    rgb_gt = (rgb_gt.cuda() + 1.) / 2.
    rgb_plot = lin2img(rgb_gt, img_res)

    tensor = torchvision.utils.make_grid(rgb_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rgbgt_{1}.png'.format(path, epoch))
    
    # add a channel of ones
    uv_plot = lin2img(uv_points, img_res)
    b,c,h,w= uv_plot.shape
    zeros=torch.zeros((b,1,h,w,))
    zeros[uv_plot[:,0,:,:].unsqueeze(1)>0]=1.0
    uv_plot=torch.cat([uv_plot,zeros.cuda()], dim=1)
    # pdb.set_trace()
    uv_plot[uv_plot<0]=0.0
    # pdb.set_trace()
    tensor = torchvision.utils.make_grid(uv_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    # pdb.set_trace()
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/uv_{1}.png'.format(path, epoch))

def plot_warp_tex(tex,uv_points, path, epoch, plot_nrow, img_res, resize=False, blur=False):
    tex = (tex + 1. ) / 2.
    b,c,h,w=tex.shape
    uv_points=uv_points.reshape(img_res[0],img_res[1],2)
    uv_points[uv_points<0]=-99.0
    uv_points=(uv_points-0.5)*2.0
    # pdb.set_trace()
    # uv_points_=torch.index_select(uv_points, -1, torch.LongTensor([1,0]).to(uv_points.device))
    b,c,w,h=tex.shape
    if resize:
        xx=cv2.resize(uv_points[:,:,0].cpu().numpy(),(w,h))
        yy=cv2.resize(uv_points[:,:,1].cpu().numpy(),(w,h))
        if blur:
            xx=cv2.blur(xx,(7,7))
            yy=cv2.blur(yy,(7,7))
        uv_points=np.stack([xx,yy], axis =-1)
        uv_points=torch.from_numpy(uv_points).cuda()
    warped_tex=F.grid_sample(tex.to(uv_points.device), uv_points.unsqueeze(0), mode='bilinear')
    # wt=warped_tex.reshape(b,c,img_res[1],img_res[0])
    tensor = torchvision.utils.make_grid(warped_tex,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/warped_{1}.png'.format(path, epoch))


def warp_tex(tex,uv_points, path, epoch, plot_nrow, img_res, resize=False, blur=False):
    tex = (tex + 1. ) / 2.
    b,c,h,w=tex.shape
    uv_points=uv_points.reshape(img_res[0],img_res[1],2)
    uv_points[uv_points<0]=-99.0
    uv_points=(uv_points-0.5)*2.0
    # pdb.set_trace()
    # uv_points_=torch.index_select(uv_points, -1, torch.LongTensor([1,0]).to(uv_points.device))
    b,c,w,h=tex.shape
    if resize:
        xx=cv2.resize(uv_points[:,:,0].cpu().numpy(),(w,h))
        yy=cv2.resize(uv_points[:,:,1].cpu().numpy(),(w,h))
        if blur:
            xx=cv2.blur(xx,(7,7))
            yy=cv2.blur(yy,(7,7))
        uv_points=np.stack([xx,yy], axis =-1)
        uv_points=torch.from_numpy(uv_points).cuda()
    warped_tex=F.grid_sample(tex.to(uv_points.device), uv_points.unsqueeze(0), mode='bilinear')
    # wt=warped_tex.reshape(b,c,img_res[1],img_res[0])
    tensor = torchvision.utils.make_grid(warped_tex,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    
    return img
    
    
    
def plot_uv_pred(uv_points, path, epoch, plot_nrow, img_res):
    # pdb.set_trace()
    # add a channel of ones
    uv_plot = lin2img(uv_points, img_res)
    b,c,h,w= uv_plot.shape
    zeros=torch.zeros((b,1,h,w,))
    zeros[uv_plot[:,0,:,:].unsqueeze(1)>0]=1.0
    uv_plot=torch.cat([uv_plot,zeros.cuda()], dim=1)
    # pdb.set_trace()
    uv_plot[uv_plot<0]=0.0
    # pdb.set_trace()
    tensor = torchvision.utils.make_grid(uv_plot,
                                        scale_each=False,
                                        normalize=False,
                                        nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    # pdb.set_trace()
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/uv_{1}.png'.format(path, epoch))
    
    
def plot_uv_maps(uv_gt, uv_points, path, epoch, plot_nrow, img_res):
    # pdb.set_trace()
    # add a channel of ones
    uv_plot = lin2img(uv_points, img_res)
    b,c,h,w= uv_plot.shape
    zeros=torch.zeros((b,1,h,w,))
    zeros[uv_plot[:,0,:,:].unsqueeze(1)>0]=1.0
    uv_plot=torch.cat([uv_plot,zeros.cuda()], dim=1)
    # pdb.set_trace()
    uv_plot[uv_plot<0]=0.0
    # pdb.set_trace()
    tensor = torchvision.utils.make_grid(uv_plot,
                                        scale_each=False,
                                        normalize=False,
                                        nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    # pdb.set_trace()
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/uv-pred_{1}.png'.format(path, epoch))
    
    uv_plot = lin2img(uv_gt, img_res)
    b,c,h,w= uv_plot.shape
    zeros=torch.zeros((b,1,h,w,))
    zeros[uv_plot[:,0,:,:].unsqueeze(1)>0]=1.0
    uv_plot=torch.cat([uv_plot,zeros.cuda()], dim=1)
    # pdb.set_trace()
    uv_plot[uv_plot<0]=0.0
    # pdb.set_trace()
    tensor = torchvision.utils.make_grid(uv_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    # pdb.set_trace()
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/uv-gt_{1}.png'.format(path, epoch))

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def plot_grad_flow(named_parameters, path, epoch, dti):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('{0}/grad_{1}_{2}.png'.format(path, epoch, dti),bbox_inches='tight',dpi=100)

def plot_grad_flow_v2(named_parameters, netname, path, epoch, dti):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, px in named_parameters:
        if(px.requires_grad) and ("bias" not in n):
            layers.append(n)
            # pdb.set_trace()
            ave_grads.append(px.grad.abs().mean())
            max_grads.append(px.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('{0}/{1}grad_{2}_{3}.png'.format(path,netname, epoch, dti),bbox_inches='tight',dpi=100)
    plt.clf()