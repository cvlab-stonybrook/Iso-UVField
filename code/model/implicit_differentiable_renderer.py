import torch
import torch.nn as nn
import numpy as np
from torchviz import make_dot

from utils import rend_util, diff_props
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.pointnet import ANEncoderPN, STN3d
import pdb

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            act='softp'
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            # embed_fn, input_ch = get_embedder(multires, 3)
            embed_fn=PositionalEncoding(num_freqs=multires, d_in=d_in, freq_factor=1.0)  
            self.embed_fn = embed_fn
            # dims[0] = input_ch
            dims[0] = embed_fn.d_out

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            
            if act=='sine':
                if l==0:
                    lin.apply(first_layer_sine_init)
                else:
                    lin.apply(sine_init)
        self.act=act     
        if self.act=='softp':
            self.actv = nn.Softplus(beta=100)
        elif self.act=='sine':
            self.actv = Sine()
        self.softp = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            
            if self.act=='softp':
                if l < self.num_layers - 2:
                    x = self.actv(x)
            else:
                if l < self.num_layers - 2:
                # pdb.set_trace()
                    x=self.actv(x)     
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class Sine(nn.Module):
    def __init__(self, f=30):
        super().__init__()
        self.factor=f
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.factor * input)


class UVNetwork(nn.Module):
    def __init__(
        self,
        embed_in,
        input_feat_size,        # if global features are used
        d_in,                   # x,y,z
        d_out,                  # u,v
        dims,                   # hidden dimensions
        act,                    # inner_activations
        out_act,                # final_activation
        norm,                   # per layer norm
        multires,               # dimensions of fourier features
        bias,                    # bias for linear layers 
        sine_factor=30
    ):
        super().__init__()
        dims = [d_in+ input_feat_size] + dims + [d_out]
        
        self.num_layers = len(dims)
        
        self.embed_fn = None
        if multires > 0:
            # pdb.set_trace()
            '''
            embed_fn, input_ch = get_embedder(multires, embed_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch-embed_in
            '''
            embed_fn=PositionalEncoding(num_freqs=multires, d_in=embed_in, freq_factor=1.0)  
            self.embed_fn = embed_fn
            dims[0] += embed_fn.d_out-embed_in
        
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim, bias=bias)

            if norm=='weight' or l==self.num_layers - 2:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if act=='sine':
                if l==0:
                    lin.apply(first_layer_sine_init)
                elif l < self.num_layers - 2:
                    if sine_factor==10:
                        # pdb.set_trace()
                        lin.apply(sine_init_ten)
                    else:
                        lin.apply(sine_init)
                else:
                    lin.apply(final_layer_init_normal)
            else:
                '''
                for the last layer initialize so that outputs are uniform [0,1]
                '''
                lin.apply(final_layer_init_normal)

        self.relu = nn.ReLU()
        self.sine = Sine(f=sine_factor)
        self.sigm=nn.Sigmoid()
        self.htanh = nn.Hardtanh(0,1)
        self.htanhm1 = nn.Hardtanh(-1,1)
        self.softp=nn.Softplus(beta=100)
        self.act=act
        self.out_act=out_act
        
    def forward(self, points, uv_inp, global_feats=None):
        # pdb.set_trace()
        if self.embed_fn is not None:
            if len(points.shape)==3: #B,N,D
                # pdb.set_trace()
                B,N,D=points.shape
                points=points.view(B*N,D)
                points = self.embed_fn(points)
                points=points.view(B,N,-1)
            else:
                points = self.embed_fn(points)
        if uv_inp is not None and global_feats is not None:
            x = torch.cat([points, uv_inp, global_feats], dim=-1)
        elif global_feats is not None and uv_inp is None:
            x = torch.cat([points, global_feats], dim=-1)
        elif global_feats is None and uv_inp is not None:
            x = torch.cat([points, uv_inp], dim=-1)
        else:
            # pdb.set_trace()
            x = points

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            
            x = lin(x)
            
            if l < self.num_layers - 2:
                if self.act=='sine':
                    x = self.sine(x)
                elif self.act=='relu':
                    x = self.relu(x)
                elif self.act=='sigm':
                    x = self.sigm(x)
                else: 
                    x = self.softp(x)
        
        if self.out_act=='sine':
            x = self.sine(x)
        elif self.out_act=='htanm1':
            x = self.htanhm1(x)
        elif self.out_act=='htan':
            x = self.htanh(x)
        elif self.out_act=='sigm':
            x = self.sigm(x)
        # pdb.set_trace()
        return x
    

class UVOffNetwork(nn.Module):
    def __init__(
        self,
        embed_in,
        input_feat_size,        # if global features are used
        d_in,                   # x,y,z
        d_out,                  # u,v
        dims,                   # hidden dimensions
        act,                    # inner_activations
        out_act,                # final_activation
        norm,                   # per layer norm
        multires                # dimensions of fourier features 
    ):
        super().__init__()
        dims = [d_in+ input_feat_size] + dims + [d_out]
        
        self.num_layers = len(dims)
        
        self.embed_fn = None
        if multires > 0:
            # pdb.set_trace()
            '''
            embed_fn, input_ch = get_embedder(multires, embed_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch-embed_in
            '''
            embed_fn=PositionalEncoding(num_freqs=multires, d_in=embed_in, freq_factor=1.0)  
            self.embed_fn = embed_fn
            dims[0] += embed_fn.d_out-embed_in
        
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if norm=='weight' or l==self.num_layers - 2:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if act=='sine':
                if l==0:
                    lin.apply(first_layer_sine_init)
                elif l < self.num_layers - 2:
                    lin.apply(sine_init)
                else:
                    lin.apply(final_layer_init_normal)
            else:
                '''
                for the last layer initialize so that outputs are uniform [0,1]
                '''
                lin.apply(final_layer_init_normal)

        self.relu = nn.ReLU()
        self.sine = Sine()
        self.sigm=nn.Sigmoid()
        self.htanh = nn.Hardtanh(0,1)
        self.htanhm1 = nn.Hardtanh(-1,1)
        self.softp=nn.Softplus(beta=100)
        self.act=act
        self.out_act=out_act
        
    def forward(self, points, uv_inp, global_feats=None):
        # pdb.set_trace()
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        if uv_inp is not None and global_feats is not None:
            x = torch.cat([points, uv_inp, global_feats], dim=-1)
        elif global_feats is not None and uv_inp is None:
            x = torch.cat([points, global_feats], dim=-1)
        elif global_feats is None and uv_inp is not None:
            x = torch.cat([points, uv_inp], dim=-1)
        else:
            # pdb.set_trace()
            x = points

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            
            x = lin(x)
            
            if l < self.num_layers - 2:
                if self.act=='sine':
                    x = self.sine(x)
                elif self.act=='relu':
                    x = self.relu(x)
                elif self.act=='sigm':
                    x = self.sigm(x)
                else: 
                    x = self.softp(x)
        x=x+uv_inp
        
        if self.out_act=='sine':
            x = self.sine(x)
        elif self.out_act=='htan':
            x = self.htanh(x)
        elif self.out_act=='sigm':
            x = self.sigm(x)
        
        return x


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            act,
            weight_norm=True,
            multires_view=0,
            sine_factor=30
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            '''
            embedview_fn, input_ch = get_embedder(multires_view, 3)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
            '''
            embedview_fn=PositionalEncoding(num_freqs=multires_view, d_in=3, freq_factor=1.0)  
            self.embedview_fn = embedview_fn
            dims[0] += embedview_fn.d_out-3

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if act=='sine':
                if l==0:
                    lin.apply(first_layer_sine_init)
                else:
                    if sine_factor==10:
                        # pdb.set_trace()
                        lin.apply(sine_init_ten)
                    else:
                        lin.apply(sine_init)

        self.relu = nn.ReLU()
        self.sine = Sine(f=sine_factor)
        self.tanh = nn.Tanh()
        self.act=act

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                if self.act=='sine':
                    x = self.sine(x)
                else:
                    x = self.relu(x)
        if self.act=='sine':
            x = self.sine(x)
        else:
            x = self.tanh(x)
        return x

class RenderingUVNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            act,
            weight_norm=True,
            multires_view=0,
            sine_factor=30
    ):
        super().__init__()

        self.mode = mode
        if self.mode =='no_normal_feat':
            dims = [d_in] + dims + [d_out]
        else:
            dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            '''
            embedview_fn, input_ch = get_embedder(multires_view, 3)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
            '''
            embedview_fn=PositionalEncoding(num_freqs=multires_view, d_in=3, freq_factor=1.0)  
            self.embedview_fn = embedview_fn
            dims[0] += embedview_fn.d_out-3

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if act=='sine':
                if l==0:
                    lin.apply(first_layer_sine_init)
                else:
                    if sine_factor==10:
                        # pdb.set_trace()
                        lin.apply(sine_init_ten)
                    else:
                        lin.apply(sine_init)
                    
        self.relu = nn.ReLU()
        self.sine = Sine(f=sine_factor)
        self.tanh = nn.Tanh()
        self.act=act

    def forward(self, uv, points, normals, view_dirs, feature_vectors):
        # pdb.set_trace()
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, uv, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, uv, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, uv, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'no_normal_feat':
            rendering_input = torch.cat([points, uv, view_dirs], dim=-1)
        elif self.mode == 'no_point':
            rendering_input = torch.cat([uv, view_dirs, normals, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                if self.act=='sine':
                    x = self.sine(x)
                else:
                    x = self.relu(x)
        if self.act=='sine':
            x = self.sine(x)
        else:
            x = self.tanh(x)
        return x



class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        
        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                cam_loc=cam_loc,
                                                                object_mask=object_mask,
                                                                ray_directions=ray_dirs)
        self.implicit_network.train()
        # pdb.set_trace()
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3) # all points along the ray

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]  # ray intersection points
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]  # sdf output of the raytraced points
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0) # random points within range + all points along the ray

            points_all = torch.cat([surface_points, eikonal_points], dim=0) # ray intersection points , random points within range + all points along the ray

            output = self.implicit_network(surface_points)  # evaluate implicit function on ray intersection points
            surface_sdf_values = output[:N, 0:1].detach()

            # calculate gradient wrt ray intersection points
            # random points within range + all points along the ray
            g = self.implicit_network.gradient(points_all) 
            
            # pdb.set_trace()
            surface_points_grad = g[:N, 0, :]
            grad_theta = g[N:, 0, :]
            grad_eikonal_points = g[N:n_eik_points, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad.clone().detach(),
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
            surface_points_grad=None
            surface_output=None 
            grad_eikonal_points=None
            surface_points=None
            diff_normals=None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        normals = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask], diff_normals = self.get_rbg_value(differentiable_surface_points, view)
            normals[surface_mask] = diff_normals

        output = {
            'points': points,
            'surface_points': surface_points, # these are the ray surface intersection points 
            'diff_surface_points': differentiable_surface_points,
            'diff_normals': diff_normals, # normals at differentiable surface points
            'rgb_values': rgb_values,
            'normals': normals,
            'sdf_output': sdf_output,
            'surface_output': surface_output, # sdf values at the ray surface intersection points
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'grad_points': surface_points_grad, # normals at the ray surface intersection points
            'grad_eikpoints': grad_eikonal_points
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals, normals

class UVFieldNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.train_backward = conf.get_bool('train_backward')
        self.joint_backward = conf.get_bool('joint_backward')
        self.globalpc_feats=conf.get_bool('global_feats')
        self.uv_inp=conf.get_bool('uv_inp')
        self.forward_network = UVNetwork(3,**conf.get_config('forward_network'))
        self.backward_network = UVNetwork(2,**conf.get_config('backward_network'))
        if self.globalpc_feats:
            self.pcenc= ANEncoderPN(**conf.get_config('pcencoder'))
        
    def forward(self, input):
        object_mask = input["object_mask"].reshape(-1)
        points=input['wc'].reshape(-1,3)
        imguv=input['uv'].reshape(-1,2)/447.0
        surfpoints= points[object_mask]
        uv_points = torch.ones(points.shape[0],2).float().cuda() * -1
        if surfpoints.shape[0]>0:
            if self.globalpc_feats:
                pc_feats=self.pcenc(surfpoints.unsqueeze(0))
                # pdb.set_trace()
                pc_feats=pc_feats.repeat(surfpoints.shape[0],1)
                if self.uv_inp:
                    uv_points[object_mask]=self.forward_network(surfpoints, imguv[object_mask], global_feats=pc_feats)
                else:
                    uv_points[object_mask]=self.forward_network(surfpoints, None, global_feats=pc_feats)
            else:
                if self.uv_inp:
                    uv_points[object_mask]=self.forward_network(surfpoints, imguv[object_mask], global_feats=None)
                else:
                    uv_points[object_mask]=self.forward_network(surfpoints, None, global_feats=None)
                
                
            
        uv2surface_points=None
        if self.train_backward:
            uv2surface_points = torch.zeros_like(surfpoints).float().cuda()
            # uv points to backward network for surface points
            if self.joint_backward:
                uv2surface_points=self.backward_network(uv_points[object_mask], None, global_feats=None)
            else:
                uv2surface_points=self.backward_network(uv_points[object_mask].detach(), None, global_feats=None)

        output={
            'uv_points': uv_points,
            'uv2surface_points': uv2surface_points,
            'object_mask': object_mask
        }
        
        return output
    
class Doc3dUVFieldNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.globalpc_feats=conf.get_bool('global_feats')
        self.stn=conf.get_bool('stn')
        self.rot=conf.get_bool('stn_rot')
        self.trns=conf.get_bool('stn_trns')
        self.scale=conf.get_bool('stn_scale')
        self.forward_network = UVNetwork(3,**conf.get_config('forward_network'))
        if self.globalpc_feats:
            self.pcenc= ANEncoderPN(**conf.get_config('pcencoder'))
        if self.stn:
            # extract global features and use a 3d stn apply rigid transformation
            self.stn_net = STN3d(rot=self.rot, trans=self.trns, scale=self.scale)
        
    def forward(self, input):
        # pdb.set_trace()
        b,p=input["object_mask"].shape
        object_mask = input["object_mask"]#.reshape(b,-1)
        points=input['wc']#.reshape(b,-1,3)
        surfpoints= points[object_mask]
        # pdb.set_trace()
        pred_rotangle=None
        pred_scale=None        
        uv_points = torch.ones(b,p,2).float().cuda() * -1
        if surfpoints.shape[0]>0 and surfpoints.shape[1]>0:
            if self.stn:
                # pdb.set_trace()
                if len(surfpoints.shape)==2:
                    surfpoints=surfpoints.unsqueeze(0)
                surfpoints=surfpoints.transpose(2,1)
                stnmat, pred_rotangle, pred_scale, pred_trans= self.stn_net(surfpoints)
                surfpoints = surfpoints.transpose(2,1)
                surfpoints = torch.bmm(surfpoints, stnmat)
                
            if self.globalpc_feats:
                pc_feats=self.pcenc(surfpoints)
                # pdb.set_trace()
                pc_feats=pc_feats.repeat(surfpoints.shape[1],1)
                uv_points[object_mask]=self.forward_network(surfpoints, None, global_feats=pc_feats)
            else:
                uv_points[object_mask]=self.forward_network(surfpoints, None, global_feats=None)

        output={
            'uv_points': uv_points,
            'object_mask': object_mask,
            'pred_rotangle':pred_rotangle,
            'pred_scale':pred_scale,
            'surf_points':surfpoints
        }
        return output


class Doc3dUVCamFieldNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.globalpc_feats=conf.get_bool('global_feats')
        self.forward_network = UVNetwork(3,**conf.get_config('forward_network'))
        if self.globalpc_feats:
            self.pcenc= ANEncoderPN(**conf.get_config('pcencoder'))
        
    def forward(self, input):
        b,p=input["object_mask"].shape
        object_mask = input["object_mask"]#.reshape(-1)
        points=input['pxc']#.reshape(-1,3)
        # pdb.set_trace()
        surfpoints= points[object_mask]
        uv_points = torch.ones(b,p,2).float().cuda() * -1
        if surfpoints.shape[0]>0 and surfpoints.shape[1]>0:
            if self.globalpc_feats:
                if len(surfpoints.shape)==2:
                    surfpoints=surfpoints.unsqueeze(0)
                pc_feats=self.pcenc(surfpoints)
                # pdb.set_trace()
                pc_feats=pc_feats.repeat(surfpoints.shape[1],1)
                uv_points[object_mask]=self.forward_network(surfpoints, None, global_feats=pc_feats)
            else:
                uv_points[object_mask]=self.forward_network(surfpoints, None, global_feats=None)

        output={
            'uv_points': uv_points,
            'object_mask': object_mask
        }
        return output


class IDRUVNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.train_backward = conf.get_bool('train_backward')
        self.backprop_fwd = conf.get_bool('backprop_fwd')
        self.joint_backward = conf.get_bool('joint_backward')
        # self.normalize = conf.get_bool('normalize')
        self.uv_inp=conf.get_bool('uv_inp')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.forward_network = UVNetwork(3, **conf.get_config('forward_network'))
        self.backward_network = UVNetwork(2, **conf.get_config('backward_network'))
        
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input):
        # pdb.set_trace()
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        uv_inp = input["uv_inp"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                cam_loc=cam_loc,
                                                                object_mask=object_mask,
                                                                ray_directions=ray_dirs)
        self.implicit_network.train()
        # pdb.set_trace()
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3) # all points along the ray

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            # pdb.set_trace()
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]  # ray intersection points
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]  # sdf output of the raytraced points
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0) # random points within range + all points along the ray

            points_all = torch.cat([surface_points, eikonal_points], dim=0) # ray intersection points , random points within range + all points along the ray

            output = self.implicit_network(surface_points)  # evaluate implicit function on ray intersection points
            surface_sdf_values = output[:N, 0:1].detach()

            # calculate gradient wrt ray intersection points
            # random points within range + all points along the ray
            g = self.implicit_network.gradient(points_all) 
            
            # pdb.set_trace()
            surface_points_grad = g[:N, 0, :]
            grad_theta = g[N:, 0, :]
            grad_eikonal_points = g[N:n_eik_points, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad.clone().detach(),
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)
        
        else:
            surface_mask = network_object_mask & object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
            surface_points_grad=None
            surface_output=None 
            grad_eikonal_points=None
            surface_points=None
            diff_normals=None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        normals = torch.ones_like(points).float().cuda()
        uv_points = torch.ones(points.shape[0],2).float().cuda() * -1
        uv2surface_points=None
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value(differentiable_surface_points, view)
            # pdb.set_trace()
            normals[surface_mask] = diff_normals
            # noise_std = 0.1
            # noise = (
            #     torch.randn(
            #         differentiable_surface_points.shape,
            #         dtype=differentiable_surface_points.dtype,
            #         device=differentiable_surface_points.device,
            #     )
            #     * noise_std
            # )
            # surface points to forward network for uv points
            image_uv_inp=uv_inp.reshape(-1,2)[surface_mask]
            # pdb.set_trace()
            if self.backprop_fwd:
                dsp=differentiable_surface_points.clone().detach()
            else:
                dsp=differentiable_surface_points
                
            # pdb.set_trace()
            if self.uv_inp:
                uv_points[surface_mask]=self.forward_network(dsp, image_uv_inp, global_feats=None)
            else:
                uv_points[surface_mask]=self.forward_network(dsp, None , global_feats=None)
            
            # if self.normalize:
            #     uvp=uv_points[surface_mask]
            #     uvp[:,0]=(uvp[:,0]-uvp[:,0].min())/(uvp[:,0].max()-uvp[:,0].min())
            #     uvp[:,1]=(uvp[:,1]-uvp[:,1].min())/(uvp[:,1].max()-uvp[:,1].min())
            #     uv_points[surface_mask]=uvp
            if self.train_backward:
                uv2surface_points = torch.zeros_like(differentiable_surface_points).float().cuda()
                # uv points to backward network for surface points
                if self.joint_backward:
                    uv2surface_points=self.backward_network(uv_points[surface_mask], None, global_feats=None)
                else:
                    uv2surface_points=self.backward_network(uv_points[surface_mask].detach(), None, global_feats=None)
            
        output = {
            'points': points,
            'surface_points': surface_points, # these are the ray surface intersection points 
            'diff_surface_points': differentiable_surface_points,
            'diff_normals': diff_normals, # normals at differentiable surface points
            'rgb_values': rgb_values,
            'normals': normals,
            'sdf_output': sdf_output,
            'surface_output': surface_output, # sdf values at the ray surface intersection points
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'grad_points': surface_points_grad, # normals at the ray surface intersection points
            'grad_eikpoints': grad_eikonal_points,
            'uv_points': uv_points,
            'uv2surface_points':uv2surface_points,
            'diff_props':None
        }
        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals, normals, feature_vectors
    
    def get_rbg_value_uv(self, points, uv, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(uv, normals, view_dirs, feature_vectors)

        return rgb_vals, normals, feature_vectors


class IDRUVRendNetwork(nn.Module):
    # this network takes uv prediction and the 3d surface (xyz) both as input to the rendering network
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.isometry = conf.get_bool('isometry')
        self.train_backward = conf.get_bool('train_backward')
        self.backprop_fwd = conf.get_bool('backprop_fwd')
        self.joint_backward = conf.get_bool('joint_backward')
        self.normalize = conf.get_bool('normalize')
        self.uv_inp=conf.get_bool('uv_inp')
        self.transform=conf.get_bool('transform')
        self.fixed_transform=conf.get_bool('fixed_transform')
        self.uv_noise=conf.get_bool('uv_noise', False)
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.renderinguv_network = RenderingUVNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.forward_network = UVNetwork(3, **conf.get_config('forward_network'))
        self.backward_network = UVNetwork(2, **conf.get_config('backward_network'))
        if self.transform:
            self.stn = STN3d()
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.view=None
        if self.isometry:
            self.diffprops=diff_props.DiffGeomProps(normals=False, curv_mean=False, curv_gauss=False, fff=True)

    def forward(self, input):
        # pdb.set_trace()
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        if self.uv_inp:
            uv_inp = input["uv_inp"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                cam_loc=cam_loc,
                                                                object_mask=object_mask,
                                                                ray_directions=ray_dirs)
        self.implicit_network.train()
        # pdb.set_trace()
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3) # all points along the ray

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            # pdb.set_trace()
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]  # ray intersection points
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]  # sdf output of the raytraced points
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0) # random points within range + all points along the ray

            points_all = torch.cat([surface_points, eikonal_points], dim=0) # ray intersection points , random points within range + all points along the ray

            output = self.implicit_network(surface_points)  # evaluate implicit function on ray intersection points
            surface_sdf_values = output[:N, 0:1].detach()

            # calculate gradient wrt ray intersection points
            # random points within range + all points along the ray
            g = self.implicit_network.gradient(points_all) 
            
            # pdb.set_trace()
            surface_points_grad = g[:N, 0, :]
            grad_theta = g[N:, 0, :]
            grad_eikonal_points = g[N:n_eik_points, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad.clone().detach(),
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)
        
        else:
            surface_mask = network_object_mask & object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
            surface_points_grad=None
            surface_output=None 
            grad_eikonal_points=None
            surface_points=None
            diff_normals=None

        view = -ray_dirs[surface_mask]

        fwdrgb_values = torch.ones_like(points).float().cuda()
        # bwdrgb_values = torch.ones_like(points).float().cuda()
        normals = torch.ones_like(points).float().cuda()
        uv_points = torch.ones(points.shape[0],2, requires_grad=True).float().cuda() * -1
        uv2surface_points=None
        tdsp=None
        diff_prop=None
        if differentiable_surface_points.shape[0] > 0:
            if self.uv_inp:
                # surface points to forward network for uv points
                image_uv_inp=uv_inp.reshape(-1,2)[surface_mask]
            # pdb.set_trace()
            if self.backprop_fwd:
                dsp=differentiable_surface_points
            else:
                dsp=differentiable_surface_points.clone().detach()
                
            # pdb.set_trace()
            
            if self.normalize and self.training:
                norm_dsp=dsp.clone().detach()
                # normalize the surface points to be in unit sphere
                centroid_x=norm_dsp[:,0].mean()
                centroid_y=norm_dsp[:,1].mean()
                centroid_z=norm_dsp[:,2].mean()
                # translate to (0,0,0)
                norm_dsp[:,0]=norm_dsp[:,0]-centroid_x
                norm_dsp[:,1]=norm_dsp[:,1]-centroid_y
                norm_dsp[:,2]=norm_dsp[:,2]-centroid_z
                #furthest point from (0,0,0)
                furthest_distance = torch.max(torch.sqrt(torch.sum(abs(norm_dsp.clone())**2,dim=-1)))
                if furthest_distance>=1e-3:
                    norm_dsp /= furthest_distance
                    
                if self.transform:
                    # pdb.set_trace()
                    norm_dsp=norm_dsp.unsqueeze(0).transpose(2,1)
                    stnmat= self.stn(norm_dsp)
                    norm_dsp=norm_dsp.transpose(2,1)
                    norm_dsp = torch.bmm(norm_dsp, stnmat).squeeze(0)
                        
                if self.uv_inp:
                    uv_points[surface_mask]=self.forward_network(norm_dsp, image_uv_inp, global_feats=None)
                else:
                    uv_points[surface_mask]=self.forward_network(norm_dsp, None , global_feats=None)
                # pdb.set_trace()
                fwdrgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value_uv(dsp, uv_points[surface_mask].clone().detach(), view)
                normals[surface_mask] = diff_normals
                
                if self.train_backward:
                    uv2surface_points = torch.zeros_like(differentiable_surface_points).float().cuda()
                    # uv points to backward network for surface points
                    if self.joint_backward:
                        uv2surface_points=self.backward_network(uv_points[surface_mask], None, global_feats=None)

                    else:
                        uv2surface_points=self.backward_network(uv_points[surface_mask].clone().detach(), None, global_feats=None)
                        
                    # bwdrgb_values[surface_mask], _n, _f = self.get_rbg_value_uv(uv2surface_points, uv_points[surface_mask], view)
            
            elif self.fixed_transform: # transform surface points to match with training dist.
                tdsp=dsp.clone()#.detach()
                # pdb.set_trace()
                '''
                # for pami data horizontal
                R=torch.Tensor([[[ 0.8698,  0.4921, -0.0368],
                        [-0.4924,  0.8607, -0.1292],
                        [-0.0319,  0.1306,  0.9909]]]).to(tdsp.device)
                T=torch.Tensor([[ 0.0150, -0.0067, -0.0920]]).to(tdsp.device)
                S=torch.Tensor([[3.2347, 3.2347]]).to(tdsp.device)
                
                # for pami data vertical
                R=torch.Tensor([[[ 0.8640,  0.4948, -0.0936],
                                [-0.5017,  0.8616, -0.0765],
                                [ 0.0427,  0.1131,  0.9927]]]).to(dsp.device)
                T=torch.Tensor([[-0.0148,  0.0171, -0.1117]]).to(dsp.device)
                S=torch.Tensor([[-3.2347, -3.2347]]).to(dsp.device)     #data6
                '''
                R90=torch.Tensor([[[ 0.,  -1., 0.],
                                [1.,  0., 0.],
                                [ 0.,  0., 1.]]]).to(dsp.device)
                Rm90=torch.Tensor([[[ 0.,  1., 0.],
                                [-1.,  0., 0.],
                                [ 0.,  0., 1.]]]).to(dsp.device)
                R30=torch.Tensor([[[ 0.866,  0.5, 0.],
                                [-0.5,  0.866, 0.],
                                [ 0.,  0., 1.]]]).to(tdsp.device)
                R25=torch.Tensor([[[ 0.90,  0.42, 0.],
                                [-0.42,  0.90, 0.],
                                [ 0.,  0., 1.]]]).to(tdsp.device) # use for book
                '''
                # for reciept
                R=torch.Tensor([[[ 0.9953,  0.0044, -0.0968],
                                [ 0.0044,  0.9960,  0.0895],
                                [ 0.0968, -0.0895,  0.9913]]]).to(tdsp.device)
                T=torch.Tensor([[ 0.0036,  0.0116, -0.0661]]).to(dsp.device)
                S=torch.Tensor([[ 4.0372, 8.5979 ]]).to(dsp.device)
                
                
                T=torch.Tensor([[0.0078,  0.0049, -0.0611]]).to(dsp.device)
                S=torch.Tensor([[4.0440]]).to(dsp.device)
                # pdb.set_trace()
                
                # for receipt2
                R=torch.Tensor([[[ 9.9972e-01, -4.4351e-04,  2.3859e-02],
                                [-4.4351e-04,  9.9931e-01,  3.7161e-02],
                                [-2.3859e-02, -3.7161e-02,  9.9902e-01]]]).to(tdsp.device)
                T=torch.Tensor([[ -0.0223,  0.0119, -0.1359]]).to(dsp.device)
                S=torch.Tensor([[ 2.4809 , 5.7925]]).to(dsp.device)
                
                # for book one scale
                R=torch.Tensor([[[ 9.9872e-01,  7.0167e-04,  5.0606e-02],
                                [ 7.0167e-04,  9.9962e-01, -2.7708e-02],
                                [-5.0606e-02,  2.7708e-02,  9.9833e-01]]]).to(tdsp.device)
                T=torch.Tensor([[ -0.0224,  0.0586, -0.0850]]).to(dsp.device)
                S=torch.Tensor([[ 4.3698, -0.6009]]).to(dsp.device)
                
                # for book two scale
                R=torch.Tensor([[[ 0.9974,  0.0015,  0.0719],
                                 [ 0.0015,  0.9991, -0.0418],
                                 [-0.0719,  0.0418,  0.9965]]]).to(tdsp.device)
                T=torch.Tensor([[ -0.02,  0.0, 0.0]]).to(dsp.device)
                S=torch.Tensor([[ 7.4547, 4.8431]]).to(dsp.device)
                
                # for book two scale (-1.5,1.5)
                R=torch.Tensor([[[ 0.9975,  0.0018,  0.0702],
                                 [ 0.0018,  0.9988, -0.0499],
                                 [-0.0702,  0.0499,  0.9963]]]).to(tdsp.device)
                '''
                R=torch.Tensor([[[ 0.9993, -0.0042, -0.0362],
                                [-0.0042,  0.9736, -0.2281],
                                [ 0.0362,  0.2281,  0.9730]]]).to(tdsp.device) # paper5, paper4, 1,2,3
                '''
                R=torch.Tensor([[[ 0.9981,  0.0054, -0.0614],
                                [ 0.0054,  0.9844,  0.1758],
                                [ 0.0614, -0.1758,  0.9825]]]).to(tdsp.device) #can1, can2
                
                R=torch.Tensor([[[ 0.9395, -0.0608,  0.3372],
                                [-0.0608,  0.9390,  0.3385],
                                [-0.3372, -0.3385,  0.8785]]]).to(tdsp.device) #tshirt1, tshirt2
                
                R=torch.Tensor([[[ 0.9851,  0.0045, -0.1717],
                                [ 0.0045,  0.9987,  0.0516],
                                [ 0.1717, -0.0516,  0.9838]]]).to(tdsp.device) # face1
                
                R=torch.Tensor([[[ 0.9953, -0.0079,  0.0970],
                        [-0.0079,  0.9870,  0.1606],
                        [-0.0970, -0.1606,  0.9822]]]).to(tdsp.device) #fabric1
                
                R=torch.Tensor([[[ 0.9696,  0.0136,  0.2444],
                                [ 0.0136,  0.9939, -0.1090],
                                [-0.2444,  0.1090,  0.9635]]]).to(tdsp.device) #fabric2
                
                R=torch.Tensor([[[ 0.9414,  0.0278,  0.3362],
                                 [ 0.0278,  0.9869, -0.1592],
                                 [-0.3362,  0.1592,  0.9282]]]).to(tdsp.device) #fabric3
                '''
                
                # T=torch.Tensor([[ -0.0963,  -0.0393, -0.1221]]).to(dsp.device) # face1
                # T=torch.Tensor([[ 0.0130,  0.0024, -0.1847]]).to(tdsp.device) #can1, can2
                # T=torch.Tensor([[ 0.1615,  -0.0223, -0.1717]]).to(tdsp.device) #tshirt1
                # T=torch.Tensor([[ -0.0109,  0.0030, -0.1441]]).to(tdsp.device) #fabric1
                # T=torch.Tensor([[0.0073, -0.0207, -0.1680]]).to(tdsp.device) #fabric2
                # T=torch.Tensor([[0.0256,  0.0195, -0.1190]]).to(tdsp.device) #fabric3
                # T=torch.Tensor([[ 0.0615,  -0.0523, -0.1717]]).to(tdsp.device) # tshirt2
                T=torch.Tensor([[ 0.0092, 0.0119, -0.1264]]).to(dsp.device) # paper5, paper4, 1,2,3
                # S=torch.Tensor([[3.4, 3.1]]).to(dsp.device) # paper4
                # S=torch.Tensor([[3.1, 3.1]]).to(dsp.device) # paper5
                S=torch.Tensor([[3.5, 3.5]]).to(dsp.device) # paper3
                # T=torch.Tensor([[-0.05,  -0.0, -0.0843]]).to(dsp.device) # book
                # S=torch.Tensor([[7.0, 3.1626]]).to(dsp.device) #book, doc_sheet
                # T=torch.Tensor([[ 0.0,  0.0119, -0.1359]]).to(dsp.device) #receipt2, doc_sheet2, doc_sheet
                # S=torch.Tensor([[ 3 , 7.2]]).to(dsp.device) #receipt2
                # S=torch.Tensor([[ 3 , 4.8]]).to(dsp.device) #doc_sheet2
                # S=torch.Tensor([[5.1553, 7.4737]]).to(tdsp.device)# can1, can2 
                # S=torch.Tensor([[-4.7606, -4.5778]]).to(tdsp.device)# tshirt1
                # S=torch.Tensor([[-3.7245, -4.3323]]).to(tdsp.device)# fabric1
                # S=torch.Tensor([[-2.3596, -5.0475]]).to(tdsp.device)# fabric2
                # S=torch.Tensor([[-3.29, -2.6977]]).to(tdsp.device)# fabric3
                # S=torch.Tensor([[ -4.5778,-7.7606]]).to(tdsp.device)# tshirt2
                # S=torch.Tensor([[8.8200, 13]]).to(dsp.device) # face1
                # S=torch.Tensor([[3.14, 3.2]]).to(dsp.device) # paper1
                # S=torch.Tensor([[3.14, 3.1]]).to(dsp.device) # paper2
                

                tdsp=torch.bmm(tdsp.unsqueeze(0),R)
                # tdsp=torch.bmm(tdsp,R30)  #use for receipt1, paper5, face1, fabric1, fabric3
                tdsp=torch.bmm(tdsp,R25)  #use for book, receipt2, doc_sheet, paper4, 1,2,3, can1, can2, tshirt1, tshirt2, fabric2
                tdsp=torch.bmm(tdsp,Rm90) #use for book, receipt2, doc_sheet, doc_sheet2, paper5, can1, can2
                # tdsp=torch.bmm(tdsp,R90) #use for Data6, tshirt1, tshirt2, fabric1, fabric2,fabric3
                tdsp=tdsp+T
                # tdsp=tdsp*S
                S_absolute_local=torch.eye(3,dtype=torch.float32, requires_grad=True).cuda()
                # S_absolute_relu=relu(S_absolute)
                S_absolute_local[0,0]=S[:,0]
                S_absolute_local[1,1]=S[:,1]

                tdsp=torch.bmm(tdsp,S_absolute_local.unsqueeze(0))
                
                tdsp=tdsp.squeeze(0)
                
                if self.uv_inp:
                    uv_points[surface_mask]=self.forward_network(tdsp, image_uv_inp, global_feats=None)
                else:
                    uv_points[surface_mask]=self.forward_network(tdsp, None , global_feats=None)
                
                fwdrgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value_uv(dsp, uv_points[surface_mask], view)
                # fwdrgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value_uv(dsp, uv_points[surface_mask].clone().detach(), view)
                normals[surface_mask] = diff_normals
                
                if self.train_backward:
                    uv2surface_points = torch.zeros_like(differentiable_surface_points).float().cuda()
                    # uv points to backward network for surface points
                    if self.joint_backward:
                        uv2surface_points=self.backward_network(uv_points[surface_mask], None, global_feats=None)
                    else:
                        uv2surface_points=self.backward_network(uv_points[surface_mask].clone().detach(), None, global_feats=None)

                    if self.isometry:
                        uvp=uv_points[surface_mask].clone().detach()
                        # uvsurf=self.backward_network(uvp, None, global_feats=None)
                        # make_dot(uvsurf).render("attached", format="png")
                        # zzzz=self.diffprops(uvsurf, uvp)
                        randuvp=torch.rand([uv2surface_points.shape[0],2], requires_grad=True).to(uv2surface_points.device)
                        uvp=torch.cat([uvp, randuvp], dim=0).unsqueeze(0)
                        uvsurf=self.backward_network(uvp, None, global_feats=None)
                        # make_dot(uvsurf).render("attached2", format="png")
                        diff_prop=self.diffprops(uvsurf, uvp)
                        
                    # bwdrgb_values[surface_mask], _n, _f = self.get_rbg_value_uv(uv2surface_points, uv_points[surface_mask].clone().detach(), view)
            else:
                tdsp=dsp.clone()
                if self.uv_inp:
                    uv_points[surface_mask]=self.forward_network(dsp, image_uv_inp, global_feats=None)
                else:
                    uv_points[surface_mask]=self.forward_network(dsp, None , global_feats=None)
                
                if self.uv_noise:
                    # noise = 0.0
                    # # noise = (
                    # #     torch.randn(
                    # #         uv_points[surface_mask].shape,
                    # #         dtype=uv_points.dtype,
                    # #         device=uv_points.device,
                    # #     )
                    # #     * 0.001
                    # # )
                    # # uv_points[surface_mask]+=noise
                    ps=uv_points[surface_mask].shape[0]
                    x=torch.linspace(0,0.5,ps, dtype=uv_points.dtype, device=uv_points.device)
                    x=x[torch.randperm(len(x))]
                    y=torch.linspace(0,0.5,ps, dtype=uv_points.dtype, device=uv_points.device)
                    y=y[torch.randperm(len(y))]
                    uv_points[surface_mask][:,0]+=x
                    uv_points[surface_mask][:,1]+=y
                    
                fwdrgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value_uv(dsp, uv_points[surface_mask].clone().detach(), view)
                normals[surface_mask] = diff_normals
                
                
                if self.train_backward:
                    uv2surface_points = torch.zeros_like(differentiable_surface_points, requires_grad=True).float().cuda()
                    # uv points to backward network for surface points
                    if self.joint_backward:
                        uv2surface_points=self.backward_network(uv_points[surface_mask], None, global_feats=None)
                    else:
                        uv2surface_points=self.backward_network(uv_points[surface_mask].clone().detach(), None, global_feats=None)
                    
                    if self.isometry:
                        uvp=uv_points[surface_mask].clone().detach()
                        # uvsurf=self.backward_network(uvp, None, global_feats=None)
                        # make_dot(uvsurf).render("attached", format="png")
                        # zzzz=self.diffprops(uvsurf, uvp)
                        randuvp=torch.rand([uv2surface_points.shape[0],2], requires_grad=True).to(uv2surface_points.device)
                        uvp=torch.cat([uvp, randuvp], dim=0).unsqueeze(0)
                        uvsurf=self.backward_network(uvp, None, global_feats=None)
                        # make_dot(uvsurf).render("attached2", format="png")
                        diff_prop=self.diffprops(uvsurf, uvp)
                        
                    # bwdrgb_values[surface_mask], _n, _f = self.get_rbg_value_uv_norm_feat(uv2surface_points, 
                    #                                                                     uv_points[surface_mask].clone().detach(), 
                    #                                                                     view, diff_normals, diff_feats)
                
        output = {
            'points': points,
            'surface_points': surface_points, # these are the ray surface intersection points 
            'diff_surface_points': differentiable_surface_points,
            'diff_normals': diff_normals, # normals at differentiable surface points
            'rgb_values': fwdrgb_values,
            # 'bwdrgb_values': bwdrgb_values, 
            'normals': normals,
            'sdf_output': sdf_output,
            'surface_output': surface_output, # sdf values at the ray surface intersection points
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'grad_points': surface_points_grad, # normals at the ray surface intersection points
            'grad_eikpoints': grad_eikonal_points,
            'uv_points': uv_points,
            'uv2surface_points':uv2surface_points,
            'tr_diff_surface_points': tdsp,
            'diff_props':diff_prop
        }
        return output
    
    def infer_fwd_bwd(self, input, image_uv_inp=None):
        
        uv=input['uv'].cuda()
        pose=input['pose'].cuda()
        intrinsics=input['intrinsics'].cuda()
        points=input['points'].cuda()
        norm_points=input['norm_points'].cuda()
        surface_mask=input['mask'].cuda()
        
        rgb_values = torch.ones_like(points).float().cuda()
        normals = torch.ones_like(points).float().cuda()
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        # ray_dirs = ray_dirs.reshape(-1, 3)
        view = -ray_dirs[surface_mask]
        dsp=points[surface_mask]
        norm_dsp=norm_points[surface_mask]
        if self.transform and norm_dsp.shape[0]>0:
            # pdb.set_trace()
            norm_dsp=norm_dsp.unsqueeze(0).transpose(2,1)
            stnmat= self.stn(norm_dsp)
            norm_dsp=norm_dsp.transpose(2,1)
            norm_dsp = torch.bmm(norm_dsp, stnmat).squeeze(0)
            
        uv_points = torch.ones(points.shape[0],points.shape[1],2).float().cuda() * -1
        # if self.uv_inp:
        #         uv_points[surface_mask]=self.forward_network(dsp, image_uv_inp, global_feats=None)
        #     else:
        # rendering network should take unnormalized surface points
        uv_points[surface_mask]=self.forward_network(norm_dsp, None , global_feats=None)
        rgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value_uv(dsp, uv_points[surface_mask].clone().detach(), view)
        normals[surface_mask] = diff_normals
        output={
            'uv_points': uv_points,
            'rgb':rgb_values,
            'normals': normals
        }
        
        return output
    
    def get_rbg_value_uv(self, points, uv, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.renderinguv_network(uv, points, normals, view_dirs, feature_vectors)

        return rgb_vals, normals, feature_vectors
    
    def get_rbg_value_uv_norm_feat(self, points, uv, view_dirs, normals, feature_vectors):
        rgb_vals = self.renderinguv_network(uv, points, normals, view_dirs, feature_vectors)

        return rgb_vals, normals, feature_vectors



class IDRBWDUVRendNetwork(nn.Module):
    # this network takes uv prediction and the 3d surface (xyz) both as input to the rendering network
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.train_backward = conf.get_bool('train_backward')
        self.backprop_fwd = conf.get_bool('backprop_fwd')
        self.joint_backward = conf.get_bool('joint_backward')
        # self.normalize = conf.get_bool('normalize')
        self.uv_inp=conf.get_bool('uv_inp')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.renderinguv_network = RenderingUVNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.forward_network = UVNetwork(3, **conf.get_config('forward_network'))
        self.backward_network = UVNetwork(2, **conf.get_config('backward_network'))
        
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input):
        # pdb.set_trace()
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        uv_inp = input["uv_inp"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                cam_loc=cam_loc,
                                                                object_mask=object_mask,
                                                                ray_directions=ray_dirs)
        self.implicit_network.train()
        # pdb.set_trace()
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3) # all points along the ray

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            # pdb.set_trace()
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]  # ray intersection points
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]  # sdf output of the raytraced points
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0) # random points within range + all points along the ray

            points_all = torch.cat([surface_points, eikonal_points], dim=0) # ray intersection points , random points within range + all points along the ray

            output = self.implicit_network(surface_points)  # evaluate implicit function on ray intersection points
            surface_sdf_values = output[:N, 0:1].detach()

            # calculate gradient wrt ray intersection points
            # random points within range + all points along the ray
            g = self.implicit_network.gradient(points_all) 
            
            # pdb.set_trace()
            surface_points_grad = g[:N, 0, :]
            grad_theta = g[N:, 0, :]
            grad_eikonal_points = g[N:n_eik_points, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad.clone().detach(),
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)
        
        else:
            surface_mask = network_object_mask & object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
            surface_points_grad=None
            surface_output=None 
            grad_eikonal_points=None
            surface_points=None
            diff_normals=None

        view = -ray_dirs[surface_mask]

        # fwdrgb_values = torch.ones_like(points).float().cuda()
        # bwdrgb_values = torch.ones_like(points).float().cuda()
        normals = torch.ones_like(points).float().cuda()
        uv_points = torch.ones(points.shape[0],2).float().cuda() * -1
        uv2surface_points=None
        if differentiable_surface_points.shape[0] > 0:
            # surface points to forward network for uv points
            image_uv_inp=uv_inp.reshape(-1,2)[surface_mask]
            # pdb.set_trace()
            if self.backprop_fwd:
                dsp=differentiable_surface_points
            else:
                dsp=differentiable_surface_points.clone().detach()
                
            # pdb.set_trace()
            
            if self.uv_inp:
                uv_points[surface_mask]=self.forward_network(dsp, image_uv_inp, global_feats=None)
            else:
                uv_points[surface_mask]=self.forward_network(dsp, None , global_feats=None)
            
            
            if self.train_backward:
                uv2surface_points = torch.zeros_like(differentiable_surface_points).float().cuda()
                # uv points to backward network for surface points
                if self.joint_backward:
                    uv2surface_points=self.backward_network(uv_points[surface_mask], None, global_feats=None)
                else:
                    uv2surface_points=self.backward_network(uv_points[surface_mask].clone().detach(), None, global_feats=None)
                    
                # bwdrgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value_uv(uv2surface_points, uv_points[surface_mask].clone().detach(), view)
                normals[surface_mask] = diff_normals
            
            
        output = {
            'points': points,
            'surface_points': surface_points, # these are the ray surface intersection points 
            'diff_surface_points': differentiable_surface_points,
            'diff_normals': diff_normals, # normals at differentiable surface points
            # 'rgb_values': bwdrgb_values,
            # 'bwdrgb_values': None, 
            'normals': normals,
            'sdf_output': sdf_output,
            'surface_output': surface_output, # sdf values at the ray surface intersection points
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'grad_points': surface_points_grad, # normals at the ray surface intersection points
            'grad_eikpoints': grad_eikonal_points,
            'uv_points': uv_points,
            'uv2surface_points':uv2surface_points
        }
        return output
    
    def get_rbg_value_uv(self, points, uv, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.renderinguv_network(uv, points, normals, view_dirs, feature_vectors)

        return rgb_vals, normals, feature_vectors


class IDRBWDUVDiffRendNetwork(nn.Module):
    # this network takes uv prediction and the 3d surface (xyz) both as input to the rendering network
    # uses metric tensor for normal computation as gradient of the backward network.
    # normal computed as the gradient of the backward network should be consistent with the IDR 
    # IDR computed normals.
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.train_backward = conf.get_bool('train_backward')
        self.backprop_fwd = conf.get_bool('backprop_fwd')
        self.joint_backward = conf.get_bool('joint_backward')
        # self.normalize = conf.get_bool('normalize')
        self.uv_inp=conf.get_bool('uv_inp')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.renderinguv_network = RenderingUVNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.forward_network = UVNetwork(3, **conf.get_config('forward_network'))
        self.backward_network = UVNetwork(2, **conf.get_config('backward_network'))
        
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.diffpro=diff_props.DiffGeomProps(normals=True, curv_mean=False, curv_gauss=False, fff=False)

    def forward(self, input):
        # pdb.set_trace()
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        uv_inp = input["uv_inp"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                cam_loc=cam_loc,
                                                                object_mask=object_mask,
                                                                ray_directions=ray_dirs)
        self.implicit_network.train()
        # pdb.set_trace()
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3) # all points along the ray

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            # pdb.set_trace()
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]  # ray intersection points
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]  # sdf output of the raytraced points
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0) # random points within range + all points along the ray

            points_all = torch.cat([surface_points, eikonal_points], dim=0) # ray intersection points , random points within range + all points along the ray

            output = self.implicit_network(surface_points)  # evaluate implicit function on ray intersection points
            surface_sdf_values = output[:N, 0:1].detach()

            # calculate gradient wrt ray intersection points
            # random points within range + all points along the ray
            g = self.implicit_network.gradient(points_all) 
            
            # pdb.set_trace()
            surface_points_grad = g[:N, 0, :]
            grad_theta = g[N:, 0, :]
            grad_eikonal_points = g[N:n_eik_points, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad.clone().detach(),
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)
        
        else:
            surface_mask = network_object_mask & object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
            surface_points_grad=None
            surface_output=None 
            grad_eikonal_points=None
            surface_points=None
            diff_normals=None

        view = -ray_dirs[surface_mask]

        # fwdrgb_values = torch.ones_like(points).float().cuda()
        bwdrgb_values = torch.ones_like(points).float().cuda()
        normals = torch.ones_like(points).float().cuda()
        uv_points = torch.ones(points.shape[0],2).float().cuda() * -1
        uv2surface_points=None
        if differentiable_surface_points.shape[0] > 0:
            # surface points to forward network for uv points
            image_uv_inp=uv_inp.reshape(-1,2)[surface_mask]
            # pdb.set_trace()
            if self.backprop_fwd:
                dsp=differentiable_surface_points
            else:
                dsp=differentiable_surface_points.clone().detach()
                
            # pdb.set_trace()
            
            if self.uv_inp:
                uv_points[surface_mask]=self.forward_network(dsp, image_uv_inp, global_feats=None)
            else:
                uv_points[surface_mask]=self.forward_network(dsp, None , global_feats=None)
            
            
            if self.train_backward:
                uv2surface_points = torch.zeros_like(differentiable_surface_points).float().cuda()
                # uv points to backward network for surface points
                if self.joint_backward:
                    uv2surface_points=self.backward_network(uv_points[surface_mask], None, global_feats=None)
                else:
                    uv=uv_points[surface_mask].clone().detach()
                    uv.requires_grad=True
                    uv2surface_points=self.backward_network(uv, None, global_feats=None)
                    
                bwdrgb_values[surface_mask], diff_normals, diff_feats = self.get_rbg_value_uv(uv2surface_points, uv_points[surface_mask].clone().detach(), view)
                normals[surface_mask] = diff_normals
            
            
        output = {
            'points': points,
            'surface_points': surface_points, # these are the ray surface intersection points 
            'diff_surface_points': differentiable_surface_points,
            'diff_normals': diff_normals, # normals at differentiable surface points
            'rgb_values': bwdrgb_values,
            'bwdrgb_values': None, 
            'normals': normals,
            'sdf_output': sdf_output,
            'surface_output': surface_output, # sdf values at the ray surface intersection points
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'grad_points': surface_points_grad, # normals at the ray surface intersection points
            'grad_eikpoints': grad_eikonal_points,
            'uv_points': uv_points,
            'uv2surface_points':uv2surface_points
        }
        return output
    
    def get_rbg_value_uv(self, points, uv, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]
        if self.training:
            pdb.set_trace()
            uv2surface_points=self.backward_network(uv, None, global_feats=None)
            diffproret= self.diffpro(uv2surface_points.transpose(1,0).unsqueeze(0), uv.unsqueeze(0)) # points: B,3,N uv: B,N,2
            pdb.set_trace()
        feature_vectors = output[:, 1:]
        rgb_vals = self.renderinguv_network(uv, points, normals, view_dirs, feature_vectors)

        return rgb_vals, normals, feature_vectors

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def sine_init_ten(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 10, np.sqrt(6 / num_input) / 10)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            
def final_layer_init_uniform(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(0, 1 / num_input)
            m.bias.zero_()
            
def final_layer_init_normal(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.normal_(0.5, 1 /(12* num_input))
        '''
        if hasattr(m, 'bias'):
            m.bias.zero_()
        '''