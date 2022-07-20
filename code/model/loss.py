import torch
from torch import nn
from torch.nn import functional as F
import pdb
import robust_loss_pytorch.general
import pytorch3d.loss as p3dloss

class IDRLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight, alpha, 
                lipschitz_const, lipschitz_weight, lipschitz_points, 
                lipschitz_mode, apply_normal_loss, normal_weight, radius, pnorm, delD):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.lipschitz_const=lipschitz_const
        self.lipschitz_weight=lipschitz_weight
        self.lipschitz_mode=lipschitz_mode
        self.lipschitz_points=lipschitz_points
        self.apply_normal_loss=apply_normal_loss
        self.normal_weight= normal_weight
        self.r=radius
        self.p=pnorm
        self.delD=delD
        self.l1_loss = nn.L1Loss(reduction='sum')

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_lipschitz_loss(self, grad_theta, mode='mean'):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()
        # pdb.set_trace()
        if mode=='max':
            lip_loss=torch.max(torch.abs(grad_theta).max()-self.lipschitz_const, torch.Tensor([0.0]).to(grad_theta.device))
        else:
            lip_loss=torch.max(torch.abs(grad_theta).mean()-self.lipschitz_const, torch.Tensor([0.0]).to(grad_theta.device))
        return lip_loss.mean()

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss
    
    def get_normal_loss(self, surf_points, normals, decay, p, delD):
        
        if len(surf_points.shape)==2:
            surf_points=surf_points.unsqueeze(0)
            normals=normals.unsqueeze(0)
        # calc distance
        dist=self.distance_matrix(surf_points, surf_points)
        # pdb.set_trace()
        # calculate normal distance as p norm
        norm_pdist=self.p_distance_matrix(normals, normals, p=p)
        if delD is not None:
            # use delD to only take points within delD radius
            # normalize the weights as well# 
            # pdb.set_trace()
            dist_w = torch.where(dist >= delD, torch.zeros_like(dist), torch.exp(-decay*dist))
            norm_pdist_w=norm_pdist*dist_w
        else:
            # weight the norm distances
            norm_pdist_w=norm_pdist*torch.exp(-decay*dist)
        
        return norm_pdist_w.sum(dim=-1).mean()
    
    def distance_matrix(self, pc_N, pc_M):
        """ Computes a distance matrix between two point sets.
        Args:
            pc_N (torch.Tensor): shape (B, N, 3)
            pc_M (torch.Tensor): shape (B, M, 3)
        Returns:
            Distance matrix, shape (B, M, N).
        """
        eps=1e-6
        # Get per-point distance matrix.
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape

        assert (B == B2)
        assert (D == D2)
        assert (D == 3)

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))

        # return (x - y).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N) # leads to nan gradient for (0).sqrt()
        return ((x - y)+eps).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N) # leads to nan gradient for (0).sqrt()
        
    
    def p_distance_matrix(self, pc_N, pc_M, p=2):
        """ Computes a p norm distance matrix between two point sets.
        Args:
            pc_N (torch.Tensor): shape (B, N, 3)
            pc_M (torch.Tensor): shape (B, M, 3)
        Returns:
            Distance matrix, shape (B, M, N).
        """
        eps=1e-6
        # Get per-point distance matrix.
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape

        assert (B == B2)
        assert (D == D2)
        assert (D == 3)

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))

        return (torch.abs(x - y)+eps).pow(p).sum(dim=3).pow(1/p)  # (B, M, N, 3) -> (B, M, N)

    def forward(self, model_outputs, ground_truth):
        # pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss 
        
        lipsch_loss=None
        if self.lipschitz_weight>0.0:
            if self.lipschitz_points=='random':
                lipsch_loss = self.get_lipschitz_loss(model_outputs['grad_eikpoints'], self.lipschitz_mode)
            else:
                lipsch_loss = self.get_lipschitz_loss(model_outputs['grad_points'], self.lipschitz_mode)
            # pdb.set_trace()
            loss+=self.lipschitz_weight * lipsch_loss

        normal_loss=None
        if self.apply_normal_loss:
            normal_loss =self.get_normal_loss(model_outputs['diff_surface_points'],model_outputs['diff_normals'], self.r, self.p, self.delD)
            loss = loss +  (self.normal_weight * normal_loss)
        
        
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'lip_loss': lipsch_loss,
            'mask_loss': mask_loss,
            'normal_loss': normal_loss
        }
        


class IDRUVLoss(nn.Module):
    def __init__(self,rgb_mode, eikonal_weight, mask_weight, alpha, 
                lipschitz_const, lipschitz_weight, lipschitz_points, 
                lipschitz_mode, apply_normal_loss, normal_weight, radius, 
                fwd_weight, fwd_rgb_weight, bwd_weight, bwd_mode, pnorm, delD, 
                uvrgb_mode, scinv_lambda, rgbcyc_weight, isom_weight, isom_mode):
        super().__init__()
        self.rgb_mode=rgb_mode
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.lipschitz_const=lipschitz_const
        self.lipschitz_weight=lipschitz_weight
        self.lipschitz_mode=lipschitz_mode
        self.lipschitz_points=lipschitz_points
        self.apply_normal_loss=apply_normal_loss
        self.normal_weight= normal_weight
        self.uv_weight =fwd_weight
        self.fwd_rgb_weight=fwd_rgb_weight
        self.pt_weight =bwd_weight
        self.isom_weight=isom_weight
        self.isom_mode=isom_mode
        self.r=radius
        self.p=pnorm
        self.delD=delD
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss()
        self.l2_loss_per_elem = nn.MSELoss(reduction='none')
        self.l1_loss_per_elem = nn.L1Loss(reduction='none')
        self.rgbloss_mode =uvrgb_mode
        self.scinv_lambda=scinv_lambda
        self.rgb_cycle_weight= rgbcyc_weight
        self.bwd_mode=bwd_mode

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_weighted_rgb_loss(self,rgb_values, rgb_gt, weight, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        # pdb.set_trace()
        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        weight=weight.reshape(-1)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss_per_elem(rgb_values, rgb_gt) 
        rgb_loss = weight*rgb_loss.sum(dim=-1)
        rgb_loss= rgb_loss.sum()/float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_lipschitz_loss(self, grad_theta, mode='mean'):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()
        # pdb.set_trace()
        if mode=='max':
            lip_loss=torch.max(torch.abs(grad_theta).max()-self.lipschitz_const, torch.Tensor([0.0]).to(grad_theta.device))
        else:
            lip_loss=torch.max(torch.abs(grad_theta).mean()-self.lipschitz_const, torch.Tensor([0.0]).to(grad_theta.device))
        return lip_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss
    '''
    def get_uv_loss(self, uv_outputs, uv_gt, rgbalb_gt, texture_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        uv_loss={}
        # pdb.set_trace()
        uv_values_ = uv_outputs[network_object_mask & object_mask]
        uv_gt_= uv_gt.reshape(-1, 2)[network_object_mask & object_mask]
        # pdb.set_trace()
        uvpt_loss= self.l1_loss(uv_values_, uv_gt_) / float(object_mask.shape[0])
        uv_loss['uvpt_loss']=uvpt_loss #+ torch.abs(uv_values_.mean()-0.5)
        if self.fwd_rgb_weight>0.0:
            # pdb.set_trace()
            rgbalb_gt = rgbalb_gt.reshape(-1, 3)[network_object_mask & object_mask]
            # normalize uv_values in range [-1,1]
            uv_values=(2.0*uv_values_)-1
            # sample the gt texture using predicted uv values
            # texture_gt (B,3,h,w)
            # uv_values (B,1,n,2) n= number of surface points
            # rgbalb_gt (B,3,1,n)
            # rgbalb_pred (B,3,1,n)
            rgbalb_pred=F.grid_sample(texture_gt, uv_values.unsqueeze(0).unsqueeze(0), mode='bilinear')
            rgbalb_pred=rgbalb_pred.squeeze().transpose(1,0)
            uvrgb_loss = self.l1_loss(rgbalb_pred, rgbalb_gt) / float(object_mask.shape[0])
            uv_loss['uvrgb_loss']=uvrgb_loss
        else:
            uv_loss['uvrgb_loss']=None
        return uv_loss
    '''
    def get_uv_loss(self, uv_outputs, uv_gt, network_object_mask, object_mask):
        # pdb.set_trace()
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        mask=network_object_mask & object_mask
        uv_values_ = uv_outputs[mask]
        # uv_values_=torch.index_select(uv_values_, -1, torch.LongTensor([1,0]).to(uv_values_.device))
        uv_gt_= uv_gt.reshape(-1, 2)[mask]
        # pdb.set_trace()
        uvpt_loss= self.l1_loss(uv_values_, uv_gt_) / float(object_mask.shape[0])
        uv_loss=uvpt_loss #+ torch.abs(uv_values_.mean()-0.5)+ torch.abs(uv_values_.var()-(1/12.0))
        return uv_loss
    
    def get_uv_reg_loss(self, uv_outputs, uv_pts, network_object_mask, object_mask):
        # pdb.set_trace()
        '''
        uv_pts: random points sampled from uv space
        '''
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        mask=network_object_mask & object_mask
        uv_values_ = uv_outputs[mask]
        # pdb.set_trace()
        # uv_loss=torch.abs(uv_values_.mean()-0.5)+ torch.abs(uv_values_.var()-(1/12.0))
        uv_loss,_=p3dloss.chamfer_distance(uv_values_.unsqueeze(0), uv_pts)
        return uv_loss
    
    def get_uvrgb_loss(self, uv_outputs, rgbalb_gt, texture_gt, network_object_mask, object_mask, mode='l2', lambda_=0.5):
        # pdb.set_trace()
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        mask=network_object_mask & object_mask
        uv_values_ = uv_outputs[mask]
        # uv_values_=torch.index_select(uv_values_, -1, torch.LongTensor([1,0]).to(uv_values_.device))
        rgbalb_gt = rgbalb_gt.reshape(-1, 3)[mask]
        # normalize uv_values in range [-1,1]
        uv_values=(2.0*uv_values_)-1
        # sample the gt texture using predicted uv values
        # texture_gt (B,3,h,w)
        # uv_values (B,1,n,2) n= number of surface points
        # rgbalb_gt (B,3,1,n)
        # rgbalb_pred (B,3,1,n)
        rgbalb_pred=F.grid_sample(texture_gt, uv_values.unsqueeze(0).unsqueeze(0), mode='bilinear')
        rgbalb_pred=rgbalb_pred.squeeze().transpose(1,0)
        if mode=='l2':
            uvrgb_loss = self.l2_loss(rgbalb_pred, rgbalb_gt) / float(object_mask.shape[0])
        else:
            uvrgb_loss = self.scaleinv_loss(rgbalb_pred, rgbalb_gt, object_mask, lambda_)
        return uvrgb_loss
    
    # def get_uvrgbcycle_loss(self, uv_outputs, uv2surface_outputs, network_object_mask, texture_gt, pose, intrinsics, object_mask):
    #     if (network_object_mask & object_mask).sum() == 0:
    #         return torch.tensor(0.0).cuda().float()
    #     mask=network_object_mask & object_mask
    #     uv_values_ = uv_outputs[mask]
    #     # sample the gt texture using predicted uv values
    #     # texture_gt (B,3,h,w)
    #     # uv_values_ (B,1,n,2) n= number of surface points
    #     # rgb_pred (B,3,1,n)
    #     rgb_pred=F.grid_sample(texture_gt, uv_values.unsqueeze(0).unsqueeze(0), mode='bilinear')
        
    #     # project uv2surface to 2d pixel locations
        
        
    def scaleinv_loss(self,pred, gt, object_mask, lambda_=1.0):
        # pdb.set_trace()
        n=object_mask.shape[0]
        d=pred-gt
        di= torch.sum(d**2, dim=0)/n
        di2=(lambda_/(n*n))*(torch.sum(d, dim=0)**2)
        return (di-di2).sum()
    
    def get_point_loss(self,uv2surface_points, surface_points):
        pt_loss = self.l2_loss(uv2surface_points, surface_points)
        return pt_loss
    
    def get_weighted_point_loss(self,uv2surface_points, surface_points, weights):
        pt_loss = self.l2_loss_per_elem(uv2surface_points, surface_points)
        pt_loss=weights*pt_loss.sum(dim=-1)
        # pdb.set_trace()
        return pt_loss.mean()
    
    def get_isom_loss(self, diff_props, mode='scisom'):
        self._zero = torch.tensor(0.).to(diff_props['fff'].device)
        self._one = torch.tensor(1.).to(diff_props['fff'].device)
        self._eps = torch.tensor(1e-6)
        E, F, G = diff_props['fff'].permute(2, 0, 1) 
        
        # pdb.set_trace()
        # Get per point local squared area.
        A2 = torch.max(E * G - F.pow(2), self._zero)  # (B, P)
        # A = A2.sqrt()  # (B, P, spp)
        
        if mode=='scisom':
            muE = E.mean()
            muG = G.mean()
        else:
            muE =self._one
            muG =self._one
            
        L_E = ((E - muE).pow(2)).mean() #/ (A2 + self._eps)).mean() 
        L_G = ((G - muG).pow(2)).mean() #/ (A2 + self._eps)).mean()
        L_F = (F.pow(2)).mean() #/ (A2 + self._eps)).mean()
        
        return L_E, L_G, L_F
    
    def get_normal_loss(self, surf_points, normals, decay, p, delD):
        if len(surf_points.shape)==2:
            surf_points=surf_points.unsqueeze(0)
            normals=normals.unsqueeze(0)
        # calc distance
        dist=self.distance_matrix(surf_points, surf_points)
        # pdb.set_trace()
        # calculate normal distance as p norm
        norm_pdist=self.p_distance_matrix(normals, normals, p=p)
        if delD is not None:
            # use delD to only take points within delD radius
            # normalize the weights as well# 
            # pdb.set_trace()
            dist_w = torch.where(dist >= delD, torch.zeros_like(dist), torch.exp(-decay*dist))
            norm_pdist_w=norm_pdist*dist_w
        else:
            # weight the norm distances
            norm_pdist_w=norm_pdist*torch.exp(-decay*dist)
        
        return norm_pdist_w.sum(dim=-1).mean()
    
    def distance_matrix(self, pc_N, pc_M):
        """ Computes a distance matrix between two point sets.
        Args:
            pc_N (torch.Tensor): shape (B, N, 3)
            pc_M (torch.Tensor): shape (B, M, 3)
        Returns:
            Distance matrix, shape (B, M, N).
        """
        eps=1e-6
        # Get per-point distance matrix.
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape

        assert (B == B2)
        assert (D == D2)
        assert (D == 3)

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))

        # return (x - y).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N) # leads to nan gradient for (0).sqrt()
        return ((x - y)+eps).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N) # leads to nan gradient for (0).sqrt()
        
    
    def p_distance_matrix(self, pc_N, pc_M, p=2):
        """ Computes a p norm distance matrix between two point sets.
        Args:
            pc_N (torch.Tensor): shape (B, N, 3)
            pc_M (torch.Tensor): shape (B, M, 3)
        Returns:
            Distance matrix, shape (B, M, N).
        """
        eps=1e-6
        # Get per-point distance matrix.
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape

        assert (B == B2)
        assert (D == D2)
        assert (D == 3)

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))

        return (torch.abs(x - y)+eps).pow(p).sum(dim=3).pow(1/p)  # (B, M, N, 3) -> (B, M, N)

    def forward(self, model_outputs, ground_truth):
        # pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        rgbalb_gt = ground_truth['alb'].cuda()
        texture_gt = ground_truth['tex'].cuda()
        uv_gt= ground_truth['uv_gt'].cuda()
        uv_pts= ground_truth['uv_pts'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']
        imp_map = ground_truth['imp_map'].cuda()
        rgb_weights=ground_truth['rgb_weight'].cuda()
        uv_outputs= model_outputs['uv_points']
        uv2surf_points= model_outputs['uv2surface_points']
        diff_props=model_outputs['diff_props']
        
        if self.rgb_mode=='wl2':
            rgb_loss = self.get_weighted_rgb_loss(model_outputs['rgb_values'], rgb_gt, rgb_weights, network_object_mask, object_mask)
        else:
            rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
            
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        # if self.lipschitz_points=='random':
        #     lipsch_loss = self.get_lipschitz_loss(model_outputs['grad_eikpoints'], self.lipschitz_mode)
        # else:
        #     lipsch_loss = self.get_lipschitz_loss(model_outputs['grad_points'], self.lipschitz_mode)

        loss = rgb_loss + \
                self.eikonal_weight * eikonal_loss + \
                self.mask_weight * mask_loss 
                # self.lipschitz_weight * lipsch_loss
        
        # get uvloss
        uv_loss=None
        if uv_gt is not None and self.uv_weight > 0.0:
            uv_loss= self.get_uv_loss(uv_outputs, uv_gt, network_object_mask, object_mask)
            uvr_loss= self.get_uv_reg_loss(uv_outputs, uv_pts, network_object_mask, object_mask)
            loss +=self.uv_weight * uvr_loss
            
        uvrgb_loss=None 
        if self.fwd_rgb_weight>0.0:
            uvrgb_loss=self.get_uvrgb_loss(uv_outputs, rgbalb_gt, texture_gt, network_object_mask, object_mask, mode=self.rgbloss_mode, lambda_=self.scinv_lambda)
            loss=self.fwd_rgb_weight *uvrgb_loss

        normal_loss=None
        if self.apply_normal_loss:
            normal_loss =self.get_normal_loss(model_outputs['diff_surface_points'],model_outputs['diff_normals'], self.r, self.p, self.delD)
            loss = loss +  (self.normal_weight * normal_loss)
        pt_loss=None
        if uv2surf_points is not None:
            if self.bwd_mode=='wl2':
                mask=network_object_mask & object_mask
                ptloss_weight=imp_map.reshape(-1)[mask]
                # ptloss_weight=ptloss_weight/ptloss_weight.sum()
                # pdb.set_trace()
                pt_loss= self.get_weighted_point_loss(uv2surf_points, model_outputs['diff_surface_points'].clone().detach(), weights=ptloss_weight)
            else:
                pt_loss= self.get_point_loss(uv2surf_points, model_outputs['diff_surface_points'].clone().detach())
            
            loss = loss +  (self.pt_weight * pt_loss)  
        # rgb_cycle_loss=None
        # if self.rgb_cycle_weight > 0.0:
        #     rgb_cycle_loss = self.get_rgb_loss(model_outputs['bwdrgb_values'], rgb_gt , network_object_mask, object_mask)
        #     loss +=  self.rgb_cycle_weight * rgb_cycle_loss
        
        LE=None
        LF=None
        LG=None
        if len(self.isom_weight)>0:
            LE, LG, LF=self.get_isom_loss(diff_props, mode=self.isom_mode) # scisom : scaled isometry (conformal)
            isom_loss=(self.isom_weight[0]*LE)+(self.isom_weight[1]*LF)+(self.isom_weight[2]*LG)
            # pdb.set_trace()
            loss+=isom_loss
        
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            # 'lip_loss': lipsch_loss,
            'mask_loss': mask_loss,
            'normal_loss': normal_loss,
            'fwd_loss': uv_loss,
            'bwd_loss': pt_loss,
            'fwrgb_loss':uvrgb_loss,
            # 'rgbcyc_loss':rgb_cycle_loss,
            'L_E':LE,
            'L_F':LF,
            'L_G':LG,
            
        }
        
class UVFieldLoss(nn.Module):
    def __init__(self, fwd_rgb_weight,fwd_weight, bwd_weight, rgbloss, scinv_lambda):
        super().__init__()
        self.uv_weight =fwd_weight
        self.fwd_rgb_weight=fwd_rgb_weight
        self.pt_weight =bwd_weight
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss()
        self.rgbloss_mode=rgbloss
        self.scinv_lambda=scinv_lambda
    
    def get_uv_loss(self, uv_outputs, uv_gt, object_mask):
        # pdb.set_trace()
        uv_values_ = uv_outputs[object_mask]
        # uv_values_=torch.index_select(uv_values_, -1, torch.LongTensor([1,0]).to(uv_values_.device))
        uv_gt_= uv_gt.reshape(-1, 2)[object_mask]
        # pdb.set_trace()
        uvpt_loss= self.l1_loss(uv_values_, uv_gt_) / float(object_mask.shape[0])
        uv_loss=uvpt_loss #+ torch.abs(uv_values_.mean()-0.5)+ torch.abs(uv_values_.var()-(1/12.0))
        return uv_loss
    
    def get_uvrgb_loss(self, uv_outputs, rgbalb_gt, texture_gt, object_mask, mode='l2', lambda_=0.5):
        # pdb.set_trace()
        uv_values_ = uv_outputs[object_mask]
        # uv_values_=torch.index_select(uv_values_, -1, torch.LongTensor([1,0]).to(uv_values_.device))
        rgbalb_gt = rgbalb_gt.reshape(-1, 3)[object_mask]
        # normalize uv_values in range [-1,1]
        uv_values=(uv_values_-0.5)*2.0
        # sample the gt texture using predicted uv values
        # texture_gt (B,3,h,w)
        # uv_values (B,1,n,2) n= number of surface points
        # rgbalb_gt (B,3,1,n)
        # rgbalb_pred (B,3,1,n)
        rgbalb_pred=F.grid_sample(texture_gt, uv_values.unsqueeze(0).unsqueeze(0), mode='bilinear')
        rgbalb_pred=rgbalb_pred.squeeze().transpose(1,0)
        if mode=='l2':
            uvrgb_loss = self.l2_loss(rgbalb_pred, rgbalb_gt) #/ float(object_mask.shape[0])
        else:
            uvrgb_loss = self.scaleinv_loss(rgbalb_pred, rgbalb_gt, lambda_)
        return uvrgb_loss
    
    def scaleinv_loss(self,pred, gt, lambda_=1.0):
        # pdb.set_trace()
        n=pred.shape[0]
        d=pred-gt
        di= torch.sum(d**2, dim=0)/n
        di2=(lambda_/(n*n))*(torch.sum(d, dim=0)**2)
        return (di-di2).sum()
    
    def get_point_loss(self,uv2surface_points, surface_points, object_mask):
        # pdb.set_trace()
        surface_points=surface_points.reshape(-1, 3)[object_mask]
        pt_loss = self.l2_loss(uv2surface_points, surface_points)
        return pt_loss
    
    def forward(self, model_outputs, model_inputs, ground_truth):
        # pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        rgbalb_gt = ground_truth['alb'].cuda()
        texture_gt = ground_truth['tex'].cuda()
        object_mask = model_outputs['object_mask']
        uv_outputs= model_outputs['uv_points']
        # imguv=model_inputs['uv'].cuda()/447.0
        uv2surf_points= model_outputs['uv2surface_points']
        uv_gt=ground_truth.get('uv_gt', None)
        uv_gt=uv_gt.cuda()
        # uv_gt=imguv.detach().clone()
        
        uvrgb_loss= self.get_uvrgb_loss(uv_outputs, rgbalb_gt, texture_gt, object_mask, mode=self.rgbloss_mode, lambda_=self.scinv_lambda)
        loss=self.fwd_rgb_weight *uvrgb_loss
            
        uv_loss=None
        if uv_gt is not None and self.uv_weight > 0.0:
            uv_loss= self.get_uv_loss(uv_outputs, uv_gt, object_mask)
            loss +=self.uv_weight * uv_loss
                
        pt_loss=None
        if uv2surf_points is not None:
            pt_loss= self.get_point_loss(uv2surf_points, model_inputs['wc'].detach(), object_mask)
            loss += self.pt_weight * pt_loss

        return {
            'loss': loss,
            'fwd_loss': uv_loss,
            'bwd_loss': pt_loss,
            'fwrgb_loss':uvrgb_loss
        }
        
        
class Doc3dUVFieldLoss(nn.Module):
    def __init__(self,fwd_weight, loss_mode, scinv_lambda):
        super().__init__()
        self.uv_weight =fwd_weight
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss()
        self.loss_mode=loss_mode
        self.scinv_lambda=scinv_lambda
    
    def get_uv_loss(self, uv_outputs, uv_gt, object_mask):
        # pdb.set_trace()
        uv_values_ = uv_outputs[object_mask]
        # uv_values_=torch.index_select(uv_values_, -1, torch.LongTensor([1,0]).to(uv_values_.device))
        uv_gt_= uv_gt[object_mask]
        # pdb.set_trace()
        if self.loss_mode=='l1':
            uvpt_loss= self.l1_loss(uv_values_, uv_gt_) / float(object_mask.shape[0]*object_mask.shape[1])
        elif self.loss_mode=='robust':
            uvpt_loss= self.robust_loss(uv_values_, uv_gt_, alpha=1.0, scale=0.1)
        uv_loss=uvpt_loss #+ torch.abs(uv_values_.mean()-0.5)+ torch.abs(uv_values_.var()-(1/12.0))
        return uv_loss
    
    def scaleinv_loss(self,pred, gt, lambda_=1.0):
        # pdb.set_trace()
        n=pred.shape[0]
        d=pred-gt
        di= torch.sum(d**2, dim=0)/n
        di2=(lambda_/(n*n))*(torch.sum(d, dim=0)**2)
        return (di-di2).sum()
    
    def robust_loss(self,pred, gt, alpha=2.0, scale=0.1):
        loss = torch.mean(robust_loss_pytorch.general.lossfun(
            pred - gt, alpha=torch.Tensor([alpha]).to(pred.device), scale=torch.Tensor([scale]).to(pred.device)))
        return loss
    
    def forward(self, model_outputs, model_inputs, ground_truth):
        # pdb.set_trace()
        object_mask = model_outputs['object_mask']
        uv_outputs= model_outputs['uv_points']
        uv_gt=ground_truth['uv_gt']
        uv_gt=uv_gt.cuda()

        uv_loss= self.get_uv_loss(uv_outputs, uv_gt, object_mask)
        loss =self.uv_weight * uv_loss

        return {
            'loss': loss,
            'fwd_loss': uv_loss
        }
        
class Doc3dUVSTNFieldLoss(nn.Module):
    def __init__(self,fwd_weight,pt_weight, loss_mode, scinv_lambda):
        super().__init__()
        self.uv_weight =fwd_weight
        self.pt_weight =pt_weight
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss()
        self.loss_mode=loss_mode
        self.scinv_lambda=scinv_lambda
    
    def get_uv_loss(self, uv_outputs, uv_gt, object_mask):
        # pdb.set_trace()
        uv_values_ = uv_outputs[object_mask]
        # uv_values_=torch.index_select(uv_values_, -1, torch.LongTensor([1,0]).to(uv_values_.device))
        uv_gt_= uv_gt[object_mask]
        # pdb.set_trace()
        if self.loss_mode=='l1':
            uvpt_loss= self.l1_loss(uv_values_, uv_gt_) / float(object_mask.shape[0]*object_mask.shape[1])
        elif self.loss_mode=='robust':
            uvpt_loss= self.robust_loss(uv_values_, uv_gt_, alpha=1.0, scale=0.1)
        uv_loss=uvpt_loss #+ torch.abs(uv_values_.mean()-0.5)+ torch.abs(uv_values_.var()-(1/12.0))
        return uv_loss
    
    # def get_rts_loss(self,pred_rotangle, pred_scale,gt_rotangle,gt_scale):
    #     rotloss=self.l2_loss(-pred_rotangle,gt_rotangle)
    #     scaleloss=self.l2_loss(1/pred_scale,gt_scale)
    #     rts_loss=rotloss+scaleloss
    #     return rts_loss
    
    def get_pt_loss(self, points_output, points_gt, object_mask):
        # pdb.set_trace()
        points_gt_ = points_gt[object_mask]
        # pdb.set_trace()
        # uv_loss=torch.abs(uv_values_.mean()-0.5)+ torch.abs(uv_values_.var()-(1/12.0))
        pts_loss,_=p3dloss.chamfer_distance(points_output, points_gt_.unsqueeze(0))
        # pdb.set_trace()
        return pts_loss
    
    def scaleinv_loss(self,pred, gt, lambda_=1.0):
        # pdb.set_trace()
        n=pred.shape[0]
        d=pred-gt
        di= torch.sum(d**2, dim=0)/n
        di2=(lambda_/(n*n))*(torch.sum(d, dim=0)**2)
        return (di-di2).sum()
    
    def robust_loss(self,pred, gt, alpha=2.0, scale=0.1):
        loss = torch.mean(robust_loss_pytorch.general.lossfun(
            pred - gt, alpha=torch.Tensor([alpha]).to(pred.device), scale=torch.Tensor([scale]).to(pred.device)))
        return loss
    
    def forward(self, model_outputs, model_inputs, ground_truth):
        # pdb.set_trace()
        object_mask = model_outputs['object_mask']
        uv_outputs= model_outputs['uv_points']
        pred_points= model_outputs['surf_points']
        points_gt=model_inputs['wc']
        uv_gt=ground_truth['uv_gt']
        uv_gt=uv_gt.cuda()
        # scale_gt=ground_truth['scale'].cuda()
        # angle_gt=ground_truth['rotangle'].cuda()
        # pdb.set_trace()
        pt_loss=self.get_pt_loss(pred_points, points_gt, object_mask)
        uv_loss= self.get_uv_loss(uv_outputs, uv_gt, object_mask)
        loss =(self.pt_weight*pt_loss) + (self.uv_weight * uv_loss)

        return {
            'loss': loss,
            'fwd_loss': uv_loss,
            'pt_loss':pt_loss
        }
        
        
        
class IDRUVColmapLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight, alpha, 
                lipschitz_const, lipschitz_weight, lipschitz_points, 
                lipschitz_mode, apply_normal_loss, normal_weight, radius, 
                bwd_weight, pnorm, delD, rgbcyc_weight, bwd_mode, fwd_weight,
                isom_weight,isom_mode):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.lipschitz_const=lipschitz_const
        self.lipschitz_weight=lipschitz_weight
        self.lipschitz_mode=lipschitz_mode
        self.lipschitz_points=lipschitz_points
        self.apply_normal_loss=apply_normal_loss
        self.normal_weight= normal_weight
        self.pt_weight =bwd_weight
        self.uv_weight =fwd_weight
        self.r=radius
        self.p=pnorm
        self.delD=delD
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss()
        self.l2_loss_per_elem = nn.MSELoss(reduction='none')
        self.l1_loss_per_elem = nn.L1Loss(reduction='none')
        self.rgb_cycle_weight= rgbcyc_weight
        self.bwd_mode=bwd_mode
        self.isom_weight=isom_weight
        self.isom_mode=isom_mode

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss
    
    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_lipschitz_loss(self, grad_theta, mode='mean'):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()
        # pdb.set_trace()
        if mode=='max':
            lip_loss=torch.max(torch.abs(grad_theta).max()-self.lipschitz_const, torch.Tensor([0.0]).to(grad_theta.device))
        else:
            lip_loss=torch.max(torch.abs(grad_theta).mean()-self.lipschitz_const, torch.Tensor([0.0]).to(grad_theta.device))
        return lip_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss
    
    def get_uv_reg_loss(self, uv_outputs, uv_pts, network_object_mask, object_mask):
        # pdb.set_trace()
        '''
        uv_pts: random points sampled from uv space
        '''
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        mask=network_object_mask & object_mask
        uv_values_ = uv_outputs[mask]
        # pdb.set_trace()
        # uv_loss=torch.abs(uv_values_.mean()-0.5)+ torch.abs(uv_values_.var()-(1/12.0))
        uv_loss,_=p3dloss.chamfer_distance(uv_values_.unsqueeze(0), uv_pts)
        return uv_loss
    
    def get_uvrgb_loss(self, uv_outputs, rgbalb_gt, texture_gt, network_object_mask, object_mask, mode='l2', lambda_=0.5):
        # pdb.set_trace()
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        mask=network_object_mask & object_mask
        uv_values_ = uv_outputs[mask]
        # uv_values_=torch.index_select(uv_values_, -1, torch.LongTensor([1,0]).to(uv_values_.device))
        rgbalb_gt = rgbalb_gt.reshape(-1, 3)[mask]
        # normalize uv_values in range [-1,1]
        uv_values=(2.0*uv_values_)-1
        # sample the gt texture using predicted uv values
        # texture_gt (B,3,h,w)
        # uv_values (B,1,n,2) n= number of surface points
        # rgbalb_gt (B,3,1,n)
        # rgbalb_pred (B,3,1,n)
        rgbalb_pred=F.grid_sample(texture_gt, uv_values.unsqueeze(0).unsqueeze(0), mode='bilinear')
        rgbalb_pred=rgbalb_pred.squeeze().transpose(1,0)
        if mode=='l2':
            uvrgb_loss = self.l2_loss(rgbalb_pred, rgbalb_gt) / float(object_mask.shape[0])
        else:
            uvrgb_loss = self.scaleinv_loss(rgbalb_pred, rgbalb_gt, object_mask, lambda_)
        return uvrgb_loss
    
    def scaleinv_loss(self,pred, gt, object_mask, lambda_=1.0):
        # pdb.set_trace()
        n=object_mask.shape[0]
        d=pred-gt
        di= torch.sum(d**2, dim=0)/n
        di2=(lambda_/(n*n))*(torch.sum(d, dim=0)**2)
        return (di-di2).sum()
    
    def get_point_loss(self,uv2surface_points, surface_points):
        pt_loss = self.l2_loss(uv2surface_points, surface_points)
        return pt_loss
    
    def get_weighted_point_loss(self,uv2surface_points, surface_points, weights):
        pt_loss = self.l2_loss_per_elem(uv2surface_points, surface_points)
        pt_loss=weights*pt_loss.sum(dim=-1)
        # pdb.set_trace()
        return pt_loss.mean()
    
    def get_isom_loss(self, diff_props, mode='scisom'):
        self._zero = torch.tensor(0.).to(diff_props['fff'].device)
        self._one = torch.tensor(1.).to(diff_props['fff'].device)
        self._eps = torch.tensor(1e-6)
        E, F, G = diff_props['fff'].permute(2, 0, 1) 
        
        # pdb.set_trace()
        # Get per point local squared area.
        # A2 = torch.max(E * G - F.pow(2), self._zero)  # (B, P)
        # A2 = torch.sum((E * G - F.pow(2)+self._eps).sqrt())  # (B, P)
        # pdb.set_trace()
        # A = A2.sqrt()  # (B, P, spp)
        
        if mode=='scisom':
            muE = E.mean()
            muG = G.mean()
        else:
            muE =self._one
            muG =self._one
            
        L_E = ((E - muE).pow(2)).mean() #/ (A2 + self._eps)).mean() 
        L_G = ((G - muG).pow(2)).mean() #/ (A2 + self._eps)).mean()
        L_F = (F.pow(2)).mean() #/ (A2 + self._eps)).mean()
        '''
        L_E = ((E - muE).pow(2)/ (A2 + self._eps)).mean() 
        L_G = ((G - muG).pow(2)/ (A2 + self._eps)).mean()
        L_F = ((F.pow(2)).mean()/ (A2 + self._eps)).mean()
        '''
        return L_E, L_G, L_F
    
    def get_normal_loss(self, surf_points, normals, decay, p, delD):
        if len(surf_points.shape)==2:
            surf_points=surf_points.unsqueeze(0)
            normals=normals.unsqueeze(0)
        # calc distance
        dist=self.distance_matrix(surf_points, surf_points)
        # pdb.set_trace()
        # calculate normal distance as p norm
        norm_pdist=self.p_distance_matrix(normals, normals, p=p)
        if delD is not None:
            # use delD to only take points within delD radius
            # normalize the weights as well# 
            # pdb.set_trace()
            dist_w = torch.where(dist >= delD, torch.zeros_like(dist), torch.exp(-decay*dist))
            norm_pdist_w=norm_pdist*dist_w
        else:
            # weight the norm distances
            norm_pdist_w=norm_pdist*torch.exp(-decay*dist)
        
        return norm_pdist_w.sum(dim=-1).mean()
    
    def distance_matrix(self, pc_N, pc_M):
        """ Computes a distance matrix between two point sets.
        Args:
            pc_N (torch.Tensor): shape (B, N, 3)
            pc_M (torch.Tensor): shape (B, M, 3)
        Returns:
            Distance matrix, shape (B, M, N).
        """
        eps=1e-6
        # Get per-point distance matrix.
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape

        assert (B == B2)
        assert (D == D2)
        assert (D == 3)

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))

        # return (x - y).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N) # leads to nan gradient for (0).sqrt()
        return ((x - y)+eps).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N) # leads to nan gradient for (0).sqrt()
    
    def p_distance_matrix(self, pc_N, pc_M, p=2):
        """ Computes a p norm distance matrix between two point sets.
        Args:
            pc_N (torch.Tensor): shape (B, N, 3)
            pc_M (torch.Tensor): shape (B, M, 3)
        Returns:
            Distance matrix, shape (B, M, N).
        """
        eps=1e-6
        # Get per-point distance matrix.
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape

        assert (B == B2)
        assert (D == D2)
        assert (D == 3)

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))

        return (torch.abs(x - y)+eps).pow(p).sum(dim=3).pow(1/p)  # (B, M, N, 3) -> (B, M, N)

    def forward(self, model_outputs, ground_truth):
        # pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        rgbalb_gt = ground_truth['rgb'].cuda()
        texture_gt = ground_truth['tex'].cuda()
        uv_pts= ground_truth['uv_pts'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']
        uv_outputs= model_outputs['uv_points']
        uv2surf_points= model_outputs['uv2surface_points']
        diff_props=model_outputs['diff_props']
        imp_map = ground_truth['imp_map'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        # if self.lipschitz_points=='random':
        #     lipsch_loss = self.get_lipschitz_loss(model_outputs['grad_eikpoints'], self.lipschitz_mode)
        # else:
        #     lipsch_loss = self.get_lipschitz_loss(model_outputs['grad_points'], self.lipschitz_mode)

        loss = rgb_loss + \
                self.eikonal_weight * eikonal_loss + \
                self.mask_weight * mask_loss
                # self.lipschitz_weight * lipsch_loss
        
        # get uv rgb loss, to understand the uv prediction quality, do not backprop
        # uvrgb_loss=self.get_uvrgb_loss(uv_outputs.detach(), rgbalb_gt.detach(), texture_gt.detach(), network_object_mask.detach(), object_mask.detach())
        
        uv_loss=None
        if self.uv_weight > 0.0:
            uvr_loss= self.get_uv_reg_loss(uv_outputs, uv_pts, network_object_mask, object_mask)
            loss +=self.uv_weight * uvr_loss
            
        normal_loss=None
        if self.apply_normal_loss:
            normal_loss =self.get_normal_loss(model_outputs['diff_surface_points'],model_outputs['diff_normals'], self.r, self.p, self.delD)
            loss = loss +  (self.normal_weight * normal_loss)
        
        pt_loss=None
        if uv2surf_points is not None:
            if self.bwd_mode=='wl2':
                mask=network_object_mask & object_mask
                ptloss_weight=imp_map.reshape(-1)[mask]
                # ptloss_weight=ptloss_weight/ptloss_weight.sum()
                # pdb.set_trace()
                pt_loss= self.get_weighted_point_loss(uv2surf_points, model_outputs['diff_surface_points'].clone().detach(), weights=ptloss_weight)
            elif self.bwd_mode=='trl2':
                pt_loss= self.get_point_loss(uv2surf_points, model_outputs['tr_diff_surface_points'].clone().detach())
            else:
                pt_loss= self.get_point_loss(uv2surf_points, model_outputs['diff_surface_points'].clone().detach())
            
            loss = loss +  (self.pt_weight * pt_loss)  

        rgb_cycle_loss=None
        if self.rgb_cycle_weight > 0.0:
            rgb_cycle_loss = self.get_rgb_loss(model_outputs['bwdrgb_values'], rgb_gt , network_object_mask, object_mask)
            loss +=  self.rgb_cycle_weight * rgb_cycle_loss

        LE=None
        LF=None
        LG=None
        if len(self.isom_weight)>0:
            LE, LG, LF=self.get_isom_loss(diff_props, mode=self.isom_mode) # scisom : scaled isometry (conformal)
            isom_loss=(self.isom_weight[0]*LE)+(self.isom_weight[1]*LF)+(self.isom_weight[2]*LG)
            # pdb.set_trace()
            loss+=isom_loss
        
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            # 'lip_loss': lipsch_loss,
            'mask_loss': mask_loss,
            'normal_loss': normal_loss,
            'fwd_loss': uv_loss,
            'bwd_loss': pt_loss,
            # 'fwrgb_loss':uvrgb_loss,
            'rgbcyc_loss':rgb_cycle_loss,
            'L_E':LE,
            'L_F':LF,
            'L_G':LG,
        }