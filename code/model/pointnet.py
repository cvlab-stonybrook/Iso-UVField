# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import pdb


#UTILITIES
class STN3d(nn.Module):
    def __init__(self, rot=True, trans=True, scale=True):
        super().__init__()
        # self.num_points = num_points
        out_dim=0
        # 1 parameter for rotation (rotation wrt z), 2 translation, 2 scale
        self.rot=rot
        self.trans=trans
        self.scale=scale
        
        '''
        if self.rot:
            out_dim+=1
        if self.trans:
            out_dim+=2
        if self.scale:
            out_dim+=2
        '''
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # self.fc3 = nn.Linear(256, out_dim)
        if self.rot:
            self.fcrot = nn.Linear(256, 1)
            self.fcrot.weight.data.zero_()
            self.fcrot.bias.data.zero_()
        if self.trans:
            self.fctrans = nn.Linear(256, 2)
            # self.fctrans.weight.data.zero_()
            # self.fctrans.bias.data.zero_()
        if self.scale:
            self.fcscale = nn.Linear(256, 1)
            self.fcscale.weight.data.zero_()
            self.fcscale.bias.data.copy_(torch.Tensor([1.5]))
        self.relu = nn.ReLU()

        # self._iden = torch.from_numpy(np.eye(3, dtype=np.float32)).\
        #     reshape((1, 9))
        ''' 
        self.fc3.weight.data.zero_()
        bias=torch.zeros(out_dim)
        if scale:
            bias[-2:]=torch.Tensor([1,1])
        self.fc3.bias.data.copy_(bias)
        '''
        

    def forward(self, x):
        # pdb.set_trace()
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # pdb.set_trace() 
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        # iden = self._iden.repeat(batchsize, 1).to(x.device)
        # x[0]=angle, x[1]= tx, x[2]= ty, x[3]= sx, x[4]= sy 
        # pdb.set_trace()
        xrot=None
        if self.rot:
            xrot=F.relu(self.fcrot(x))
            tmat=self.get_rotation_matrix(xrot)
        
        xscale=None
        if self.scale:
            xscale=self.fcscale(x)
            tmat[:,0,0]*=(1/xscale[:,0])
            tmat[:,0,1]*=(1/xscale[:,0])
            tmat[:,1,0]*=(1/xscale[:,0])
            tmat[:,1,1]*=(1/xscale[:,0])
        xtrans=None
        if self.trans:
            xtrans=self.fctrans(x)
            tmat[:,0,2]=xtrans[:,0]
            tmat[:,1,2]=xtrans[:,0]
            
        # x= torch.cat([x, torch.zeros(batchsize,3).to(x.device)], dim=-1)
        # x = x + iden
        # x = x.view(-1, 3, 3)
        return tmat, xrot, xscale, xtrans
    
    def get_rotation_matrix(self, angle):
        batchsize=angle.shape[0]
        rotmat = torch.zeros(batchsize,3,3).to(angle.device)
        cosval = torch.cos(torch.Tensor([angle]))
        sinval = torch.sin(torch.Tensor([angle]))
        rotmat[:,0,0]=cosval
        rotmat[:,0,1]=sinval
        rotmat[:,1,0]=-sinval
        rotmat[:,1,1]=cosval
        rotmat[:,2,2]=1.0
        # rotation_matrixz = torch.Tensor([[cosval, -sinval, 0],[sinval, cosval, 0],[0, 0, 1]])
        return rotmat
        


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, trans=False,
                gpu=True):

        super().__init__()
        self.stn = STN3d(gpu=gpu)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans

        # self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input pcloud, shape (B, N, 3).
        Returns:
        """

        # Adapt input shape (B, N, 3) to (B, 3, N) for nn.Conv1D to work.
        x = x.transpose(2, 1)
        # pdb.set_trace()

        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x))
        # pdb.set_trace()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                raise NotImplementedError
        else:
            return x


class ANEncoderPN(nn.Module):
    """ PointNet-based encoder used by AtlasNet.
    Args:
        N (int):
        code (int): Size of the CW.
        normalize_cw (bool): Whether to normalize the CW.
    """
    def __init__(self, code=1048, normalize_cw=False):
        super().__init__()
        self._normalize_cw = normalize_cw
        self.pnet=PointNetfeat(global_feat=True, trans=False)
        self.layers = nn.Sequential(
            nn.Linear(1024, code),
            # nn.BatchNorm1d(code),
            nn.Tanh(),
            nn.Linear(code, 64))

    def forward(self, pcloud):
        penc=self.pnet(pcloud)
        cws = self.layers(penc)

        if self._normalize_cw:
            cws = F.normalize(cws, dim=1)

        return cws