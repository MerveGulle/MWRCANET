import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import SupportingFunctions as sf

# x0  : initial solution
# zn  : Output of nth denoiser block
# L   : regularization coefficient
# tol : tolerance for breaking the CG iteration
def DC_layer(x0,zn,L,S,mask,tol=0,cg_iter=10):
    _,Nx,Ny = x0.shape
    # xn = torch.zeros((Nx, Ny), dtype=torch.cfloat)
    xn = x0[0,:,:]*0
    a  = torch.squeeze(x0 + L*zn)
    p  = a
    r  = a
    for i in np.arange(cg_iter):
        delta = torch.sum(r*torch.conj(r)).real/torch.sum(a*torch.conj(a)).real
        if(delta<tol):
            break
        else:
            p1 = p[None,:,:]
            q  = torch.squeeze(sf.decode(sf.encode(p1,S,mask),S)) + L* p
            t  = (torch.sum(r*torch.conj(r))/torch.sum(q*torch.conj(p)))
            xn = xn + t*p 
            rn = r  - t*q 
            p  = rn + (torch.sum(rn*torch.conj(rn))/torch.sum(r*torch.conj(r)))*p
            r  = rn
            
    return xn[None,:,:]

class HITVPCTeam:
    r"""
        DWT and IDWT block written by: Yue Cao
        """
    class CALayer(nn.Module):
        def __init__(self, channel=64, reduction=16):
            super(HITVPCTeam.CALayer, self).__init__()

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.conv_du(y)
            return x * y

    # conv - prelu - conv - sum
    class RB(nn.Module):
        def __init__(self, filters):
            super(HITVPCTeam.RB, self).__init__()
            self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
            self.act = nn.PReLU()
            self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
            self.cuca = HITVPCTeam.CALayer(channel=filters)

        def forward(self, x):
            c0 = x
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            out = self.cuca(x)
            return out + c0

    class NRB(nn.Module):
        def __init__(self, n, f):
            super(HITVPCTeam.NRB, self).__init__()
            nets = []
            for i in range(n):
                nets.append(HITVPCTeam.RB(f))
            self.body = nn.Sequential(*nets)
            self.tail = nn.Conv2d(f, f, 3, 1, 1)

        def forward(self, x):
            return x + self.tail(self.body(x))

    class DWTForward(nn.Module):
        def __init__(self):
            super(HITVPCTeam.DWTForward, self).__init__()
            ll = np.array([[0.5, 0.5], [0.5, 0.5]])
            lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
            hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
            hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
            filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                              hl[None,::-1,::-1], hh[None,::-1,::-1]],
                             axis=0)
            self.weight = nn.Parameter(
                torch.tensor(filts).to(torch.get_default_dtype()),
                requires_grad=False)
        def forward(self, x):
            C = x.shape[1]
            filters = torch.cat([self.weight,] * C, dim=0)
            y = F.conv2d(x, filters, groups=C, stride=2)
            return y

    class DWTInverse(nn.Module):
        def __init__(self):
            super(HITVPCTeam.DWTInverse, self).__init__()
            ll = np.array([[0.5, 0.5], [0.5, 0.5]])
            lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
            hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
            hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
            filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                              hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                             axis=0)
            self.weight = nn.Parameter(
                torch.tensor(filts).to(torch.get_default_dtype()),
                requires_grad=False)

        def forward(self, x):
            C = int(x.shape[1] / 4)
            filters = torch.cat([self.weight, ] * C, dim=0)
            y = F.conv_transpose2d(x, filters, groups=C, stride=2)
            return y


class Net(nn.Module):
    def __init__(self, channels=1, filters_level1=96, filters_level2=256//2, filters_level3=256//2, n_rb=4*5):
    #def __init__(self, channels=2, filters_level1=16, filters_level2=16, filters_level3=16, n_rb=4*5):
        super(Net, self).__init__()

        self.head = HITVPCTeam.DWTForward()

        self.down1 = nn.Sequential(
            nn.Conv2d(channels * 4, filters_level1, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.NRB(n_rb, filters_level1))

        # sum 1
        # self.down1 = HITVPCTeam.NRB(n_rb, filters_level1),

        # sum 2
        self.down2 = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level1 * 4, filters_level2, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.NRB(n_rb, filters_level2))

        self.down3 = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level2 * 4, filters_level3, 3, 1, 1),
            nn.PReLU())

        self.middle = HITVPCTeam.NRB(n_rb, filters_level3)

        self.up1 = nn.Sequential(
            nn.Conv2d(filters_level3, filters_level2 * 4, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.DWTInverse())

        self.up2 = nn.Sequential(
            HITVPCTeam.NRB(n_rb, filters_level2),
            nn.Conv2d(filters_level2, filters_level1 * 4, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.DWTInverse())

        self.up3 = nn.Sequential(
            HITVPCTeam.NRB(n_rb, filters_level1),
            nn.Conv2d(filters_level1, channels * 4, 3, 1, 1))

        self.tail = HITVPCTeam.DWTInverse()
        
        self.L = nn.Parameter(torch.tensor(0.05, requires_grad=True))
    

    def forward(self, inputs):
        inputs = sf.ch1to2(inputs[None,:,:,:]).float()
        c1 = self.head(inputs)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        outputs = self.down3(c3)
        outputs = self.middle(outputs)
        outputs = self.up1(outputs) + c3
        outputs = self.up2(outputs) + c2
        outputs = self.up3(outputs) + c1
        outputs = self.tail(outputs)
        return self.L, sf.ch2to1(outputs)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)