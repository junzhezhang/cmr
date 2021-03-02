"""
leveraged from texture fields

mesh2tex/texnet/models/decoder.py

DecoderEachLayerCLarger
DecoderEachLayerC

for each decoder, input is: uv coordinates, image decoded embedding, w/o geom_embedding
uv coordinates optionally have positional encoding.

### for tex_flow_decoder
DecoderEachLayerC --> DecoderEachLayerCFlow

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# # from mesh2tex import common
# from mesh2tex.layers import (
#     ResnetBlockPointwise,
#     EqualizedLR
# )

class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Resnet Blocks
class ResnetBlockConv1D(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockPointwise(nn.Module):
    def __init__(self, f_in, f_out=None, f_hidden=None,
                 is_bias=True, actvn=F.relu, factor=1., eq_lr=False):
        super().__init__()
        # Filter dimensions
        if f_out is None:
            f_out = f_in

        if f_hidden is None:
            f_hidden = min(f_in, f_out)

        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out

        self.factor = factor
        self.eq_lr = eq_lr

        # Activation function
        self.actvn = actvn

        # Submodules
        self.conv_0 = nn.Conv1d(f_in, f_hidden, 1)
        self.conv_1 = nn.Conv1d(f_hidden, f_out, 1, bias=is_bias)

        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)

        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv1d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)

        # Initialization
        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        net = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(net))
        x_s = self.shortcut(x)
        return x_s + self.factor * dx


class ResnetBlockConv2d(nn.Module):
    def __init__(self, f_in, f_out=None, f_hidden=None,
                 is_bias=True, actvn=F.relu, factor=1.,
                 eq_lr=False, pixel_norm=False):
        super().__init__()
        # Filter dimensions
        if f_out is None:
            f_out = f_in

        if f_hidden is None:
            f_hidden = min(f_in, f_out)

        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out
        self.factor = factor
        self.eq_lr = eq_lr
        self.use_pixel_norm = pixel_norm

        # Activation
        self.actvn = actvn

        # Submodules
        self.conv_0 = nn.Conv2d(self.f_in, self.f_hidden, 3,
                                stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.f_hidden, self.f_out, 3,
                                stride=1, padding=1, bias=is_bias)

        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)

        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv2d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)

        # Initialization
        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        x_s = self.shortcut(x)

        if self.use_pixel_norm:
            x = pixel_norm(x)
        dx = self.conv_0(self.actvn(x))

        if self.use_pixel_norm:
            dx = pixel_norm(dx)
        dx = self.conv_1(self.actvn(dx))

        out = x_s + self.factor * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class EqualizedLR(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._make_params()

    def _make_params(self):
        weight = self.module.weight

        height = weight.data.shape[0]
        width = weight.view(height, -1).data.shape[1]

        # Delete parameters in child
        del self.module._parameters['weight']
        self.module.weight = None

        # Add parameters to myself
        self.weight = nn.Parameter(weight.data)

        # Inherit parameters
        self.factor = np.sqrt(2 / width)

        # Initialize
        nn.init.normal_(self.weight)

        # Inherit bias if available
        self.bias = self.module.bias
        self.module.bias = None

        if self.bias is not None:
            del self.module._parameters['bias']
            nn.init.zeros_(self.bias)

    def forward(self, *args, **kwargs):
        self.module.weight = self.factor * self.weight
        if self.bias is not None:
            self.module.bias = 1. * self.bias
        out = self.module.forward(*args, **kwargs)
        self.module.weight = None
        self.module.bias = None
        return out


def pixel_norm(x):
    sigma = x.norm(dim=1, keepdim=True)
    out = x / (sigma + 1e-5)
    return out

class DecoderEachLayerC(nn.Module):
    def __init__(self, z_dim=128, dim=2,
                 hidden_size=128, leaky=True, 
                 resnet_leaky=True, eq_lr=False):
        super().__init__()
        # self.c_dim = c_dim
        self.eq_lr = eq_lr

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if not resnet_leaky:
            self.resnet_actvn = F.relu
        else:
            self.resnet_actvn = lambda x: F.leaky_relu(x, 0.2)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.block0 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block1 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block2 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block3 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block4 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

        self.fc_cz_0 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_1 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_2 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_3 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_4 = nn.Linear(z_dim, hidden_size)

        self.conv_out = nn.Conv1d(hidden_size, 3, 1)

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_out = EqualizedLR(self.conv_out)
            self.fc_cz_0 = EqualizedLR(self.fc_cz_0)
            self.fc_cz_1 = EqualizedLR(self.fc_cz_1)
            self.fc_cz_2 = EqualizedLR(self.fc_cz_2)
            self.fc_cz_3 = EqualizedLR(self.fc_cz_3)
            self.fc_cz_4 = EqualizedLR(self.fc_cz_4)

        # Initialization
        nn.init.zeros_(self.conv_out.weight)
        
        self.umin = 100
        self.umax = -100
        self.vmin = 100
        self.vmin = -100
    # def forward(self, p, z, **kwargs):
    def forward(self, **kwargs):
        
        p = kwargs['p']
        z_origin = kwargs['z']
        B = z_origin.shape[0]
        
        z = F.adaptive_avg_pool2d(z_origin, output_size=1).view(B, -1)
        
        # z = z_origin.view(B,-1)
        # c = geom_descr['global']
        # import pdb; pdb.set_trace()
        batch_size, D, T = p.size()

        # self.umin = min(torch.min(p).item(),self.umin)
        # self.umax = max(torch.max(p).item(),self.umax)
        # self.vmin = min(torch.min(p).item(),self.umin)
        # self.vmin = max(torch.max(p).item(),self.umax)
        # cz = torch.cat([c, z], dim=1)
        # print('max:',self.umax, slef.vmax)
        # print(torch.max(torch.max(p,dim=2)[0], dim=0)[0],
# torch.min(torch.min(p,dim=2)[0], dim=0)[0])

        net = self.conv_p(p)
        net = net + self.fc_cz_0(z).unsqueeze(2)
        net = self.block0(net)
        net = net + self.fc_cz_1(z).unsqueeze(2)
        net = self.block1(net)
        net = net + self.fc_cz_2(z).unsqueeze(2)
        net = self.block2(net)
        net = net + self.fc_cz_3(z).unsqueeze(2)
        net = self.block3(net)
        net = net + self.fc_cz_4(z).unsqueeze(2)
        net = self.block4(net)

        out = self.conv_out(self.actvn(net))
        out = torch.sigmoid(out)

        return out


# class DecoderEachLayerCLarger(nn.Module):
#     def __init__(self, c_dim=128, z_dim=128, dim=2,
#                  hidden_size=128, leaky=True, 
#                  resnet_leaky=True, eq_lr=False):
#         super().__init__()
#         self.c_dim = c_dim
#         self.eq_lr = eq_lr
#         if not leaky:
#             self.actvn = F.relu
#         else:
#             self.actvn = lambda x: F.leaky_relu(x, 0.2)
        
#         if not resnet_leaky:
#             self.resnet_actvn = F.relu
#         else:
#             self.resnet_actvn = lambda x: F.leaky_relu(x, 0.2)

#         # Submodules
#         self.conv_p = nn.Conv1d(dim, hidden_size, 1)

#         self.block0 = ResnetBlockPointwise(
#             hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
#         self.block1 = ResnetBlockPointwise(
#             hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
#         self.block2 = ResnetBlockPointwise(
#             hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
#         self.block3 = ResnetBlockPointwise(
#             hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
#         self.block4 = ResnetBlockPointwise(
#             hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
#         self.block5 = ResnetBlockPointwise(
#             hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
#         self.block6 = ResnetBlockPointwise(
#             hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

#         self.fc_cz_0 = nn.Linear(c_dim + z_dim, hidden_size)
#         self.fc_cz_1 = nn.Linear(c_dim + z_dim, hidden_size)
#         self.fc_cz_2 = nn.Linear(c_dim + z_dim, hidden_size)
#         self.fc_cz_3 = nn.Linear(c_dim + z_dim, hidden_size)
#         self.fc_cz_4 = nn.Linear(c_dim + z_dim, hidden_size)
#         self.fc_cz_5 = nn.Linear(c_dim + z_dim, hidden_size)
#         self.fc_cz_6 = nn.Linear(c_dim + z_dim, hidden_size)

#         self.conv_out = nn.Conv1d(hidden_size, 3, 1)

#         if self.eq_lr:
#             self.conv_p = EqualizedLR(self.conv_p)
#             self.conv_out = EqualizedLR(self.conv_out)
#             self.fc_cz_0 = EqualizedLR(self.fc_cz_0)
#             self.fc_cz_1 = EqualizedLR(self.fc_cz_1)
#             self.fc_cz_2 = EqualizedLR(self.fc_cz_2)
#             self.fc_cz_3 = EqualizedLR(self.fc_cz_3)
#             self.fc_cz_4 = EqualizedLR(self.fc_cz_4)
#             self.fc_cz_5 = EqualizedLR(self.fc_cz_5)
#             self.fc_cz_6 = EqualizedLR(self.fc_cz_6)

#         # Initialization
#         nn.init.zeros_(self.conv_out.weight)

#     def forward(self, p, geom_descr, z, **kwargs):
#         c = geom_descr['global']
#         batch_size, D, T = p.size()

#         cz = torch.cat([c, z], dim=1)

#         net = self.conv_p(p)
#         net = net + self.fc_cz_0(cz).unsqueeze(2)
#         net = self.block0(net)
#         net = net + self.fc_cz_1(cz).unsqueeze(2)
#         net = self.block1(net)
#         net = net + self.fc_cz_2(cz).unsqueeze(2)
#         net = self.block2(net)
#         net = net + self.fc_cz_3(cz).unsqueeze(2)
#         net = self.block3(net)
#         net = net + self.fc_cz_4(cz).unsqueeze(2)
#         net = self.block4(net)
#         net = net + self.fc_cz_5(cz).unsqueeze(2)
#         net = self.block5(net)
#         net = net + self.fc_cz_6(cz).unsqueeze(2)
#         net = self.block6(net)

#         out = self.conv_out(self.actvn(net))
#         out = torch.sigmoid(out)

#         return out


class DecoderEachLayerCFlow(nn.Module):
    def __init__(self, z_dim=128, dim=2,
                 hidden_size=128, leaky=True, 
                 resnet_leaky=True, eq_lr=False):
        super().__init__()
        # self.c_dim = c_dim
        self.eq_lr = eq_lr

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if not resnet_leaky:
            self.resnet_actvn = F.relu
        else:
            self.resnet_actvn = lambda x: F.leaky_relu(x, 0.2)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.block0 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block1 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block2 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block3 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block4 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

        self.fc_cz_0 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_1 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_2 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_3 = nn.Linear(z_dim, hidden_size)
        self.fc_cz_4 = nn.Linear(z_dim, hidden_size)

        self.conv_out = nn.Conv1d(hidden_size, 2, 1) # output xy coordinates

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_out = EqualizedLR(self.conv_out)
            self.fc_cz_0 = EqualizedLR(self.fc_cz_0)
            self.fc_cz_1 = EqualizedLR(self.fc_cz_1)
            self.fc_cz_2 = EqualizedLR(self.fc_cz_2)
            self.fc_cz_3 = EqualizedLR(self.fc_cz_3)
            self.fc_cz_4 = EqualizedLR(self.fc_cz_4)

        # Initialization
        nn.init.zeros_(self.conv_out.weight)
        
        self.umin = 100
        self.umax = -100
        self.vmin = 100
        self.vmin = -100
    # def forward(self, p, z, **kwargs):
    def forward(self, **kwargs):
        
        p = kwargs['p']
        z = kwargs['z']
        B = z.shape[0]
        
        # z = F.adaptive_avg_pool2d(z_origin, output_size=1).view(B, -1)
        
        # z = z_origin.view(B,-1)
        # c = geom_descr['global']
        # import pdb; pdb.set_trace()
        batch_size, D, T = p.size()

        # self.umin = min(torch.min(p).item(),self.umin)
        # self.umax = max(torch.max(p).item(),self.umax)
        # self.vmin = min(torch.min(p).item(),self.umin)
        # self.vmin = max(torch.max(p).item(),self.umax)
        # cz = torch.cat([c, z], dim=1)
        # print('max:',self.umax, slef.vmax)
        # print(torch.max(torch.max(p,dim=2)[0], dim=0)[0],
# torch.min(torch.min(p,dim=2)[0], dim=0)[0])

        net = self.conv_p(p)
        net = net + self.fc_cz_0(z).unsqueeze(2)
        net = self.block0(net)
        net = net + self.fc_cz_1(z).unsqueeze(2)
        net = self.block1(net)
        net = net + self.fc_cz_2(z).unsqueeze(2)
        net = self.block2(net)
        net = net + self.fc_cz_3(z).unsqueeze(2)
        net = self.block3(net)
        net = net + self.fc_cz_4(z).unsqueeze(2)
        net = self.block4(net)

        out = self.conv_out(self.actvn(net))
        # out = torch.sigmoid(out)
        # change to tanh
        out = torch.tanh(out)

        return out