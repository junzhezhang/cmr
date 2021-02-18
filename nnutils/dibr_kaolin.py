from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import scipy.misc
import tqdm
import cv2

import torch

from nnutils import geom_utils

# from kaolin.graphics.dib_renderer.rasterizer import linear_rasterizer
# from kaolin.graphics.dib_renderer.utils import datanormalize
# from kaolin.graphics.dib_renderer.renderer.phongrender import PhongRender
from kaolin.graphics.dib_renderer.renderer.texrender import TexRender
from kaolin.graphics.dib_renderer.utils.perspective import lookatnp, perspectiveprojectionnp

from kaolin.graphics.dib_renderer.utils.mesh import loadobj, face2pfmtx, loadobjtex, savemesh


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

class NeuralRenderer(torch.nn.Module):
    """
    replace NeuralRenderer from nmr.py with the kaolin's
    """
    # 512 --> 256 TODO
    def __init__(self, img_size=256,uv_sampler=None):
        self.img_size = img_size
        super(NeuralRenderer, self).__init__()
        self.renderer = TexRender(height=img_size,width=img_size)
        # self.renderer = NeuralMeshRenderer(image_size=img_size, camera_mode='look_at',perspective=False,viewing_angle=30,light_intensity_ambient=0.8)
        self.offset_z = 5.
        self.proj_fn = geom_utils.orthographic_proj_withz
        if uv_sampler is not None:
            self.uv_sampler = uv_sampler.clone()
        else:
            print('no uv sampler')
        print('DIB-R...')
    
    def ambient_light_only(self):
        # Make light only ambient.
        # self.renderer.light_intensity_ambient = 1
        # self.renderer.light_intensity_directional = 0
        print("TODO: ambient_light_only")
        pass
    
    def set_bgcolor(self, color):
        # self.renderer.background_color = color
        print("TODO: set_bgcolor")
        pass

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]
    
    def forward(self, vertices, faces, cams, textures=None):
        ### TODO save mesh
        if textures is not None:
            v_np = vertices[0].detach().cpu().numpy()
            f_np = faces[0].detach().cpu().numpy()
            file_name = 'vis/bird.obj'
            try:
                savemesh(v_np, f_np, file_name)
            except:
                import pdb; pdb.set_trace()
        # ours = False
        ours = True
        if ours:
            translation = cams[:,:3]
            quant = cams[:,-4:]
            tfcamviewmtx_bx3x3 =  quaternion_to_matrix(quant)
            tfcamshift_bx3 = - translation

            # camfovy = 45 / 180.0 * np.pi
            camfovy = 90 / 180.0 * np.pi
            camprojmtx = perspectiveprojectionnp(camfovy, 1.0 * 1.0 / 1.0)
            tfcamproj_3x1 = torch.from_numpy(camprojmtx).cuda()

            tfcameras = [tfcamviewmtx_bx3x3,
                 tfcamshift_bx3,
                 tfcamproj_3x1]
        else:
            tfcameras = self.get_sample_cams(bs=vertices.shape[0])
        # import pdb; pdb.set_trace()
        print('1:',tfcameras[0].shape)
        print('2:',tfcameras[1].shape)
        print('3:',tfcameras[2].shape)
        
        
        if textures is None:
            tex_flag = False
            # shape = [vertices.shape[0], 1280, 6,6,6,3]
            # textures = torch.ones(vertices.shape[0], 1280, 6,6,6,3).cuda()*256
            textures = torch.ones(vertices.shape[0],3,self.img_size,self.img_size).cuda()
        else:
            tex_flag = True
            
            # # TODO try with convmesh output
            imfile = '/mnt/lustre/zhangjunzhe/tm/convmesh/output/pretrained_cub_512x512_class/mesh_0.png'
            # textures_np = cv2.imread(imfile)[:, :, ::-1].astype(np.float32) / 255.0
            textures_np = cv2.imread(imfile)[:, :, ::-1].astype(np.float32) 
            dim = (self.img_size, self.img_size)
            resized = cv2.resize(textures_np, dim, interpolation = cv2.INTER_AREA)
            textures = torch.from_numpy(resized).cuda().unsqueeze(0)
            textures = textures.permute([0, 3, 1, 2])
            # print('tex shape:', textures.shape)
            # # import pdb; pdb.set_trace()
            # textures = torch.ones(vertices.shape[0],3,self.img_size,self.img_size).cuda()

        # print(texture)
            # renderer.set_smooth(pfmtx) # TODO for phong renderer
        tfp_bxpx3 = vertices
        tff_fx3 = faces[0] # TODO to verify if fixed topology within a batch
        # tff_fx3 = tff_fx3.type(int64)
        tff_fx3 = tff_fx3.type(torch.long)
        points = [tfp_bxpx3, tff_fx3]
        uvs = self.uv_sampler
        # TODO texture to clone?
        # TODOL ft_fx3
        # ft_fx3??? TODO
        #only keep rgb, no alpha and depth
        print('uv shape:',uvs.shape)
        imgs = self.renderer(points=points,
                                cameras=tfcameras,
                                uv_bxpx2 = uvs,
                                texture_bx3xthxtw=textures,
                                ft_fx3=None)[0]
        if tex_flag:
            for i, img in enumerate(imgs):
                img = img.detach().cpu().numpy()

                cv2.imwrite('./vis/lam'+str(i)+'.jpg',img*255)
                print('saved img')
            print('!!!imgs:',imgs.shape)
        
        imgs = imgs.permute([0,3,1,2])
        print('new shape:',imgs.shape)
        # print('  cam:',cams)          
        return imgs

    def get_sample_cams(self,bs):
        ##########################################################
        # campos = np.array([0, 0, 1.5], dtype=np.float32)  # where camera it is
        # campos = np.array([0, 0, 4], dtype=np.float32)
        # campos = np.array([0, 4, 0], dtype=np.float32)
        campos = np.array([4, 0, 0], dtype=np.float32)
        
        camcenter = np.array([0, 0, 0], dtype=np.float32)  # where camra is looking at
        
        # camup = np.array([-1, 1, 0], dtype=np.float32)  # y axis of camera view
        # camup = np.array([-1, 0, 1], dtype=np.float32)
        # camup = np.array([0, -1, 1], dtype=np.float32)
        # camup = np.array([0, 1, -1], dtype=np.float32)
        # camup = np.array([1, -1, 0], dtype=np.float32)
        # camup = np.array([1, 0, -1], dtype=np.float32)
        # camup = np.array([1, 1, 0], dtype=np.float32)
        # camup = np.array([-1, 0, -1], dtype=np.float32)
        camup = np.array([1, 0, 1], dtype=np.float32)
        
        camviewmtx, camviewshift = lookatnp(campos.reshape(3, 1), camcenter.reshape(3, 1), camup.reshape(3, 1))
        camviewshift = -np.dot(camviewmtx.transpose(), camviewshift)

        camfovy = 45 / 180.0 * np.pi
        camprojmtx = perspectiveprojectionnp(camfovy, 1.0 * 1.0 / 1.0)

        #####################################################
        # tfp_px3 = torch.from_numpy(p)
        # tfp_px3.requires_grad = True

        # tff_fx3 = torch.from_numpy(f)

        # tfuv_tx2 = torch.from_numpy(uv)
        # tfuv_tx2.requires_grad = True
        # tfft_fx3 = torch.from_numpy(ft)

        # tftex_thxtwx3 = torch.from_numpy(np.ascontiguousarray(texturenp))
        # tftex_thxtwx3.requires_grad = True

        tfcamviewmtx = torch.from_numpy(camviewmtx)
        tfcamshift = torch.from_numpy(camviewshift)
        tfcamproj = torch.from_numpy(camprojmtx)

        ##########################################################
        # tfp_1xpx3 = torch.unsqueeze(tfp_px3, dim=0)
        # tfuv_1xtx2 = torch.unsqueeze(tfuv_tx2, dim=0)
        # tftex_1xthxtwx3 = torch.unsqueeze(tftex_thxtwx3, dim=0)

        tfcamviewmtx_1x3x3 = torch.unsqueeze(tfcamviewmtx, dim=0)
        tfcamshift_1x3 = tfcamshift.view(-1, 3)
        tfcamproj_3x1 = tfcamproj

        # bs = 4
        # tfp_bxpx3 = tfp_1xpx3.repeat([bs, 1, 1])
        # tfuv_bxtx2 = tfuv_1xtx2.repeat([bs, 1, 1])
        # tftex_bxthxtwx3 = tftex_1xthxtwx3.repeat([bs, 1, 1, 1])

        tfcamviewmtx_bx3x3 = tfcamviewmtx_1x3x3.repeat([bs, 1, 1])
        tfcamshift_bx3 = tfcamshift_1x3.repeat([bs, 1])  

        tfcameras = [tfcamviewmtx_bx3x3.cuda(),
                 tfcamshift_bx3.cuda(),
                 tfcamproj_3x1.cuda()]
        return tfcameras

    # def compute_uvsampler(self,verts_t, faces_t, tex_size=2):
    #     """
    #     NOTE: copied from utils/mesh.py
    #     tex_size texture resolution per face default = 6
    #     TODO : merge with backbone

    #     For this mesh, pre-computes the UV coordinates for
    #     F x T x T points.
    #     Returns F x T x T x 2
    #     """
    #     verts = verts_t[0].clone().detach().cpu().numpy()
    #     faces = faces_t[0].clone().detach().cpu().numpy()
    #     # import pdb; pdb.set_trace()
    #     alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    #     beta = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    #     import itertools
    #     # Barycentric coordinate values
    #     coords = np.stack([p for p in itertools.product(*[alpha, beta])])
    #     vs = verts[faces]
    #     # Compute alpha, beta (this is the same order as NMR)
    #     v2 = vs[:, 2]
    #     v0v2 = vs[:, 0] - vs[:, 2]
    #     v1v2 = vs[:, 1] - vs[:, 2]    
    #     # F x 3 x T*2
    #     samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)    
    #     # F x T*2 x 3 points on the sphere 
    #     samples = np.transpose(samples, (0, 2, 1))

    #     # Now convert these to uv.
    #     uv = get_spherical_coords(samples.reshape(-1, 3))
    #     # uv = uv.reshape(-1, len(coords), 2)

    #     uv = uv.reshape(-1, tex_size, tex_size, 2)
    #     return uv