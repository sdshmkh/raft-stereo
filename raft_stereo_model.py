import numpy as np
from PIL import Image
import torch

from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from core.utils.validation import *


import matplotlib.pyplot as plt
import cv2 as cv


get_device = lambda: "cuda" if torch.cuda.is_available() else "cpu"

def to_image_tensor(image):
    image = image.astype(np.uint8)
    img = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
    return img[None].to(get_device())


class RAFTStereoConfig:
    def __init__(self):
        self.context_norm = 'batch'
        self.corr_implementation = 'reg'
        self.corr_levels = 4
        self.corr_radius = 4
        self.hidden_dims = [128, 128, 128]
        self.mixed_precision = True
        self.n_downsample = 2
        self.n_gru_layers = 3
        self.restore_ckpt = 'models/raftstereo-middlebury.pth'
        self.shared_backbone = False
        self.slow_fast_gru = False
        self.valid_iters = 32

        # setup camera intrinsic params
        self.fx = 1038.244
        self.fy = 1041.935
        self.cx1 = 938.00
        self.cx2 = 953.29
        self.cy = 488.40
        self.baseline = 130 # in mm


class RaftStereodepthEstimation():
    def __init__(self, config=None):
        # init raft model
        if config:
            self.config = config
        else:
            self.config = RAFTStereoConfig()
        model = torch.nn.DataParallel(RAFTStereo(self.config), device_ids=[0])
        model.load_state_dict(torch.load(self.config.restore_ckpt))

        self.model = model.module
        self.model.to(get_device())
    
    def predict(self, left_image, right_image):
        # convert images to tensors
        left_image_tensor = to_image_tensor(left_image)
        right_image_tensor = to_image_tensor(right_image)

        # model eval
        self.model.eval()

        with torch.no_grad():
        # no grad, estimate depth map
            padder = InputPadder(left_image.shape, divis_by=32)
            image1, image2 = padder.pad(left_image_tensor, right_image_tensor)

            _, flow_up = self.model(image1, image2, iters=self.config.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()
            disparity = flow_up.cpu().numpy().squeeze()

        return disparity
    
    def depth_map(self, disparity):
        depth = (self.config.fx * self.config.baseline) / (-disparity + (self.config.cx2 - self.config.cx1)) 
        return depth

    def predict_depth(self, left_image, right_image):
        disparity = self.predict(left_image, right_image)
        depth = self.depth_map(disparity)
        
        H, W = depth.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        points_grid = np.stack(((xx-self.config.cx1)/self.config.fx, (yy-self.config.cy)/self.config.fy, np.ones_like(xx)), axis=0) * depth
        points_grid = points_grid.transpose(1, 2, 0)

        return points_grid, disparity




model = RaftStereodepthEstimation()


point_cloud_selector(model, ValidationOptions("datasets/staircase_validation_images", left_cam=0, right_cam=1))


