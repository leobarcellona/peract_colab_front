# Adapted from https://github.com/stepjam/RLBench/blob/master/rlbench/utils.py

import os
import pickle
import numpy as np
from PIL import Image

from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor

# constants
EPISODE_FOLDER = 'episode%d'

CAMERA_FRONT = 'front'
CAMERAS = [CAMERA_FRONT]

IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_TYPES = [IMAGE_RGB, IMAGE_DEPTH]
IMAGE_FORMAT  = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'

DEPTH_SCALE = 2**24 - 1

# functions
def get_stored_demo(data_path, index, cameras):
  episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
  
  # low dim pickle file
  with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
    obs = pickle.load(f)

  # variation number
  with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
    obs.variation_number = pickle.load(f)

  num_steps = len(obs)
  for i in range(num_steps):
    for camera in cameras:

      near = obs[i].misc['%s_camera_near' % (camera)]
      far = obs[i].misc['%s_camera_far' % (camera)]

      decompressed_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
      decompressed_depth = near + decompressed_depth * (far - near)
      point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(decompressed_depth,
                                                                         obs[i].misc['%s_camera_extrinsics' % camera],
                                                                         obs[i].misc['%s_camera_intrinsics' % camera])

      setattr(obs[i], '%s_rgb' % camera,  np.array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_RGB), IMAGE_FORMAT % i))))
      setattr(obs[i], '%s_depth' % camera, decompressed_depth)
      setattr(obs[i], '%s_point_cloud' % camera, point_cloud)


      #obs[i].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_RGB), IMAGE_FORMAT % i)))
      #obs[i].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
      #obs[i].front_depth = near + obs[i].front_depth * (far - near)
      #obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth,
      #                                                                                obs[i].misc['%s_camera_extrinsics' % camera],
      #                                                                                obs[i].misc['%s_camera_intrinsics' % camera])

  return obs
