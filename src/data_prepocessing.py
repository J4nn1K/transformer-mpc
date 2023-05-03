import rosbag
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

dataset_name = 'obstacles12'

BAG_PATH = f'../data/session2023_05_01_{dataset_name}.bag'
OUT_PATH = f'../data/{dataset_name}.npz'

print('Loading rosbag...')
bag = rosbag.Bag(BAG_PATH)


# GET DATA FROM ROSBAG
print('Extracting topics...')
topics = ['/astra/color/image_raw',
          '/astra/depth/image_raw',
          '/local_map',
          '/ridgeback_velocity_controller/cmd_vel']

data = {}
for topic in topics:
  data[topic] = {'time': [], 'data': []}
    
for topic, msg, time in tqdm(bag.read_messages(topics=topics)):
  data[topic]['time'].append(time.to_nsec())
  
  if topic =='/astra/color/image_raw':
    data[topic]['data'].append(
      np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    )
  elif topic == '/astra/depth/image_raw':
    data[topic]['data'].append(
      np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    )
  elif topic == '/local_map':
    data[topic]['data'].append(
      np.array(msg.data).reshape(msg.info.height, msg.info.width)
    )
  elif topic == '/ridgeback_velocity_controller/cmd_vel':
    data[topic]['data'].append(
      np.array([msg.linear.x, msg.linear.y, msg.angular.z])
    )
  else: print('unknown topic')
  
print('Creating dataframes...')
  
color_images = pd.DataFrame.from_dict(data['/astra/color/image_raw'])
depth_images = pd.DataFrame.from_dict(data['/astra/depth/image_raw'])
maps = pd.DataFrame.from_dict(data['/local_map'])
cmd_vels = pd.DataFrame.from_dict(data['/ridgeback_velocity_controller/cmd_vel'])

df = pd.DataFrame()

df['time'] = color_images['time']
df['color_images'] = color_images['data']
df['depth_images'] = None
df['maps'] = None
df['cmd_vels'] = None


# TIMESTAMP MATCHING
print('Matching timestamps...')
for i, time in enumerate(df['time']):
  # find closest match in depth_images
  idx = depth_images['time'].sub(time).abs().idxmin()
  df['depth_images'][i] = depth_images['data'][idx]
  # find closest match in maps
  idx = maps['time'].sub(time).abs().idxmin()
  df['maps'][i] = maps['data'][idx]
  # find closest
  idx = cmd_vels['time'].sub(time).abs().idxmin()
  df['cmd_vels'][i] = cmd_vels['data'][idx]
  
  
# PROCESSING
print('Processing data...')
color_images = np.array(df['color_images'].to_list())

depth_images = np.array(df['depth_images'].to_list())
depth_images = np.expand_dims(depth_images, axis=-1)
depth_images = cv2.normalize(depth_images, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

maps = np.array(df['maps'].to_list())
maps[maps==100]=1

cmd_vels = np.array(df['cmd_vels'].to_list())


# DOWNSAMPLING
color_images=color_images[::3]
depth_images=depth_images[::3]
maps=maps[::3]
cmd_vels=cmd_vels[::3]


# SAVING
print('Saving data...')
np.savez_compressed(OUT_PATH, 
                    color_images=color_images, 
                    depth_images=depth_images,
                    maps=maps,
                    cmd_vels=cmd_vels)

print(f'Dataset saved to {OUT_PATH}')