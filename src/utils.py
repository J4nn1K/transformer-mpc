import cv2
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import torch


def save_grids_as_video(grids, indices, video_filename, freq_out, upscale_factor=8):
    if not indices:
        print("No indices provided. Nothing to save.")
        return

    # Get the dimensions of the grids and upscale them
    height, width = grids[0].shape
    upscaled_width = int(width * upscale_factor)
    upscaled_height = int(height * upscale_factor)

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, freq_out, (upscaled_width, upscaled_height), isColor=True)

    # Define the color map
    cmap = ListedColormap(['grey', 'white', 'black'])

    for index in indices:
        # Get the grid to save and apply the color map
        grid_to_save = grids[index]
        grid_color_mapped = cmap(grid_to_save + 1) / 2  # Normalize values to [0, 1])

        # Convert the grid to the correct format for the video writer
        grid_BGR = (grid_color_mapped[:, :, :3] * 255).astype(np.uint8)
        grid_BGR = cv2.cvtColor(grid_BGR, cv2.COLOR_RGB2BGR)

        # Resize the frame
        grid_resized = cv2.resize(grid_BGR, (upscaled_width, upscaled_height), interpolation=cv2.INTER_AREA)

        # Write the frame to the video
        video_writer.write(grid_resized)

    # Close the video writer
    video_writer.release()
    print("Video saved as", video_filename)
    
def save_images_as_video(images, indices, video_filename, freq_out, upscale_factor=1):
    if not indices:
        print("No indices provided. Nothing to save.")
        return

    # Get the dimensions of the grids and upscale them
    height, width, _ = images[0].shape
    upscaled_width = int(width * upscale_factor)
    upscaled_height = int(height * upscale_factor)

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, freq_out, (upscaled_width, upscaled_height), isColor=True)

    for index in indices:
        # Convert the grid to the correct format for the video writer
        grid_BGR = cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR)

        # Resize the frame
        grid_resized = cv2.resize(grid_BGR, (upscaled_width, upscaled_height), interpolation=cv2.INTER_AREA)

        # Write the frame to the video
        video_writer.write(grid_resized)

    # Close the video writer
    video_writer.release()
    print("Video saved as", video_filename)
    

def get_model_output(model, state, dataset, indices):
  commands ={'target': [], 'pred': []}
  for i in tqdm(indices):
    grid, u_target = dataset.__getitem__(i)
    grid = torch.unsqueeze(grid, 0).numpy()
  
    u_pred = model.apply({'params': state.params}, grid, train=False)
  
    commands['target'].append(u_target[0,0])
    commands['pred'].append(u_pred[0,0,0])
    
  return commands