import numpy as np
import cv2
import time
import os
from CarlaHandler import *

def main():
    # Create a random patch (BGR format)
    
    patch_size = 16  # Size of the adversarial patch
    patch = np.full((patch_size, patch_size, 3), (0, 255, 255), dtype=np.uint8)
    #patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
 
    # Initialize CARLA with custom resolution
    handler = CarlaHandler(x_res=600, y_res=400)
    handler.world_tick(10)
    
    # Spawn and configure vehicle
    handler.spawn_vehicle('vehicle.tesla.model3')
    handler.update_view('3d')
    handler.change_spawn_point()
    handler.update_distance(10)
    handler.update_pitch(15)
    handler.update_yaw(100)
    handler.change_vehicle_color((255, 255, 0))  # Yellow (BGR)
    handler.update_sun_altitude_angle(45)
    handler.world_tick(100)
    time.sleep(2)  # Wait for changes to apply

    # Get images
    image = handler.get_image()  # RGB format
    seg_image = handler.get_segmentation()  # BGR format with cars in blue
    
    # Create precise vehicle mask (blue pixels in segmentation)
    vehicle_mask = (seg_image[:,:,0] == 255) & \
                   (seg_image[:,:,1] == 0) & \
                   (seg_image[:,:,2] == 0)
    
    # Calculate required tiling to cover the entire image
    h, w = image.shape[:2]
    tiles_x = int(np.ceil(w / patch_size))
    tiles_y = int(np.ceil(h / patch_size))
    
    # Create tiled texture that exactly matches image dimensions
    tiled_texture = np.tile(patch, (tiles_y, tiles_x, 1))[:h, :w]
    
    # Apply texture only to vehicle pixels
    textured_vehicle = image.copy()
    textured_vehicle[vehicle_mask] = tiled_texture[vehicle_mask]
    
    # Display and save results
    cv2.imshow('Original Vehicle', image)
    cv2.imshow('Textured Vehicle', textured_vehicle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save images (convert RGB to BGR for saving)
    os.makedirs('./data', exist_ok=True)
    cv2.imwrite('./data/original.png', image)
    cv2.imwrite('./data/segmentation.png', seg_image)
    cv2.imwrite('./data/textured_vehicle.png', textured_vehicle)
    
    # Cleanup
    del handler

if __name__ == '__main__':
    main()