import numpy as np
import cv2
import time
import os
import random
from CarlaHandler import *

# Configurable Parameters
vehicle_id = 'vehicle.tesla.model3'
res = 500
ref_color = (124, 124, 124)  # BGR format
patch_size = 500  # Size of the adversarial patch


def main():

    file_n = 880 # file number to start from
    total_sample_n = 150 # files to generate
    sample_n = 0


    # Create directories
    os.makedirs('./dataset/reference', exist_ok=True)
    os.makedirs('./dataset/texture', exist_ok=True)
    os.makedirs('./dataset/rendered', exist_ok=True)

    # Initialize CARLA
    handler, n_spawn_points = initialize_carla()

    

    while True:
        # Check if the number of samples is reached
        if sample_n == total_sample_n:
            print("Sample limit reached. Exiting...")
            break

        try:
            # Randomize parameters
            handler.change_spawn_point(random.randint(1, n_spawn_points-1))
            handler.update_distance(random.randint(5, 10))
            handler.update_pitch(random.randint(0, 60))
            handler.update_yaw(random.randint(0, 359))
            handler.update_sun_altitude_angle(random.randint(20, 150))
            handler.update_sun_azimuth_angle(random.randint(0, 360))
            handler.world_tick(100)
            time.sleep(0.1)

            # Generate random color
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # 1. Get reference image (original color)
            handler.change_vehicle_color(ref_color)
            handler.world_tick(100)
            time.sleep(0.1)
            
            # Get images
            image = handler.get_image()  # RGB format
            seg_image = handler.get_segmentation()  # BGR format
            
            
            # Create vehicle mask (blue pixels in segmentation)
            vehicle_mask = (seg_image[:,:,0] == 255) & \
                        (seg_image[:,:,1] == 0) & \
                        (seg_image[:,:,2] == 0)
            
            # Validate vehicle mask
            if not validate_car_mask(vehicle_mask):
                print(f"⚠️ Sample {sample_n}: Invalid mask (multiple cars or no car detected)")
                continue

            # Extract reference car pixels
            reference_image = np.zeros_like(image)
            reference_image[vehicle_mask] = image[vehicle_mask]
            cv2.imwrite(f'./dataset/reference/{file_n}.png', reference_image)

            # 2. Create texture image (black background with rand_color car)
            texture_image = np.zeros_like(seg_image)  # Black background
            texture_image[vehicle_mask] = rand_color  # Apply random color to car pixels
            cv2.imwrite(f'./dataset/texture/{file_n}.png', cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR))

            # 3. Get rendered image (actual RGB appearance with new color)
            handler.change_vehicle_color(rand_color)
            handler.world_tick(100)
            time.sleep(0.1)  # Allow time for color change to render
            
            # Get the actual rendered RGB image
            rendered_rgb = handler.get_image()  # RGB format
            
            # Create rendered image (black background with actual rendered car)
            rendered_image = np.zeros_like(rendered_rgb)
            rendered_image[vehicle_mask] = rendered_rgb[vehicle_mask]
            cv2.imwrite(f'./dataset/rendered/{file_n}.png', rendered_image)

            print(f"✅ Sample {sample_n}: Generated reference, texture, and rendered images.")

            sample_n += 1
            file_n += 1

        except Exception as e:
            print(f"Error generating sample {sample_n}: {e}")
            continue

    if 'handler' in locals():
        handler.destroy_all_vehicles()
        del handler
        os.system('pkill -9 Carla')
        print("Cleanup completed")


def initialize_carla():
        """
        Initialize CARLA and spawn a vehicle.
        """

        handler = CarlaHandler(x_res=res, y_res=res, town='Town03')
        handler.world_tick(10)
        handler.destroy_all_vehicles()
        handler.world_tick(100)

        # Spawn vehicle
        handler.spawn_vehicle(vehicle_id)
        handler.update_view('3d')
        n_spawn_points = handler.get_spawn_points()
        return handler, n_spawn_points


def validate_car_mask(mask):
    """
    Returns True if mask contains exactly one connected car region.
    - mask: Binary vehicle mask (True/False or 0/255)
    """
    # Convert to uint8 if needed
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # Background counts as label 0, so:
    # num_labels = 1 (only background) → No car
    # num_labels = 2 → Exactly one car
    # num_labels > 2 → Multiple cars/objects
    return num_labels == 2  # Exactly one car


if __name__ == '__main__':
    main()