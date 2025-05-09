import carla
import random
import time

client = carla.Client('localhost', 2000)
client.set_timeout(30.0)  # Set timeout for client requests
#client.load_world('Town10HD')
#time.sleep(5)  # Wait for the world to load
world = client.get_world()
weather = world.get_weather()
actors_list = world.get_actors()#.filter('vehicle.*')


print(actors_list)

time.sleep(10)

exit()


# Configure CARLA world settings
settings = world.get_settings()
settings.no_rendering_mode = True
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

