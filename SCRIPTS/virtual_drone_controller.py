import airsim
import time
import sys
import numpy as np

class VirtualDroneController:
    """
    Connects to the AirSim simulation environment (Unreal Engine)
    and controls the virtual drone.
    """
    def __init__(self):
        print("Connecting to Unreal Engine / AirSim...")
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("Connected successfully!")
        except Exception as e:
            print(f"Failed to connect to simulation: {e}")
            print("Please ensure Unreal Engine is running and you have pressed 'Play'.")
            sys.exit(1)

    def takeoff(self):
        print("Taking off...")
        self.client.takeoffAsync().join()
        print("Drone is airborne.")

    def hover(self):
        print("Hovering...")
        self.client.hoverAsync().join()

    def get_camera_feed(self, camera_name="0", image_type=airsim.ImageType.Scene):
        """
        Retrieves the latest camera frame from the virtual drone.
        Returns the image as an OpenCV-compatible BGR numpy array.
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, image_type, False, False)
        ])
        
        if len(responses) > 0 and responses[0].image_data_uint8:
            response = responses[0]
            # Convert binary image data to numpy 1D array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
            # Reshape the 1D array to an OpenCV HxWxC 3D array (RGB/BGR)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            return img_rgb
        return None

    def move_to_position(self, x, y, z, velocity):
        """
        Moves the drone to local coordinates (North/East/Down) synchronously.
        """
        print(f"Moving to X:{x}, Y:{y}, Z:{z} at {velocity} m/s")
        self.client.moveToPositionAsync(x, y, z, velocity).join()

    def move_to_position_async(self, x, y, z, velocity):
        """
        Moves the drone asynchronously (doesn't block the camera feed).
        """
        print(f"Patrolling towards X:{x}, Y:{y}, Z:{z} at {velocity} m/s")
        return self.client.moveToPositionAsync(x, y, z, velocity)

    def move_by_velocity_body_async(self, vx, vy, vz, duration):
        """
        Flies at a constant velocity RELATIVE to the drone's current heading.
        vx: forward/back, vy: right/left, vz: down/up
        """
        return self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration)
        
    def move_velocity_world_async(self, vx, vy, vz, yaw_rate, duration):
        """
        Calculates world velocity manually to bypass AirSim BodyFrame bugs.
        """
        import math
        state = self.client.getMultirotorState()
        q = state.kinematics_estimated.orientation
        pitch, roll, current_yaw = airsim.to_eularian_angles(q)
        
        vx_world = vx * math.cos(current_yaw) - vy * math.sin(current_yaw)
        vy_world = vx * math.sin(current_yaw) + vy * math.cos(current_yaw)
        
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        return self.client.moveByVelocityAsync(
            vx_world, vy_world, vz, duration, 
            airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode
        )

    def rotate_by_yaw_rate_async(self, yaw_rate, duration):
        """
        Spins the drone at a constant rate (degrees per second)
        """
        return self.client.rotateByYawRateAsync(yaw_rate, duration)

    def hover(self):
        print("Hovering...")
        self.client.hoverAsync().join()
        
    def land(self):
        print("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

if __name__ == "__main__":
    # Test script: Connect, take off, spin around, land.
    drone = VirtualDroneController()
    drone.takeoff()
    
    # Move up 5 meters
    drone.move_to_position(0, 0, -5, 2)
    
    # Spin around
    for angle in range(0, 360, 45):
        drone.rotate_to_yaw(angle)
        time.sleep(1)
        
    drone.land()
    print("Test complete.")
