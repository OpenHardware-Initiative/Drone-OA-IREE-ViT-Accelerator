import airsimdroneracinglab as airsim
import time

class Drone(object):

    def __init__(self, drone_name="drone_1"):
        self.drone_name = drone_name
        self.airsim_client = airsim.MultirotorClient()
        
    # loads desired level
    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(sleep_sec)  # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=1):
        self.airsim_client.simStartRace(tier)

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset_race(self):
        self.airsim_client.simResetRace()

    def init_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

    def takeoff(self):
        self.airsim_client.takeoffAsync(vehicle_name=self.drone_name).join()
    
    def fly(self, x, y, z, V=5, sleep_sec=2.0):
        """
        Moves drone by given x, y, z values
        :param x: value to move in x direction
        :param y: value to move in y direction
        :param z: value to move in z direction
        :param V: speed of the drone
        """

        start_pos = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position
        
        self.airsim_client.moveToPositionAsync(
            start_pos.x_val + x,
            start_pos.y_val + y,
            start_pos.z_val + z,
            V,
            vehicle_name=self.drone_name
        ).join()
        time.sleep(sleep_sec)


def main():

    drone = Drone(drone_name="drone_1")

    drone.load_level(level_name="Qualifier_Tier_1")
    drone.start_race()
    drone.init_drone()
    drone.takeoff()
    drone.fly(x=10, y=10, z=0, V=5)
    

if __name__ == "__main__":
    main()
