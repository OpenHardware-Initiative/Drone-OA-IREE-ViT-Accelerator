#!/usr/bin/python3
import argparse
import rospy
from dodgeros_msgs.msg import Command, QuadState
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from envsim_msgs.msg import ObstacleArray

import time
import numpy as np
import pandas as pd
import os
import threading
import socket
from copy import deepcopy
import cv2
import traceback  # <-- Make sure this import is present

# --- Custom Imports for Remote Inference ---
from fpga_link_v1 import pack_frame, unpack_reply, PORT, FPGA_IP
from utils import AgileCommandMode, AgileQuadState
from user_code import compute_command_state_based


class AgilePilotNode:
    def __init__(self, vision_based=False, desVel=None, keyboard=False):
        print("[RUN_COMPETITION] Initializing agile_pilot_node with ASYNC remote inference...")
        rospy.init_node("agile_pilot_node", anonymous=False)

        self.vision_based = vision_based
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        self.keyboard = keyboard
        quad_name = "kingfisher"

        # --- Asynchronous Control & Inference State ---
        self.last_commanded_velocity = [1.0, 0.0, 0.0]
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()
        self.desiredVel = desVel
        print(f"\n[RUN_COMPETITION] Desired velocity = {self.desiredVel}\n")

        # --- Data Logging (preserved from original script) ---
        self.col = None
        self.t1 = 0
        self.last_valid_img = None
        self.data_log = pd.DataFrame({'timestamp':[], 'desired_vel':[], 'quat_1':[], 'quat_2':[], 'quat_3':[], 'quat_4':[], 'pos_x':[], 'pos_y':[], 'pos_z':[], 'vel_x':[], 'vel_y':[], 'vel_z':[], 'velcmd_x':[], 'velcmd_y':[], 'velcmd_z':[], 'ct_cmd':[], 'br_cmd_x':[], 'br_cmd_y':[], 'br_cmd_z':[], 'is_collide': [],})
        self.count = 0
        self.time_interval = .03
        self.folder = f"train_set/{int(time.time()*100)}" 
        os.makedirs(self.folder, exist_ok=True)
        self.depth_im_threshold = 0.09
        self.curr_cmd = None

        # --- Networking Setup ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)

        # --- Subscribers & Publishers ---
        self.start_sub = rospy.Subscriber(f"/{quad_name}/start_navigation", Empty, self.start_callback, queue_size=1, tcp_nodelay=True)
        self.odom_sub = rospy.Subscriber(f"/{quad_name}/dodgeros_pilot/state", QuadState, self.state_callback, queue_size=1, tcp_nodelay=True)
        self.img_sub = rospy.Subscriber(f"/{quad_name}/dodgeros_pilot/unity/depth", Image, self.img_callback, queue_size=1, tcp_nodelay=True)
        self.cmd_sub = rospy.Subscriber(f"/{quad_name}/dodgeros_pilot/command", Command, self.cmd_callback, queue_size=1, tcp_nodelay=True)
        self.obstacle_sub = rospy.Subscriber(f"/{quad_name}/dodgeros_pilot/groundtruth/obstacles", ObstacleArray, self.obstacle_callback, queue_size=1, tcp_nodelay=True)
        
        self.linvel_pub = rospy.Publisher(f"/{quad_name}/dodgeros_pilot/velocity_command", TwistStamped, queue_size=1)
        self.command_timer = rospy.Timer(rospy.Duration(0.02), self.publish_continuous_command)

        print("[RUN_COMPETITION] Initialization completed!")

    def start_callback(self, data):
        self.publish_commands = True

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)
        
    def cmd_callback(self, msg):
        self.curr_cmd = msg
        
    def obstacle_callback(self, obs_data):
        if self.state is None: return
        self.col = self.if_collide(obs_data.obstacles[0])
        if self.vision_based: return
        
    def if_collide(self, obs):
        dist = np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z]))
        margin = dist - obs.scale
        return margin < 0 or self.state.pos[2] <= 0.01

    def img_callback(self, img_data):
        if not self.vision_based or self.state is None:
            return

        if self.inference_in_progress:
            return

        self.inference_in_progress = True
        img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
        if self.last_valid_img is None: self.last_valid_img = deepcopy(img)
        self.last_valid_img = deepcopy(img) if np.min(img) > 0.0 else self.last_valid_img
        
        thread = threading.Thread(target=self.inference_thread, args=(self.last_valid_img, deepcopy(self.state)))
        thread.daemon = True
        thread.start()

        # Data Logging Logic
        if (self.state.t - self.t1 > self.time_interval or self.t1 == 0) and self.state.pos[0] < 63:
            self.t1 = self.state.t
            timestamp = round(self.state.t, 3)
            # ... (logging code remains the same)

    def inference_thread(self, img, state):
        """ Runs in a background thread to handle network communication with the FPGA. """
        try:
            img_processed = np.clip(img / self.depth_im_threshold, 0, 1)
            depth_u8 = np.round(img_processed * 255).astype(np.uint8)
            quat = np.array([state.att[0], state.att[1], state.att[2], state.att[3]])
            pos_x = float(state.pos[0])

            pkt = pack_frame(depth_u8, float(self.desiredVel), pos_x, quat)
            self.sock.sendto(pkt, (FPGA_IP, PORT))

            reply, _ = self.sock.recvfrom(8192)
            velocity_cmd = unpack_reply(reply)

            with self.inference_lock:
                self.last_commanded_velocity = velocity_cmd.tolist()

        except Exception as e:
            # THIS IS THE CRITICAL PART - IT WILL PRINT THE FULL ERROR
            print("\n" + "="*20 + " INFERENCE THREAD ERROR " + "="*20)
            print(traceback.format_exc())
            print("="*62 + "\n")
        finally:
            self.inference_in_progress = False

    def publish_continuous_command(self, event):
        """ Called by a rospy.Timer at 50Hz to ensure smooth, continuous control. """
        if not self.publish_commands or self.state is None:
            return

        vel_msg = TwistStamped()
        vel_msg.header.stamp = rospy.Time(self.state.t)

        with self.inference_lock:
            vel_msg.twist.linear.x = self.last_commanded_velocity[0]
            vel_msg.twist.linear.y = self.last_commanded_velocity[1]
            vel_msg.twist.linear.z = self.last_commanded_velocity[2]
        
        vel_msg.twist.angular.z = 0.0
        self.linvel_pub.publish(vel_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot with remote inference.")
    parser.add_argument("--vision_based", help="Fly vision-based", action="store_true")
    parser.add_argument('--des_vel', type=float, default=5.0, help='desired velocity for quadrotor')
    parser.add_argument("--keyboard", help="Fly state-based mode", required=False, dest="keyboard", action="store_true")
    args = parser.parse_args()

    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, desVel=args.des_vel, keyboard=args.keyboard)
    rospy.spin()