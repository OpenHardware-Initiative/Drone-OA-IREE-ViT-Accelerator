#!/usr/bin/env python3
"""
Streams (1) state vector and (2) depth images from Flightmare to an FPGA
over UDP.

 * State packet  :  4 floats   [ stamp | vx | vy | vz ]      → 16 bytes
 * Image packet¹ :  1 uint32   [ seq  ]                      →  4 bytes
                    N bytes    [ PNG-compressed depth image ]

¹One PNG frame may exceed the Ethernet MTU.  For first tests use the
network’s default (the kernel will IP-fragment it).  Once things work,
switch to explicit chunking or jumbo frames.
"""
import socket, struct, rospy, cv2
from dodgeros_msgs.msg  import QuadState
from sensor_msgs.msg    import Image
from cv_bridge          import CvBridge

FPGA_IP      = '10.42.0.42'
FPGA_PORT    = 5005
STATE_PORT   = 5006          # keep images and state on separate ports
bridge       = CvBridge()
udp_state    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_img      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_state.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)
udp_img  .setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)

IMG_SEQ = 0                  # monotonically increasing image counter

def state_cb(msg: QuadState):
    print(f"[udp_streamer] state_cb: {msg.header.stamp.to_sec()}")
    stamp = msg.header.stamp.to_sec()
    # timestamp (you could also use msg.t if you prefer)
    stamp = msg.header.stamp.to_sec()
    # QuadState.velocity is a geometry_msgs/Twist:
    vx = msg.velocity.linear.x
    vy = msg.velocity.linear.y
    vz = msg.velocity.linear.z

    pkt = struct.pack('<4f', stamp, vx, vy, vz)
    udp_state.sendto(pkt, (FPGA_IP, STATE_PORT))

def depth_cb(img_msg: Image):
    print(f"[udp_streamer] depth_cb: {img_msg.header.stamp.to_sec()}")
    global IMG_SEQ
    # convert 16-bit depth image → uint16 numpy array
    depth = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
    # compress; depth is already 0-1 m float in sim, keep as 16-bit PNG
    ok, buf = cv2.imencode('.png', depth)
    if not ok:
        rospy.logwarn_once("PNG encode failed")
        return
    header  = struct.pack('<I', IMG_SEQ)
    udp_img.sendto(header + buf.tobytes(), (FPGA_IP, FPGA_PORT))
    IMG_SEQ += 1 & 0xFFFFFFFF            # wrap after 2³²–1

if __name__ == '__main__':
    rospy.init_node('udp_streamer', anonymous=False)
    rospy.Subscriber('/kingfisher/dodgeros_pilot/state',
                     QuadState, state_cb,  queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/kingfisher/dodgeros_pilot/unity/depth',
                     Image,     depth_cb,  queue_size=1, tcp_nodelay=True)
    rospy.spin()
