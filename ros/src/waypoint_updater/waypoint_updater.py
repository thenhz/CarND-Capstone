#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf as ros_tf
import math
import copy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100
KPH_TO_MPS = 1.0/3.6

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.track_waypoints = None               # Contains the waypoints of the track

        self.current_waypoint_id = None           # Holds the id of the closest current waypoint
        self.current_pose = None                  # Holds the current pose message
        self.current_yaw = None                   # Holds the current value of the yaw
        self.current_velocity = None              # Holds the current twist message

        self.waypoint_saved_speed = None          # Holds the previous value of suggested speed
        self.tl_waypoint_id = None                # Holds the
        self.behavior_state = None

        self.MAX_SPEED = KPH_TO_MPS*rospy.get_param("velocity")   # The Max speed the car needs to use

        self.LIMIT_DIST = rospy.get_param('limit_dist')
        self.BRAKE_DIST = rospy.get_param('brake_dist')
        self.HARD_LIMIT_DIST = rospy.get_param('hard_limit_dist')

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.update()
        rospy.spin()

    # publish next N waypoints to /final_waypoints interval rate
    def update(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.track_waypoints and self.current_pose and self.current_velocity:
                self.find_nearest_waypoint()
                self.publish_next_waypoints()
            rate.sleep()

    # find index of nearest waypoint in self.track_waypoints
    def find_nearest_waypoint(self):
        nearest_waypoint = [-1, 100000]  # index, ceiling for min distance
        car_coord = self.current_pose.pose.position

        for i in range(len(self.track_waypoints)):
            wp_coord = self.track_waypoints[i].pose.pose.position
            distance = self.euclid_distance(car_coord, wp_coord)
            direction = math.atan2(car_coord.y - wp_coord.y, car_coord.x - wp_coord.x)
            # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            angle_diff = math.atan2(math.sin(direction - self.current_yaw), math.cos(direction - self.current_yaw))
            if (distance < nearest_waypoint[1]) and (abs(angle_diff) < math.pi / 4.0):
                nearest_waypoint = [i, distance]
        self.current_waypoint_id = nearest_waypoint[0]

    def publish_next_waypoints(self):
        waypoints = Lane()
        waypoints.header.stamp = rospy.Time(0)
        waypoints.header.frame_id = self.current_pose.header.frame_id

        if (self.current_waypoint_id + LOOKAHEAD_WPS) < len(self.track_waypoints):
            waypoints.waypoints = copy.deepcopy(self.track_waypoints[self.current_waypoint_id: self.current_waypoint_id
                                                                                           + LOOKAHEAD_WPS])
        else:
            part_1 = copy.deepcopy(self.track_waypoints[self.current_waypoint_id:])
            part_2 = copy.deepcopy(self.track_waypoints[:LOOKAHEAD_WPS-len(part_1)])
            waypoints.waypoints = part_1 + part_2

        waypoints.waypoints = self.behavior(waypoints.waypoints)
        self.final_waypoints_pub.publish(waypoints)

    def pose_cb(self, msg):
        self.current_pose = msg
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.current_yaw = ros_tf.transformations.euler_from_quaternion(quaternion)[2]

    def velocity_cb(self, msg):
        self.current_velocity = msg

    def waypoints_cb(self, waypoints):
        self.track_waypoints = waypoints.waypoints
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        self.tl_waypoint_id = msg.data

    def constant_speed(self, waypoints, speed):
        #This is needed to reset the braking behavior
        self.waypoint_saved_speed = None
        for i in range(LOOKAHEAD_WPS):
            waypoints[i].twist.twist.linear.x = speed
        return waypoints

    def following(self, waypoints, multiplier = 1.0):
        self.waypoint_saved_speed = None
        for i in range(LOOKAHEAD_WPS):
            waypoints[i].twist.twist.linear.x *= multiplier
        return waypoints

    def brake(self, waypoints):
        stop_waypoint = self.tl_waypoint_id
        current_speed = self.current_velocity.twist.linear.x

        print "Breaking, stop waypoint index : ", stop_waypoint
        print "Current speed: ", current_speed

        dist_to_stop_wp = self.wp_distance(self.current_waypoint_id, stop_waypoint)
        print "Distance to stop_waypoint: ", dist_to_stop_wp

        wp_to_stop = stop_waypoint - self.current_waypoint_id

        #TODO Understand better this parameter (5), can we evaluate it better?
        wp_to_stop = min(wp_to_stop - 5, LOOKAHEAD_WPS)
        print "Waypoints until stops: ", wp_to_stop

        if self.waypoint_saved_speed is None:
            self.waypoint_saved_speed = current_speed

        for i in range(wp_to_stop, LOOKAHEAD_WPS):
            waypoints[i].twist.twist.linear.x = 0.0
        for i in range(wp_to_stop):
            waypoint_velocity = self.waypoint_saved_speed * math.sqrt(
                self.wp_distance(self.current_waypoint_id + i,
                                 stop_waypoint) / dist_to_stop_wp)
            waypoints[i].twist.twist.linear.x = waypoint_velocity

        self.waypoint_saved_speed = waypoints[1].twist.twist.linear.x
        return waypoints

    def eval_behavior(self):

        # Traffic light with red light detected
        if self.tl_waypoint_id >= 0:
            distance_to_tl = self.wp_distance(self.current_waypoint_id, self.tl_waypoint_id)
            print "Distance to TL: ", distance_to_tl
            if distance_to_tl > self.LIMIT_DIST:
                self.behavior_state = "HIGH"
            elif self.BRAKE_DIST < distance_to_tl < self.LIMIT_DIST:
                self.behavior_state = "LOW"
            elif self.HARD_LIMIT_DIST < distance_to_tl < self.BRAKE_DIST:
                self.behavior_state = "BRAKE"
            else:
                self.behavior_state = "EXTREME"

        # No traffic light detected
        if self.tl_waypoint_id == -1:
            self.behavior_state = "HIGH"

        # Unsure of trafficl light detection
        if self.tl_waypoint_id == -2:
            self.behavior_state = "DANGER"

        # Traffic light detected with Green light
        if self.tl_waypoint_id == -3:
            self.behavior_state = "LOW"

    def behavior(self, waypoints):

        # If no trafficlight messages have yet arrived, do not move and wait for the first one
        if self.tl_waypoint_id is None:
            print "Initializing TL Detector...waiting for first message..."
            return self.constant_speed(waypoints, 0.0)

        HIGH_SPEED = self.MAX_SPEED
        LOW_SPEED = self.MAX_SPEED*0.75
        DANGER_SPEED = self.MAX_SPEED*0.4

        self.eval_behavior()

        if self.behavior_state == "HIGH":
            print "Traffic light far from horizon..."
            #return self.constant_speed(waypoints, HIGH_SPEED)
            return self.following(waypoints)
        elif self.behavior_state == "LOW":
            print "Approaching a traffic light green light detected..."
            #return self.constant_speed(waypoints, LOW_SPEED)
            return self.following(waypoints, 0.75)
        elif self.behavior_state == "EXTREME":
            print "Red detected, very close full brake..."
            return self.constant_speed(waypoints, 0.0)
        elif self.behavior_state == "DANGER":
            print "Traffic light detection not clear...slowing down to be sure..."
            #return self.constant_speed(waypoints, DANGER_SPEED)
            return self.following(waypoints, 0.4)
        else:
            print "Red detected, beginning braking..."
            return self.brake(waypoints)

    def wp_distance(self, wp1, wp2):
        # TODO Attention when the track closes...here it gives problems use modulo...(not needed in the simulator but may be needed in CARLA)
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        if wp1 <= wp2:
            for i in range(wp2 - wp1):
                dist += dl(self.track_waypoints[wp1 + i].pose.pose.position,
                           self.track_waypoints[wp1 + i + 1].pose.pose.position)
            return dist
        else:
            for i in range(wp1 - wp2):
                dist += dl(self.track_waypoints[wp2 + i].pose.pose.position,
                           self.track_waypoints[wp2 + i + 1].pose.pose.position)
            return -dist

    # calculate the euclidian distance between our car and a waypoint
    # TODO This is a static method
    def euclid_distance(self, car_pos, wpt_pos):
        a = np.array((car_pos.x, car_pos.y))
        b = np.array((wpt_pos.x, wpt_pos.y))
        return np.linalg.norm(a - b)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')