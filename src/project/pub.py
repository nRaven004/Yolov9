#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def talker():
    rospy.init_node('pub1', anonymous=True)
    pub = rospy.Publisher('Test', String, queue_size=10)
    rate = rospy.Rate(0.1)  # 1 Hz
    while not rospy.is_shutdown():
        msg = "Hello from pub1"
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

