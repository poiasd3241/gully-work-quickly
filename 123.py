import os
import rospy
import cv2
from clover import srv
from clover.srv import SetLEDEffect
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import pigpio

rospy.init_node('flight')
bridge = CvBridge()
pi = pigpio.pi()

navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)

model = YOLO(os.path.expanduser('~/Desktop/best.pt'))
debug_image = rospy.Publisher('/person_debug', Image, queue_size=1)
save_dir = os.path.expanduser('~/Desktop/Модуль_Б_Ф_И')
os.makedirs(save_dir, exist_ok=True)
pi.set_mode(13, pigpio.OUTPUT)

set_effect(effect='blink', r=0, g=255, b=0)
navigate(x=0, y=0, z=1, speed=1, frame_id='body', auto_arm=True)
rospy.sleep(5)

results = model(frame)
    
for marker_id in [1, 2, 3]:
    navigate(x=0, y=0, z=1, speed=0.5, frame_id='aruco_' + str(marker_id))
    rospy.sleep(5)

    message = rospy.wait_for_message('/main_camera/image_raw', Image)
    frame = bridge.imgmsg_to_cv2(message, 'bgr8')
    result = model.predict(frame, verbose=False)[0]
    marked_frame = result.plot()
    debug_image.publish(bridge.cv2_to_imgmsg(marked_frame, 'bgr8'))

    for person_number, box in enumerate(result.boxes, start=1):
        person_type = result.names[int(box.cls[0].item())]
        print('Найден человек - {}'.format(person_type))
        cv2.imwrite(
            os.path.join(save_dir, 'marker{}_person{}.jpg'.format(marker_id, person_number)),
            marked_frame)

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print('Координаты рамки:',x1, y1, x2, y2)

        if person_type == 'браконьер':
            print('\a', end='')
            for _ in range(6):
                set_effect(effect='fill', r=255, g=0, b=0)
                rospy.sleep(0.25)
                set_effect(effect='fill', r=0, g=0, b=255)
                rospy.sleep(0.25)
        elif person_type == 'грибник':
            set_effect(effect='fill', r=255, g=255, b=0)
            rospy.sleep(3)
            pi.set_servo_pulsewidth(13, 1000)
        elif person_type == 'турист':
            set_effect(effect='fill', r=160, g=0, b=255)
            rospy.sleep(3)
            pi.set_servo_pulsewidth(13, 1000)
rospy.sleep(3)

navigate(x=0, y=0, z=1, speed=1, frame_id='aruco_0')
set_effect(effect='fill', r=255, g=0, b=0)
rospy.sleep(1)
land()
