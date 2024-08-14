### Importing Libraries ###
import cv2
import mediapipe as mp
import numpy as np
import math

### Mediapipe Model Initialization ###
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

### Angle Calculation Function ###
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle 

### Setting both stage and feedback to None to start ###
stage = None
feedback = None 

### Using cv2 to capture video ###

cap = cv2.VideoCapture(0)


##Mediapipe Instance 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        #Give me current feed from webcam, frame is the image, we don't really need ret
        ret, frame = cap.read()

        #Recoloring our current image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Make detection using holistic from above
        results = holistic.process(image)

        #Re-coloring back to normal
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        #Use a try and except when all joints are not visible 
        try:
            landmarks = results.pose_landmarks.landmark
            
            #Get coordinates for important body parts
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


            #Calculate Right Armpit Angle
            r_armpit_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

            #Calculate Left Armpit Angle
            l_armpit_angle = calculate_angle(l_hip,l_shoulder,l_elbow)

            #Calculate Left Shoulder Angle
            l_shoulder_attack_angle = calculate_angle(l_elbow,l_shoulder,r_shoulder)

            #Calculate Left ELbow Angle
            left_elbow_angle = calculate_angle(l_shoulder,l_elbow,l_wrist)

            #Calculate Right Elbow Angle
            right_elbow_angle = calculate_angle(r_shoulder,r_elbow,r_wrist)

            # Calculate distance between both wrists
            wrist_distance = math.sqrt((r_wrist[0] - l_wrist[0]) ** 2 + (r_wrist[1] - l_wrist[1]) ** 2)

            # Distance between both elbows 
            elbow_distance = math.sqrt((r_elbow[0] - l_elbow[0]) ** 2 + (r_elbow[1] - l_elbow[1]) ** 2)
            
            #Visualize Armpit Angle#
            r_elbow_coords = np.multiply(r_elbow, [640, 480]).astype(int)
            r_elbow_pos = (r_elbow_coords[0], r_elbow_coords[1])
            r_elbow_pos_1 = (r_elbow_coords[0], r_elbow_coords[1]-20)
            
            ### ANYTHING WITH cv2. is shows the angle or the distance on the screen...I've commented them out for now
            
            #cv2.putText(image, str(int(r_armpit_angle)), 
             #           r_elbow_pos, 
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            #cv2.putText(image, 'Right Armpit Angle', 
             #       r_elbow_pos_1, 
              #      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            #Visualize Armpit Angle#

            ###### DISTANCE RIGHT ELBOW LABELING #######

            wrist_coords = np.multiply(l_wrist, [640, 480]).astype(int)
            
            text_position = (wrist_coords[0]+200, wrist_coords[1]+200)
            text_position2 = (wrist_coords[0]+200, wrist_coords[1] - 180)
    
    
            #cv2.putText(image, 'Right Elbow Angle', 
             #               text_position, 
              #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            #cv2.putText(image, str(round(right_elbow_angle,2)), 
             #           tuple(text_position2), 
              #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            ###### DISTANCE RIGHT ELBOW LABELING #######
            
            
            ###### DISTANCE BETWEEN WRISTS LABELING #######

            wrist_coords = np.multiply(l_wrist, [640, 480]).astype(int)
            
            text_position = (wrist_coords[0], wrist_coords[1])
            text_position2 = (wrist_coords[0], wrist_coords[1] - 20)
    
    
            #cv2.putText(image, 'Wrist Distance', 
                  #          text_position, 
                   #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            #cv2.putText(image, str(round(wrist_distance,2)), 
             #           tuple(text_position2), 
              #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            ###### DISTANCE BETWEEN WRISTS LABELING #######

            ###### DISTANCE BETWEEN ELBOWS LABELING #######

            l_elbow_coords = np.multiply(l_elbow, [640, 480]).astype(int)
            
            text_position = (l_elbow_coords[0]+90, l_elbow_coords[1]+90)
            text_position2 = (l_elbow_coords[0] + 90, l_elbow_coords[1] + 70)
    
    
            #cv2.putText(image, 'Elbow Distance', 
             #               text_position, 
              #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            #cv2.putText(image, str(round(elbow_distance,2)), 
             #           tuple(text_position2), 
              #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            ###### DISTANCE BETWEEN ELBOWS LABELING #######

            ###### LEFT ELBOW ANGLE LABELING #######
            
            #Finding Location of Left Elbow on Image
            l_elbow_coords = np.multiply(l_shoulder, [640, 480]).astype(int)
            
            left_elbow_position = (l_elbow_coords[0], l_elbow_coords[1] - 20)
            
            #cv2.putText(image, 'Left Elbow Angle', 
             #           left_elbow_position, 
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            #Placing left elbow angle next to the left elbow 
            #cv2.putText(image, str(int(left_elbow_angle)), 
             #          tuple(l_elbow_coords), 
              #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            ###### LEFT ELBOW ANGLE LABELING #######

            ###### LEFT ARMPIT ANGLE #######
    
            text_position = (r_elbow_coords[0] + 80, r_elbow_coords[1] + 80)
            text_position2 = (r_elbow_coords[0] + 80, r_elbow_coords[1] + 100)
            
            # Visualize "Arm Angle" text
            #cv2.putText(image, 'Left Armpit Angle', 
             #           text_position, 
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            #cv2.putText(image, str(int(l_armpit_angle)), 
             #          tuple(text_position2), 
              #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
            ###### LEFT ARMPIT ANGLE ######


            ###### LEFT HITTING SHOULDER ANGLE #######
    
            text_position = (r_elbow_coords[0] + 160, r_elbow_coords[1] + 160)
            text_position2 = (r_elbow_coords[0] + 160, r_elbow_coords[1] + 200)
            
            # Visualize "Arm Angle" text
            #cv2.putText(image, 'Left Hitting Shoulder Angle', 
             #           text_position, 
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            #cv2.putText(image, str(int(l_shoulder_attack_angle)), 
             #          tuple(text_position2), 
              #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
            ###### LEFT HITTING SHOULDER ANGLE ######

            ###### LEFT HAND MEASUREMENT #######
            
            l_hand_coords = np.multiply(l_thumb, [640, 480]).astype(int)
            l_text_position = (l_hand_coords[0] + 160, l_hand_coords[1] + 160)
            l_text_position2 = (l_hand_coords[0] + 160, l_hand_coords[1] + 200)
            
            ###### LEFT HAND MEASUREMENT #######
            l_hand_coords = np.multiply(l_thumb, [640, 480]).astype(int)
            l_text_position = (l_hand_coords[0] + 160, l_hand_coords[1] + 160)
            l_text_position2 = (l_hand_coords[0] + 160, l_hand_coords[1] + 200)


        except Exception as e:
            print(e)
            pass
    
        # Handle Hand Landmarks separately
        try:
            r_thumb_tip = [results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x, 
                           results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y]
            r_pinky_tip = [results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x, 
                           results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y]
            l_thumb_tip = [results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x, 
                           results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y]
            l_pinky_tip = [results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x, 
                           results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y]
            
            # Calculate the distance between the thumb tip and pinky tip
            r_thumb_pinky_distance = math.sqrt((r_thumb_tip[0] - r_pinky_tip[0]) ** 2 + 
                                             (r_thumb_tip[1] - r_pinky_tip[1]) ** 2)

            # Convert coordinates to image size for displaying
            r_thumb_coords = np.multiply(r_thumb_tip, [640, 480]).astype(int)
            r_pinky_coords = np.multiply(r_pinky_tip, [640, 480]).astype(int)


            # Display the distance between thumb and pinky tips
            #cv2.putText(image, f'Right Thumb-Pinky Dist: {r_thumb_pinky_distance:.2f}', 
             #           (r_thumb_coords[0], r_thumb_coords[1] + 40), 
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Calculate the distance between the thumb tip and pinky tip
            l_thumb_pinky_distance = math.sqrt((l_thumb_tip[0] - l_pinky_tip[0]) ** 2 + 
                                             (l_thumb_tip[1] - l_pinky_tip[1]) ** 2)

            # Convert coordinates to image size for displaying
            l_thumb_coords = np.multiply(l_thumb_tip, [640, 480]).astype(int)
            l_pinky_coords = np.multiply(l_pinky_tip, [640, 480]).astype(int)


            # Display the distance between thumb and pinky tips
            #cv2.putText(image, f'Left Thumb-Pinky Dist: {l_thumb_pinky_distance:.2f}', 
             #           (l_thumb_coords[0]+40, l_thumb_coords[1] + 80), 
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass
     
        ### THIS IS THE MAIN LOGIC FOR EACH MOVEMENT. Using a combination of angles and distances

        if r_armpit_angle >= 120 and wrist_distance < 0.20:
            stage = 'Setting'
            #if wrist_distance > 0.20:
             #   feedback = 'Hands Closer Together!'
            if elbow_distance < 0.25:
                feedback = 'Elbows Further Apart!'
            elif elbow_distance > 0.34:
                feedback = 'Elbows Closer Together!'
            elif l_thumb_pinky_distance < 0.06 or r_thumb_pinky_distance < 0.06:
                feedback = 'Open Your Hands!'
            ##Do open hands!!!
            else:
                feedback = 'Good!'
        elif (r_armpit_angle >= 130 and l_armpit_angle >= 130) and left_elbow_angle > 165 and right_elbow_angle > 165:
            stage = 'Blocking'
            
            if l_thumb_pinky_distance < 0.06 or r_thumb_pinky_distance < 0.06:
                feedback = 'Open Your Hands!'
            else:
                feedback = 'Good!'
        elif (r_armpit_angle < 100 and wrist_distance < 0.09):
            stage = 'Passing'
            if r_armpit_angle > 30:
                feedback = 'Lower Both Arms!'
            elif left_elbow_angle < 100:
                feedback = "Arms Straight!"
            else:
                feedback = 'Good!'
        elif l_armpit_angle > 115 and r_armpit_angle > 50 and right_elbow_angle < 100:
            stage = 'Attack'
            if r_armpit_angle > 100:
                feedback = 'Lower Hitting Elbow!'
            elif r_armpit_angle < 80:
                feedback = 'Raise Hitting Elbow!'
            elif l_shoulder_attack_angle > 130:
                feedback = 'Move Left Arm In!'
            elif l_shoulder_attack_angle < 107: 
                feedback = 'Move left Arm Out!'
            else:
                feedback = 'Good!'
        else:
            stage = 'Standing'
            feedback = 'Pass, Set, Hit, or Block!'

        ### Creating the boxes on the video feed 
        
        # Blue Rectangle on Top Left
        cv2.rectangle(image, (0, 0), (275, 73), (245, 117, 16), -1)
        
        # White Rectangle on Top Right
        image_height, image_width, _ = image.shape
        cv2.rectangle(image, (image_width - 550, 0), (image_width, 73), (255, 255, 255), -1)
        
        # Changing Actions in the blue rectangle
        cv2.putText(image, 'ACTION', (15, 12),  # Adjusted x-coordinate for spacing
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (10, 60),  # Adjusted x-coordinate for spacing
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Feedback data in the white rectangle on top right of screen.
        cv2.putText(image, 'Feedback:', (image_width - 520, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.putText(image, feedback, (image_width - 530, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        #Since I only care about body landmarks, pose also comes built it with facial landmarks that I want to remove.
        if results.pose_landmarks:
            # Exclude facial landmarks by index
            non_facial_landmarks_indices = [
                idx for idx in range(len(results.pose_landmarks.landmark))
                if idx not in [
                    mp_pose.PoseLandmark.NOSE.value,
                    mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
                    mp_pose.PoseLandmark.LEFT_EYE.value,
                    mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
                    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
                    mp_pose.PoseLandmark.RIGHT_EYE.value,
                    mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
                    mp_pose.PoseLandmark.LEFT_EAR.value,
                    mp_pose.PoseLandmark.RIGHT_EAR.value,
                    mp_pose.PoseLandmark.MOUTH_LEFT.value,
                    mp_pose.PoseLandmark.MOUTH_RIGHT.value
                ]
            ]
            
            # Manual Drawing: Only Non-Facial Landmarks
        if results.pose_landmarks:
            non_facial_landmarks_indices = [
                idx for idx in range(len(results.pose_landmarks.landmark))
                if idx not in [
                    mp_pose.PoseLandmark.NOSE.value,
                    mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
                    mp_pose.PoseLandmark.LEFT_EYE.value,
                    mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
                    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
                    mp_pose.PoseLandmark.RIGHT_EYE.value,
                    mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
                    mp_pose.PoseLandmark.LEFT_EAR.value,
                    mp_pose.PoseLandmark.RIGHT_EAR.value,
                    mp_pose.PoseLandmark.MOUTH_LEFT.value,
                    mp_pose.PoseLandmark.MOUTH_RIGHT.value
                ]
            ]
            
            # Only draw non-facial landmarks
            for idx in non_facial_landmarks_indices:
                if idx < len(landmarks):
                    x = int(landmarks[idx].x * image.shape[1])
                    y = int(landmarks[idx].y * image.shape[0])
                    cv2.circle(image, (x, y), 4, (245, 117, 66), -1)

            # Draw connections between non-facial landmarks
            pose_connections = [
                conn for conn in mp_pose.POSE_CONNECTIONS 
                if conn[0] in non_facial_landmarks_indices and conn[1] in non_facial_landmarks_indices
            ]
            
            # Manually draw the connections
            for connection in pose_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = np.array([landmarks[start_idx].x * image.shape[1], landmarks[start_idx].y * image.shape[0]]).astype(int)
                    end = np.array([landmarks[end_idx].x * image.shape[1], landmarks[end_idx].y * image.shape[0]]).astype(int)
                    cv2.line(image, tuple(start), tuple(end), (245, 66, 230), 2)

        # Right hand using holistic
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand using holistic
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        
        
        #Visualize this
        #Mediapipe Feed is the name
        #frame is what we're seeing
        cv2.imshow('Mediapipe Feed', image)
    
        #What to do when we clear our feed aka by pressing q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
