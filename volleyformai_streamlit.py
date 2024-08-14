import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

st.set_page_config(layout="wide")

# CSS to center the title, underline it, and make the font bigger
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 3em; /* Increase the font size */
        font-weight: bold;
        text-decoration: underline;
    }
    .subheader {
        font-size: 1.5em;
        font-weight: bold;
    }
    .small-text {
        font-size: 0.9em;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .center-columns {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        margin: 0 auto;
        width: 100%;
    }
    .highlight-checkbox {
        font-weight: bold;
        font-size: 1.2em;
        color: #FF5733; /* Orange color */
    }
    .video-frame {
        border: 5px solid black;
        width: 800px;
        height: 600px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.markdown('<div class="centered-title">VolleyForm AI</div>', unsafe_allow_html=True)
st.write('### Background')
st.write("""
    Inspired by the Olympics, I sought to merge my Data Science and Volleyball backgrounds to create an app that showcases the 'correct' form for the four main volleyball actions: Pass, Set, Attack, and Block.
""")

# Display YouTube Video for Demonstration
st.write("### See Volleyform AI in Action")
st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <iframe width="800" height="450" src="https://www.youtube.com/embed/5HBTQOSB7sI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

# Display Images for Volleyball Actions with Tips
st.markdown('<h3>Volleyball Actions</h3>', unsafe_allow_html=True)
st.markdown('<div class="center-content">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**Passing**")
    passing_img = Image.open("images/passing.png")
    st.image(passing_img)
    st.markdown("""
        Keep your hands below your shoulders. Bring your hands together without bending your arms. <b>Try raising or lowering your platform or bending your elbows to see the difference. </b>
    """, unsafe_allow_html=True)

with col2:
    st.write("**Setting**")
    setting_img = Image.open("images/setting.png")
    st.image(setting_img)
    st.markdown("""
        Keep your arms above your head with elbows shoulder-width apart. Open your fingers wide. <b> Experiment with closing your fingers or moving your elbows to change the set.</b>
    """, unsafe_allow_html=True)

with col3:
    st.write("**Attacking**")
    hitting_img = Image.open("images/hitting.png")
    st.image(hitting_img)
    st.markdown("""
       Raise your left arm up and pull your right elbow back like drawing a bow. Keep your right hand open. <b> Adjust by moving your left arm in/out (keeping it straight) or dropping your right shoulder. </b>
    """, unsafe_allow_html=True)

with col4:
    st.write("**Blocking**")
    hitting_img = Image.open("images/blocking.jpg")
    st.image(hitting_img)
    st.markdown("""
        Extend your arms straight above your head, shoulder-width apart with thumbs pointing up. <b> Practice opening and closing your hands to improve your block.</b>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Video Feed Section
st.write("### Let's See Your Form!")
st.markdown("**<u>You'll need to allow webcam access. Make sure the camera can see your hands when you raise your arms all the way up, and the bottom should cut off at your waist.</u>**", unsafe_allow_html=True)


# Create a two-column layout for the checkboxes and align them closer to the header
col1, col2 = st.columns([1, 1])  # Adjust column widths for checkboxes

with col1:
    run = st.checkbox("Click here to check out your form!", key="start_video",help="This will start the video feed to analyze your form.")


with col2:
    show_metrics = st.checkbox("Show Data", key="show_metrics")

# Create placeholders for the metrics
metrics_placeholder = st.empty()

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle 

# Video Feed
FRAME_WINDOW = st.image([])
stage = None
feedback = None
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

if run:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                r_armpit_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
                l_armpit_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
                l_shoulder_attack_angle = calculate_angle(l_elbow, l_shoulder, r_shoulder)
                left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                wrist_distance = math.sqrt((r_wrist[0] - l_wrist[0]) ** 2 + (r_wrist[1] - l_wrist[1]) ** 2)
                elbow_distance = math.sqrt((r_elbow[0] - l_elbow[0]) ** 2 + (r_elbow[1] - l_elbow[1]) ** 2)
                
                #Less than 100, then down, else up.
                if r_armpit_angle >= 120 and wrist_distance < 0.20:
                    stage = 'Setting'
                    if elbow_distance < 0.25:
                        feedback = 'Elbows Further Apart!'
                    elif elbow_distance > 0.34:
                        feedback = 'Elbows Closer Together!'
                    elif l_thumb_pinky_distance < 0.06 or r_thumb_pinky_distance < 0.06:
                        feedback = 'Open Your Hands!'
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
                    feedback = ''

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
                
                # Draw only body landmarks excluding facial landmarks
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

                # Center the image and apply a black outline
                resized_image = cv2.resize(image, (800, 600))
                bordered_image = cv2.copyMakeBorder(resized_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                # Centering and outlining the video feed
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                FRAME_WINDOW.image(image, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if show_metrics:
                    metrics_placeholder.markdown(f"""
                        **Right Shoulder Angle:** {r_armpit_angle:.2f}  
                        **Left Armpit Angle:** {l_armpit_angle:.2f}  
                        **Left Shoulder Angle:** {l_shoulder_attack_angle:.2f}  
                        **Left Elbow Angle:** {left_elbow_angle:.2f}  
                        **Distance between wrists:** {wrist_distance:.2f}  
                        **Distance between elbows:** {elbow_distance:.2f}   
                    """)

            except Exception as e:
                st.write(f"Error: {e}")

else:
    cap.release()
