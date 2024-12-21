import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def calculate_orientation(face_landmarks, frame):
    """Calculate head orientation angles from facial landmarks."""
    img_h, img_w = frame.shape[:2]
    
    face_3d = []
    face_2d = []
    
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])
    
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )
    
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
    return x, y, z

def process_video(video_file):
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Open the video file
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    frame_counter = st.empty()
    current_frame = 0
    
    # Create placeholders for video and orientation display
    video_placeholder = st.empty()
    orientation_text = st.empty()
    
    # Add play/pause button
    playing = st.button('Play/Pause')
    
    # Add slider for video navigation
    frame_slider = st.slider('Frame', 0, total_frames - 1, 0)
    
    # Set video to selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = current_frame / total_frames
        progress_bar.progress(progress)
        frame_counter.text(f'Frame: {current_frame}/{total_frames}')
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
                
                # Calculate and display orientation
                x, y, z = calculate_orientation(face_landmarks, frame)
                orientation_text.text(f"""
                Head Orientation:
                X (Pitch): {x:.2f}°
                Y (Yaw): {y:.2f}°
                Z (Roll): {z:.2f}°
                """)
                
                # Draw orientation arrows
                center = (frame_width//2, frame_height//2)
                length = 100
                
                # X-axis (Pitch) - Red
                cv2.line(frame, center, 
                        (center[0], center[1] - int(length * math.sin(math.radians(x)))),
                        (0, 0, 255), 3)
                # Y-axis (Yaw) - Green
                cv2.line(frame, center,
                        (center[0] + int(length * math.sin(math.radians(y))), center[1]),
                        (0, 255, 0), 3)
                # Z-axis (Roll) - Blue
                roll_x = center[0] + int(length * math.cos(math.radians(z)))
                roll_y = center[1] + int(length * math.sin(math.radians(z)))
                cv2.line(frame, center, (roll_x, roll_y), (255, 0, 0), 3)
        
        # Display frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        # Control video playback
        if not playing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
            
        # Add delay to match video FPS
        if playing:
            cv2.waitKey(int(1000/fps))
    
    # Clean up
    cap.release()
    os.unlink(tfile.name)

def main():
    st.title("Video Head Pose Tracking")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        process_video(uploaded_file)

if __name__ == "__main__":
    main()