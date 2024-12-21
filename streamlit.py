import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

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
    # Get image dimensions
    img_h, img_w = frame.shape[:2]
    
    # Face mesh coordinates
    face_3d = []
    face_2d = []
    
    # Key points for orientation calculation
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            # Get the 2D coordinates
            face_2d.append([x, y])
            # Get the 3D coordinates
            face_3d.append([x, y, lm.z])
    
    # Convert to numpy arrays
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # Camera matrix
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])
    
    # Distortion coefficients
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )
    
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # Calculate head orientation
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
    return x, y, z

def main():
    st.title("Real-time Head Pose Tracking")
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    # Create placeholders for orientation values
    orientation_text = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
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
                
                # Calculate orientation
                x, y, z = calculate_orientation(face_landmarks, frame)
                
                # Update orientation text
                orientation_text.text(f"""
                Head Orientation:
                X (Pitch): {x:.2f}°
                Y (Yaw): {y:.2f}°
                Z (Roll): {z:.2f}°
                """)
                
                # Draw arrows indicating orientation
                center = (frame.shape[1]//2, frame.shape[0]//2)
                
                # Draw coordinate axes
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
        
        # Convert frame back to RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update video feed
        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
    # Release resources
    cap.release()

if __name__ == "__main__":
    main()