import numpy as np
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio (EAR).
    
    The EAR is the ratio of the height of the eye to the width of the eye.
    A decreasing EAR indicates that the eye is closing.
    
    Args:
        eye: A list of 6 (x, y) coordinates representing the eye landmarks.
        
    Returns:
        The eye aspect ratio as a floating point number.
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    # Return the eye aspect ratio
    return ear

def get_eye_landmarks(landmarks, eye_indices):
    """
    Extract the eye landmarks from the facial landmarks.
    
    Args:
        landmarks: The facial landmarks from dlib's predictor.
        eye_indices: A list of indices corresponding to the eye landmarks.
        
    Returns:
        A numpy array of shape (6, 2) containing the (x, y) coordinates of the eye landmarks.
    """
    eye = []
    for i in eye_indices:
        point = (landmarks.part(i).x, landmarks.part(i).y)
        eye.append(point)
    return np.array(eye)

def draw_eye_landmarks(frame, eye_points, color=(0, 255, 0)):
    """
    Draw the eye landmarks on the frame.
    
    Args:
        frame: The frame to draw on.
        eye_points: The (x, y) coordinates of the eye landmarks.
        color: The color to draw the landmarks in (BGR format).
    """
    # Draw the contour of the eye
    for i in range(0, len(eye_points)):
        pt1 = (eye_points[i][0], eye_points[i][1])
        pt2 = (eye_points[(i + 1) % len(eye_points)][0], eye_points[(i + 1) % len(eye_points)][1])
        cv2.line(frame, pt1, pt2, color, 1)

import cv2
def draw_text(frame, text, position, font_scale=0.7, color=(0, 0, 255), thickness=2):
    """
    Draw text on the frame.
    
    Args:
        frame: The frame to draw on.
        text: The text to draw.
        position: The position (x, y) to draw the text at.
        font_scale: The scale of the font.
        color: The color of the text (BGR format).
        thickness: The thickness of the text.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness) 