# homography_utils.py
import cv2
import numpy as np

def compute_homography_auto(frame1, frame2):
    """
    Computes homography between two frames using ORB + RANSAC
    Args:
        frame1: image from broadcast
        frame2: image from tacticam
    Returns:
        homography matrix H (3x3)
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    if des1 is None or des2 is None:
        raise ValueError("ORB failed to find keypoints.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 4:
        raise ValueError("Not enough matches found to compute homography.")

    # Sort by distance and use top-N matches
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H
