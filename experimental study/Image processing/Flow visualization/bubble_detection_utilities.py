#!/usr/bin/env python3
"""
===================================================================================================================
Author: Montadhar Guesmi
Date: 04/02/2025
Institution: TU Dresden, Institut für Verfahrenstechnik und Umwelttechnik,
             Professur für Energieverfahrenstechnik (EVT)
E-Mail: montadhar.guesmi@tu-dresden.de

Summary:
  Reusable detection / classification utilities for gas-object (bubble) analysis
  in plate heat-exchanger images.
===================================================================================================================
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Retain LaTeX font settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

from shapely.geometry import Polygon, LineString
from shapely.prepared import prep
import pandas as pd
import time
import networkx as nx
import math
from scipy.interpolate import splprep, splev  # For segmented spline-fitting
from skimage.segmentation import active_contour
from skimage.filters import gaussian
# =============================================================================
# CONFIGURABLE PARAMETERS (General experiment and image processing settings)
# =============================================================================

# Frames per second of the image sequence (used to compute time step and velocities)
FPS = 2000

# Time interval between two consecutive frames [s]
DT = 1.0 / FPS

# Physical scale: conversion factor from pixels to meters [m/px]
# => 0.0752 mm per pixel as determined experimentally (e.g. using a reference object)
PIXEL_TO_METER = 0.0752 / 1000

# Minimum contour area to consider as a valid bubble [px²]
# Used to filter out noise and very small artifacts
MIN_AREA_PX = 6 * 6
#MIN_AREA_PX = 12 * 12 # For extra test 13_02_25-010_1035
# Maximum contour area to accept as a valid bubble [px²]
# Avoids detecting large background regions or merged objects as bubbles
MAX_AREA_PX = 1024 * 1024

# Number of pixels to mask out on the left side of the frame (ignored region)
ROI_LEFT_PX = 75

# Number of pixels to mask out on the right side of the frame (ignored region)
ROI_RIGHT_PX = 75

# Base intensity difference threshold for adaptive background subtraction
# Helps suppress fixed-pattern background noise
BASE_TOL = 5

# Relative threshold (as a fraction of background intensity) for adaptive subtraction
# Allows dynamic tolerance based on brightness variability in the image
REL_TOL = 0.07

# Complexity threshold based on visibility graph analysis
# Used to classify bubbles as "irregular" if their contour structure is complex
IRREGULAR_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# 1. IMAGE IO
# ---------------------------------------------------------------------------

def load_images(folder: str, background_name: str = "background.bmp", pattern: str = "Bild*.bmp"):
    """Return (background_img, list_of_frame_paths). Raises if missing."""
    bg_path = os.path.join(folder, background_name)
    bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    if bg is None:
        raise FileNotFoundError(bg_path)
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    files = [f for f in files if os.path.basename(f).lower()!=background_name.lower()]
    if len(files) < 2:
        raise RuntimeError("Need at least two frames for tracking.")
    return bg, files

# ---------------------------------------------------------------------------
# 2. PRE‑PROCESSING UTILITIES
# ---------------------------------------------------------------------------

def adaptive_background_subtraction(frame: np.ndarray, background: np.ndarray,
                                    base_tol: float = BASE_TOL,
                                    rel_tol: float  = REL_TOL) -> np.ndarray:
    diff = cv2.absdiff(frame, background).astype(np.float32)
    dyn  = base_tol + rel_tol * background.astype(np.float32)
    diff[diff < dyn] = 0
    return diff.clip(0,255).astype(np.uint8)


def apply_roi(img: np.ndarray, left: int = ROI_LEFT_PX, right: int = ROI_RIGHT_PX) -> np.ndarray:
    out = img.copy()
    h,w = out.shape
    out[:, :left] = 0
    out[:, w-right:] = 0
    return out


def apply_clahe(img: np.ndarray, clip: float = 1.5, grid: tuple[int,int]=(12,12)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(img)

# ---------------------------------------------------------------------------
# 3. RAW MASK DETECTION (two alternative pipelines)
# ---------------------------------------------------------------------------

def GaussianFilter_Based_Threshold_Edge_Detection(diff):
    '''
        > Uses a simple Gaussian blur
    '''
    thresh = 65
    gauss = 5
    canny_low = 0
    canny_high = 86
    morph = 5
    morph_iter = 5

    _, binary_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(diff, (gauss,gauss), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    combined = cv2.bitwise_or(binary_mask, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph,morph))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    return combined


def BilateralFilter_Based_Threshold_Edge_Detection(diff):
    d = 9
    sigmaC = 75
    sigmaS = 75
    thresh = 20
    canny_low = 0
    canny_high = 20
    morph = 5
    morph_iter = 5

    filtered = cv2.bilateralFilter(diff, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)
    filtered = cv2.medianBlur(filtered, 11)
    _, binary_mask = cv2.threshold(filtered, thresh, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(filtered, canny_low, canny_high)
    combined = cv2.bitwise_or(binary_mask, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph,morph))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    return closed


def detect_method_bubbles(frame_gray: np.ndarray, background: np.ndarray, method: str="method1", roi_left: int = ROI_LEFT_PX, roi_right: int = ROI_RIGHT_PX) -> np.ndarray:
    diff = adaptive_background_subtraction(frame_gray, background)
    diff = apply_roi(diff, left=roi_left, right=roi_right)
    diff = apply_clahe(diff, clip=2.0, grid=(8,8))
    if method=="method1":
        return GaussianFilter_Based_Threshold_Edge_Detection(diff)
    elif method=="method2":
        return BilateralFilter_Based_Threshold_Edge_Detection(diff)
    else:
        raise ValueError("Unknown method")

# ---------------------------------------------------------------------------
# 4. SHAPE METRIC HELPERS
# ---------------------------------------------------------------------------

def _approx_pts(contour):
    return [tuple(pt[0]) for pt in cv2.approxPolyDP(contour,2.0,True)]


def compute_aspect_ratio(contour):
    """
    Calculate the aspect ratio of a bubble contour, ensuring it's ≥ 1.
    
    Parameters:
    -----------
    contour : numpy.ndarray
        Contour points array from OpenCV
        
    Returns:
    --------
    float
        Aspect ratio as max(width,height)/min(width,height)
    """
    x, y, w, h = cv2.boundingRect(contour)
    return max(w, h) / float(min(w, h))


def compute_circularity(contour):
    a=cv2.contourArea(contour); p=cv2.arcLength(contour,True)
    return (4*np.pi*a/p**2) if p else 0

def _downsample(contour,step=1):
    pts=_approx_pts(contour)
    return pts if len(pts)<4 else pts[::step]

# --- visibility graph utilities ------------------------------------------------

def _vis_graph_edges(pts):
    n=len(pts);
    if n<4: return []
    if pts[0]!=pts[-1]: pts = pts+[pts[0]]
    poly_p = prep(Polygon(pts).buffer(0))
    edges=[]
    for i in range(n):
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1):
                continue
            if poly_p.contains(LineString([pts[i],pts[j]])):
                edges.append((i,j))
    return edges


def visibility_complexity(pts):
    e=_vis_graph_edges(pts); n=len(pts)
    return len(e)/(n*(n-1)/2) if n>1 else 0.0


def visibility_ratio(pts):
    n=len(pts)
    return len(_vis_graph_edges(pts))/(n*(n-3)/2) if n>=4 else 1.0

# ---------------------------------------------------------------------------
# 5. METRIC BUNDLE + CLASSIFICATION
# ---------------------------------------------------------------------------

def quantification_metrics(contour):
    area = cv2.contourArea(contour)
    if area==0:  # degenerate
        return None
    ds = _downsample(contour, step=2 if len(contour)>200 else 1)
    comp = visibility_complexity(ds)
    vis  = visibility_ratio(ds)
    ar   = compute_aspect_ratio(contour)
    circ = compute_circularity(contour)
    eqd  = math.sqrt(4*area/np.pi)*PIXEL_TO_METER*1000  # mm
    per  = cv2.arcLength(contour,True)
    return {
        "area":area, "perimeter":per, "complexity_score":comp,
        "visibility_ratio":vis, "aspect_ratio":ar,
        "circularity":circ, "equivalent_diameter":eqd
    }


def classify_shape(contour, irregular_threshold=0.3):
    """
    Classifies bubble shapes based on their geometric properties.
    
    Parameters:
    -----------
    contour : object
        Contour object containing the bubble outline
    irregular_threshold : float, optional
        Threshold for complexity score to determine regularity (default: 0.3)
    
    Returns:
    --------
    tuple
        (category_id, metrics)
        - category_id: integer from -1 to 5 representing bubble type
        - metrics: dictionary of calculated shape metrics or None if metrics calculation failed
    
    Categories:
    -----------
    -1: Artifact
     0: Round Small Bubble
     1: Round Coarse Bubble
     2: Elongated Bubble
     3: Big Gas Object (Regular)
     4: Big Gas Object (Irregular)
     5: Irregular Bubble
    """
    # Calculate metrics from contour
    metrics = quantification_metrics(contour)
    
    # Return artifact category if metrics calculation failed
    if metrics is None:
        return -1, None
    
    # Extract relevant metrics for classification
    complexity_score = metrics["complexity_score"]
    circularity = metrics["circularity"]
    equivalent_diameter = metrics["equivalent_diameter"]
    aspect_ratio = metrics["aspect_ratio"]
    
    # Define thresholds as constants for better readability
    BIG_DIAMETER = 8.0       # mm - threshold for big gas objects
    SMALL_DIAMETER = 1.0     # mm - threshold for small bubbles
    MEDIUM_DIAMETER = 4.0    # mm - upper threshold for medium bubbles
    CIRCULARITY_THRESHOLD = 0.7  # threshold for round shapes
    MIN_ASPECT_RATIO = 1.0   # minimum aspect ratio for elongated bubbles
    MAX_ASPECT_RATIO = 50.0  # maximum aspect ratio for elongated bubbles
    
    # 1. Big Gas Objects (diameter ≥ 8.0mm)
    if equivalent_diameter >= BIG_DIAMETER:
        # Check if regular or irregular based on complexity score
        if 0 < complexity_score < irregular_threshold:
            return 4, metrics  # Irregular big gas object
        else:
            return 3, metrics  # Regular big gas object
    
    # 2. Irregular Bubbles with low complexity score
    if 0 < complexity_score < irregular_threshold:
        return 5, metrics  # Irregular bubble
    
    # 3. Small and Medium Bubbles (diameter ≤ 4.0mm)
    if equivalent_diameter <= MEDIUM_DIAMETER:
        # Very small bubbles (diameter < 1.0mm)
        if equivalent_diameter < SMALL_DIAMETER:
            if circularity > CIRCULARITY_THRESHOLD:
                return 0, metrics  # Round small bubble
            else:
                return -1, metrics  # Artifact
        # Small to medium bubbles (1.0mm ≤ diameter ≤ 4.0mm)
        else:
            if circularity > CIRCULARITY_THRESHOLD:
                return 1, metrics  # Round coarse bubble
            else:
                return 5, metrics  # Irregular bubble
    
    # 4. Medium to Large Bubbles (4.0mm < diameter < 8.0mm)
    else:
        # Check for elongated shape based on aspect ratio
        if MIN_ASPECT_RATIO < aspect_ratio <= MAX_ASPECT_RATIO:
            return 2, metrics  # Elongated bubble
        else:
            return 5, metrics  # Irregular bubble


# ---------------------------------------------------------------------------
# 6. CONTOUR POST‑PROCESS + MASTER DETECT
# ---------------------------------------------------------------------------

def close_openings(mask: np.ndarray, contours):
    h,w = mask.shape; out=mask.copy()
    for c in contours:
        pts=c[:,0,:]
        on_border=[tuple(p) for p in pts if p[0]<=1 or p[0]>=w-2 or p[1]<=1 or p[1]>=h-2]
        if len(on_border)>=2:
            bp=np.array(on_border)
            d2=((bp[:,None,:]-bp[None,:,:])**2).sum(-1)
            i,j=np.unravel_index(np.argmax(d2),d2.shape)
            cv2.line(out, tuple(bp[i]), tuple(bp[j]),255,1)
    return out


def detect_and_classify_bubbles(frame_gray: np.ndarray, background: np.ndarray,
                                method: str="method1", roi_left: int = ROI_LEFT_PX, roi_right: int = ROI_RIGHT_PX):
    mask = detect_method_bubbles(frame_gray, background, method, roi_left=ROI_LEFT_PX, roi_right=ROI_RIGHT_PX)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask   = close_openings(mask,contours)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out={}
    idx=0
    for c in contours:
        a = cv2.contourArea(c)
        if MIN_AREA_PX<=a<=MAX_AREA_PX:
            idx+=1
            cls,metrics=classify_shape(c)
            out[idx] = {"contour":c,"id":idx,"classification":cls,"metrics":metrics}
    return out,mask

# ---------------------------------------------------------------------------
# 7. OPTIONAL COLOR / NAME MAPS (for plotting drivers)
# ---------------------------------------------------------------------------
CLUSTER_COLORS = {-1:"red",0:"darkblue",1:"green",2:"yellow",3:"cyan",4:"orange",5:"purple"}
CLUSTER_NAMES  = {-1:"Artifact",0:"Round Small Bubble",1:"Round Coarse Bubble",2:"Elongated Bubble",
                  3:"Big Gas Object (Regular)",4:"Big Gas Object (Irregular)",5:"Irregular Bubble"}

def map_cluster_to_color(cid):
    return CLUSTER_COLORS.get(cid,"red")

def map_shape_name(cid):
    return CLUSTER_NAMES.get(cid,"Artifact")


def update(frame_idx, background, all_files, ax):
    for a in ax:
        a.clear()
    fpath = all_files[frame_idx]
    frame_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    
    contours_method1, _ = detect_and_classify_bubbles(frame_gray, background, method="method1")
    contours_method2, _ = detect_and_classify_bubbles(frame_gray, background, method="method2")
    # Left Panel: Hybrid
    ax[0].imshow(frame_gray, cmap="gray")
    ax[0].set_title(f"Gaussian Filter + Threshold + Edge Detection (Frame {frame_idx})")
    for obj in contours_method1.values():
        contour, c_id = obj["contour"], obj["classification"]
        color = map_cluster_to_color(c_id)
        contour = contour.squeeze()
        if contour.ndim == 2 and contour.shape[0] > 1:
            ax[0].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)
            ax[0].text(contour[0, 0], contour[0, 1], str(obj["id"]), fontsize=8, color="white")
    
    # Right Panel: Snake
    ax[1].imshow(frame_gray, cmap="gray")
    ax[1].set_title(f"Bilateral Filter + Threshold + Edge Detection (Frame {frame_idx})")
    for obj in contours_method2.values():
        contour, c_id = obj["contour"], obj["classification"]
        color = map_cluster_to_color(c_id)
        contour = contour.squeeze()
        if contour.ndim == 2 and contour.shape[0] > 1:
            ax[1].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)
            ax[1].text(contour[0, 0], contour[0, 1], str(obj["id"]), fontsize=8, color="white")
    
    legend_patches = [
        plt.Line2D([0],[0], color=map_cluster_to_color(c), lw=2, label=map_shape_name(c))
        for c in range(-1, 6)
    ]
    ax[0].legend(handles=legend_patches, loc="upper right")
    ax[1].legend(handles=legend_patches, loc="upper right")
    ax[0].axis("off")
    ax[1].axis("off")
