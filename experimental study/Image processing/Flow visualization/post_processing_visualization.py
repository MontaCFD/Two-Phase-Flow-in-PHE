#!/usr/bin/env python3
"""
===================================================================================================================
Author: Montadhar Guesmi
Date: 23/04/2025
Institution: TU Dresden, Institut für Verfahrenstechnik und Umwelttechnik,
             Professur für Energieverfahrenstechnik (EVT)
E-Mail: montadhar.guesmi@tu-dresden.de

Summary:
  Visualization of detected gas objects with two filter based methods and classification into clusters after Extraction of contour features.
  (complexity score, equivalent diameter, circularity, aspect ratio).
===================================================================================================================
"""
from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from bubble_detection_utilities import * 

# ---------------------------------------------------------------------------
# CONFIGURABLE GLOBAL CONSTANTS 
# ---------------------------------------------------------------------------

MIN_AREA_PX       = 6 * 6            # smallest contour to keep [px²]
MAX_AREA_PX       = 896 * 896        # largest contour to keep [px²]
ROI_LEFT_PX       = 50
ROI_RIGHT_PX      = 50
BASE_TOL          = 5
REL_TOL           = 0.07             # adaptive BG subtraction
IRREGULAR_THRESHOLD = 0.3            # visibility‑graph complexity

# -------------------------------------------------------------------
# FOLDER / FILES
# -------------------------------------------------------------------
beta = 45    # chevron angle 

script_dir     = os.path.dirname(os.path.abspath(__file__))
data_dir      = r"C:/Users/Niepraschk/Desktop/Fuer Manuel/Diplomarbeit/Austausch/von Manuel/ZET/"
chevron_dir   = "aufnahmen"+ str(beta)+"grad"  
images_dir    = os.path.join(data_dir, chevron_dir)
test_case     = "25_03_25-010_970"
images_dir    = os.path.join(images_dir, test_case)

file_list = sorted([f for f in os.listdir(images_dir) if f.endswith(".bmp")])
background, all_files = load_images(images_dir)

#-------------------------------------------------------------------
# Frames per second, time step, conversion pixel to meter 
#-------------------------------------------------------------------
FPS               = len(all_files)
DT                = 1.0 / FPS
PIXEL_TO_METER    = 0.0752 / 1000    # [m / px]

# -------------------------------------------------------------------
# SAVE SINGLE FRAME AS SVG (BEFORE ANIMATION)
# -------------------------------------------------------------------
FRAME_INDEX_TO_SAVE = 480  # Define which frame to save

# Create a separate figure for saving
fig_save, ax_save = plt.subplots(1, 2, figsize=(12, 7))

# Generate the specific frame
fpath = all_files[FRAME_INDEX_TO_SAVE]
frame_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

contours_method1, _ = detect_and_classify_bubbles(frame_gray, background, method="method1", roi_left=ROI_LEFT_PX, roi_right=ROI_RIGHT_PX)
contours_method2, _ = detect_and_classify_bubbles(frame_gray, background, method="method2", roi_left=ROI_LEFT_PX, roi_right=ROI_RIGHT_PX)

# Left Panel: Method 1
ax_save[0].imshow(frame_gray, cmap="gray")
ax_save[0].set_title(f"Gaussian Filter + Threshold + Edge Detection (Frame {FRAME_INDEX_TO_SAVE})")
for obj in contours_method1.values():
    cnt, c_id = obj["contour"], obj["classification"]
    color = map_cluster_to_color(c_id)
    cnt = cnt.squeeze()
    if cnt.ndim == 2 and cnt.shape[0] > 1:
        ax_save[0].plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=1.5)
        ax_save[0].text(cnt[0, 0], cnt[0, 1], str(obj["id"]), fontsize=8, color="white")

# Right Panel: Method 2
ax_save[1].imshow(frame_gray, cmap="gray")
ax_save[1].set_title(f"Bilateral Filter + Threshold + Edge Detection (Frame {FRAME_INDEX_TO_SAVE})")
for obj in contours_method2.values():
    cnt, c_id = obj["contour"], obj["classification"]
    color = map_cluster_to_color(c_id)
    cnt = cnt.squeeze()
    if cnt.ndim == 2 and cnt.shape[0] > 1:
        ax_save[1].plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=1.5)
        ax_save[1].text(cnt[0, 0], cnt[0, 1], str(obj["id"]), fontsize=8, color="white")

# Add legends
legend_patches = [
    plt.Line2D([0],[0], color=map_cluster_to_color(c), lw=2, label=map_shape_name(c))
    for c in range(-1, 6)
]
ax_save[0].legend(handles=legend_patches, loc="upper right")
ax_save[1].legend(handles=legend_patches, loc="upper right")
ax_save[0].axis("off")
ax_save[1].axis("off")

# Save as SVG
plt.tight_layout()
output_filename = f"bubble_detection_frame_{FRAME_INDEX_TO_SAVE:04d}.svg"
fig_save.savefig(output_filename, format='svg', bbox_inches='tight', dpi=300)
print(f"Frame {FRAME_INDEX_TO_SAVE} saved as {output_filename}")

# Close the save figure
plt.close(fig_save)

# -------------------------------------------------------------------
# MATPLOTLIB FIGURE FOR ANIMATION
# -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 7))

# -------------------------------------------------------------------
# LEGEND HANDLER CLASSES
# -------------------------------------------------------------------
class HandlerEllipse(matplotlib.legend_handler.HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = matplotlib.patches.Ellipse(xy=center, width=width*0.5, height=width*0.5)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# -------------------------------------------------------------------
# UPDATE FUNCTION FOR ANIMATION
# -------------------------------------------------------------------
def update(frame_idx):
    for a in ax:
        a.clear()
    fpath = all_files[frame_idx]
    frame_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    
    contours_method1, _ = detect_and_classify_bubbles(frame_gray, background, method="method1", roi_left=ROI_LEFT_PX, roi_right=ROI_RIGHT_PX)
    contours_method2, _ = detect_and_classify_bubbles(frame_gray, background, method="method2", roi_left=ROI_LEFT_PX, roi_right=ROI_RIGHT_PX)
    
    # Left Panel: Gaussian
    ax[0].imshow(frame_gray, cmap="gray")
    ax[0].set_title(f"Gaussian Filter + Threshold + Edge Detection (Frame {frame_idx})")
    for obj in contours_method1.values():
        cnt, c_id = obj["contour"], obj["classification"]
        color = map_cluster_to_color(c_id)
        cnt = cnt.squeeze()
        if cnt.ndim == 2 and cnt.shape[0] > 1:
            ax[0].plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=1.5)
            ax[0].text(cnt[0, 0], cnt[0, 1], str(obj["id"]), fontsize=8, color="white")
    
    # Right Panel: Bilateral
    ax[1].imshow(frame_gray, cmap="gray")
    ax[1].set_title(f"Bilateral Filter + Threshold + Edge Detection (Frame {frame_idx})")
    for obj in contours_method2.values():
        cnt, c_id = obj["contour"], obj["classification"]
        color = map_cluster_to_color(c_id)
        cnt = cnt.squeeze()
        if cnt.ndim == 2 and cnt.shape[0] > 1:
            ax[1].plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=1.5)
            ax[1].text(cnt[0, 0], cnt[0, 1], str(obj["id"]), fontsize=8, color="white")
    
    legend_patches = [
        plt.Line2D([0],[0], color=map_cluster_to_color(c), lw=2, label=map_shape_name(c))
        for c in range(-1, 6)
    ]
    ax[0].legend(handles=legend_patches, loc="upper right")
    ax[1].legend(handles=legend_patches, loc="upper right")
    ax[0].axis("off")
    ax[1].axis("off")

# -------------------------------------------------------------------
# ALTERNATIVE VIS_UPDATE FUNCTION
# -------------------------------------------------------------------        
def vis_update(frame_idx):
    for a in ax: 
        a.clear()
        a.axis("off")

    fpath = all_files[frame_idx]
    frame = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    diff  = adaptive_background_subtraction(frame, background, base_tol=BASE_TOL, rel_tol=REL_TOL)
    diff  = apply_roi(diff)

    # --- Method 1 -----------------------------------------------------
    objs1,_ = detect_and_classify_bubbles(frame, background, method="method1")
    ax[0].imshow(frame, cmap="gray")
    for obj in objs1.values():
        cnt = obj["contour"].squeeze()
        cid = obj["classification"]
        col = map_cluster_to_color(cid)
        if cnt.ndim==2 and len(cnt)>1:
            ax[0].plot(cnt[:,0],cnt[:,1], color=col, lw=1.3)
            ax[0].text(cnt[0,0],cnt[0,1], str(obj["id"]), fontsize=7, color="white")

    # --- Method 2 -----------------------------------------------------
    objs2,_ = detect_and_classify_bubbles(frame, background, method="method2")
    ax[1].imshow(frame, cmap="gray")
    for obj in objs2.values():
        cnt = obj["contour"].squeeze()
        cid = obj["classification"]
        col = map_cluster_to_color(cid)
        if cnt.ndim==2 and len(cnt)>1:
            ax[1].plot(cnt[:,0],cnt[:,1], color=col, lw=1.3)
            ax[1].text(cnt[0,0],cnt[0,1], str(obj["id"]), fontsize=7, color="white")

    # --- Legend (once) -----------------------------------------------
    if frame_idx == 0:
        patches = [plt.Line2D([0],[0], color=map_cluster_to_color(c), lw=2,
                               label=map_shape_name(c)) for c in range(-1,6)]
        ax[1].legend(handles=patches, loc="lower right", fontsize=8)

# -------------------------------------------------------------------
# RUN ANIMATION
# -------------------------------------------------------------------
ani = animation.FuncAnimation(fig, update, frames=range(0,len(all_files), ROI_LEFT_PX//15), repeat=False)
plt.tight_layout()
plt.show()

