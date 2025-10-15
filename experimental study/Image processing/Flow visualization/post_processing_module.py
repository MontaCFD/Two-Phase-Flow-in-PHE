#! /usr/bin/env python3
"""
===============================================================================
Author: Montadhar Guesmi 
Date: 29/04/2025
Institution: TU Dresden, Institut für Verfahrenstechnik und Umwelttechnik, 
             Professur für Energieverfahrenstechnik (EVT)
E-Mail: montadhar.guesmi@tu-dresden.de
Summary:
  This script analyzes experimental data for tracking gas objects (bubbles) in a 
  plate heat exchanger (PHE) channel under oxygen-water two-phase flow. It extracts 
  individual gas trajectories, estimates their velocity, and examines the relationship 
  between gas size, speed, and shape classification.
Tasks:
> adaptive Bins 
> Size distribution by cluster: adding total percentage of area by each cluster
> Adding average velocity (arithmetic, area-weighted and volume-weighted) as vetical line,
  as well as the liquid velocity
> TODO II: follow the idea to determine the bubbles velocity. Consider only bubble cluster and set a maximum threshold for velocity
===============================================================================
"""

from __future__ import annotations
from calendar import c
from email.mime import image
import os, sys
from turtle import color
from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib
from pyvista import check_valid_vector
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import math
from collections import defaultdict

from bubble_detection_utilities import * 
import time 

# --------------------------------------------------------------------
# CONFIGURATION 
# --------------------------------------------------------------------
APPLY_VELOCITY_FILTER      = False  # Set to True to apply velocity threshold and consider only bubble clusters
PIXEL_TO_METER             = 0.0752 / 1000    # [m / px]
MAX_VELOCITY_THRESHOLD     = 6.0  # m/s - adjust based on your specific application needs
MAX_VELOCITY_THRESHOLD_PIX = MAX_VELOCITY_THRESHOLD/PIXEL_TO_METER
BUBBLE_CLUSTER_IDS = [0, 1, 2, 5]  # IDs that represent bubble clusters (as specified)

# ------------------------------------------------------------------
# Ensure Directory exists
# ------------------------------------------------------------------
def ensure_dir_exists(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# -------------------------------------------------------------------
# BUILD TRAJECTORIES
# -------------------------------------------------------------------
def build_trajectories(files, background, method="method1", match_radius=50, ARTIFACT_CLUSTER_ID=-1):
    prev = []
    next_id = 0
    traj = {}
    r2 = match_radius**2

    for f_idx, path in enumerate(files):
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        objs, _ = detect_and_classify_bubbles(frame, background, method=method)
        curr = []
        for o in objs.values():
            if o["classification"] == ARTIFACT_CLUSTER_ID:
                continue
            M = cv2.moments(o["contour"])
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            diam = o["metrics"]["equivalent_diameter"]
            area = cv2.contourArea(o["contour"])
            cl = o["classification"]
            curr.append((cx, cy, diam, cl, area))

        new_prev = []
        for cx, cy, diam, cl, area in curr:
            bid = None; bd = r2
            for pid, px, py in prev:
                d2 = (cx-px)**2 + (cy-py)**2
                if d2 < bd:
                    bd, bid = d2, pid
            if bid is None:
                next_id += 1; bid = next_id; traj[bid] = []
            # velocity in px/s
            if traj[bid]:
                _, px, py, *_ = traj[bid][-1]
                vel = math.hypot(cx-px, cy-py)/DT
            else:
                vel = 0.0
            traj[bid].append((f_idx, cx, cy, vel, diam, cl, area))
            new_prev.append((bid, cx, cy))
        prev = new_prev
    return traj

def summarize_trajectories(traj):
    ARTIFACT_CLUSTER_ID = -1
    rows = []
    cluster_stats = defaultdict(lambda: {'total_vxa': 0.0, 'total_area': 0.0, 'count': 0})
    
    for bid, rec in traj.items():
        if not rec or rec[-1][5] == ARTIFACT_CLUSTER_ID:
            continue
            
        start, end = rec[0][0], rec[-1][0]
        #Idea: set a max. threshold for velocity to be summed up 
        # Then in the summarize_trajectories function, modify the speeds calculation:
        APPLY_VELOCITY_FILTER_here = False
        if APPLY_VELOCITY_FILTER_here:
            speeds = [r[3] for r in rec if 0 < r[3] <= MAX_VELOCITY_THRESHOLD_PIX]
        else:
            speeds = [r[3] for r in rec if r[3] > 0]
        
        avg_v = (sum(speeds)/len(speeds))*PIXEL_TO_METER if speeds else 0.0
    
        # Get final metrics for area-weighted calculations
        final_diam = rec[-1][4]
        final_area_px = rec[-1][6]  
        cluster_id = rec[-1][5]
        cluster_name = map_shape_name(cluster_id)
        
        # Update cluster statistics 
        #Idea: take into account only Bubbles (Big gas objects can be neglected).
        # Implement the second Idea: consider only bubble clusters
        if APPLY_VELOCITY_FILTER_here:
            if cluster_id in BUBBLE_CLUSTER_IDS:
                cluster_stats[cluster_id]['total_vxa'] += avg_v * final_area_px
                cluster_stats[cluster_id]['total_area'] += final_area_px
                cluster_stats[cluster_id]['count'] += 1
        else:
            # Original behavior without filtering
            cluster_stats[cluster_id]['total_vxa'] += avg_v * final_area_px
            cluster_stats[cluster_id]['total_area'] += final_area_px
            cluster_stats[cluster_id]['count'] += 1
             
        
        rows.append({
            "bubble_id": bid,
            "start_frame": start,
            "end_frame": end,
            "track_length": len(rec),
            "avg_velocity_m_s": avg_v,
            "final_eq_diameter_mm": final_diam,
            "area_px": final_area_px,
            "cluster": cluster_name,
            "cluster_id": cluster_id
        })
    
    # Add cluster summaries with area-weighted velocities
    for cluster_id, stats in cluster_stats.items():
        if stats['total_area'] > 0:
            weighted_v = stats['total_vxa'] / stats['total_area']
            rows.append({
                "bubble_id": f"CLUSTER_{cluster_id}_SUMMARY",
                "cluster": map_shape_name(cluster_id),
                "cluster_id": cluster_id,
                "area_weighted_velocity_m_s": weighted_v,
                "count": stats['count']
            })
    
    return pd.DataFrame(rows)

def compute_gas_fractions(files, background, method="method1"):
    im0 = cv2.imread(files[0], 0)
    h, w = im0.shape; L, R = w//3, 2*w//3
    fracs = {"all": [], "L": [], "M": [], "R": []}
    
    for p in files:
        im = cv2.imread(p, 0)
        _, mask = detect_and_classify_bubbles(im, background, method=method)
        g = (mask > 0)
        fracs["all"].append(g.mean())
        fracs["L"].append(g[:, :L].mean())   # Left side
        fracs["M"].append(g[:, L:R].mean())  # Middle side
        fracs["R"].append(g[:, R:].mean())   # Right side
        
    return {k: np.mean(v) for k, v in fracs.items()}

def update(frame_idx, all_files, ax, background):
    for a in ax:
        a.clear()
        a.axis("off")
    
    fpath = all_files[frame_idx]
    frame = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    
    contours_method1, _ = detect_and_classify_bubbles(frame, background, method="method1")
    contours_method2, _ = detect_and_classify_bubbles(frame, background, method="method2")
    
    ARTIFACT_CLUSTER_ID = -1
    
    # Left Panel: Method 1
    ax[0].imshow(frame, cmap="gray")
    ax[0].set_title(f"Gaussian Filter (Frame {frame_idx})")
    for obj in contours_method1.values():
        if obj["classification"] == ARTIFACT_CLUSTER_ID:
            continue
        cnt, c_id = obj["contour"], obj["classification"]
        color = map_cluster_to_color(c_id)
        cnt = cnt.squeeze()
        if cnt.ndim == 2 and cnt.shape[0] > 1:
            ax[0].plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=1.5)
            ax[0].text(cnt[0, 0], cnt[0, 1], str(obj["id"]), fontsize=8, color="white")
    
    # Right Panel: Method 2
    ax[1].imshow(frame, cmap="gray")
    ax[1].set_title(f"Bilateral Filter (Frame {frame_idx})")
    for obj in contours_method2.values():
        if obj["classification"] == ARTIFACT_CLUSTER_ID:
            continue
        cnt, c_id = obj["contour"], obj["classification"]
        color = map_cluster_to_color(c_id)
        cnt = cnt.squeeze()
        if cnt.ndim == 2 and cnt.shape[0] > 1:
            ax[1].plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=1.5)
            ax[1].text(cnt[0, 0], cnt[0, 1], str(obj["id"]), fontsize=8, color="white")
    
    # Legend excluding artifacts
    legend_patches = [
        plt.Line2D([0],[0], color=map_cluster_to_color(c), lw=2, label=map_shape_name(c))
        for c in range(0, 6)
    ]
    ax[0].legend(handles=legend_patches, loc="upper right")
    ax[1].legend(handles=legend_patches, loc="upper right")



def export_to_csv(df, gas_frac, output_dir):
    """
    Export a DataFrame to a CSV file with gas fraction information appended.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing bubble data
    gas_frac : dict
        Dictionary with region names as keys and gas fraction values
    output_dir : str
        Directory where the CSV should be saved
    """
    # Create the filename based on the last folder in the path
    folder_name = os.path.basename(output_dir.rstrip(os.sep))
    file_path = os.path.join(output_dir, f"{folder_name}.csv")
    
    # Add gas fraction information to the DataFrame
    for region, val in gas_frac.items():
        df = pd.concat([df, pd.DataFrame({
            "bubble_id": [f"_frac_{region}"],
            "gas_fraction": [val]
        })], ignore_index=True)
    
    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)
    print(f"→ CSV saved as {file_path}")

def plot_statistics(df):
    
    # Filter out artifacts and summaries
    ARTIFACT_CLUSTER_ID = -1
    valid_df = df[(df.cluster_id != ARTIFACT_CLUSTER_ID) & 
                 (~df.bubble_id.astype(str).str.contains("CLUSTER_")) & 
                 (df.track_length > 1)]
    
    # Get cluster-level area-weighted velocities
    cluster_df = df[df.bubble_id.astype(str).str.contains("CLUSTER_")]
    
    # Calculate Overall Area-Weighted Velocity as arithmetic mean of cluster velocities ***
    overall_area_weighted_velocity = 0.0
    if not cluster_df.empty and 'area_weighted_velocity_m_s' in cluster_df.columns:
        # Extract area-weighted velocities from cluster summaries
        cluster_velocities = cluster_df['area_weighted_velocity_m_s'].values
        # Calculate arithmetic mean
        overall_area_weighted_velocity = np.mean(cluster_velocities)
        print(f"Overall area-weighted velocity (arithmetic mean of clusters): {overall_area_weighted_velocity:.4f} m/s")
    
    # Create figure with subplots
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    
    # Velocity histogram with adaptive bins 
    min_vel = max(0.0, valid_df.avg_velocity_m_s.min())
    max_vel = valid_df.avg_velocity_m_s.max()
    max_vel = MAX_VELOCITY_THRESHOLD   # set a max value for the plot 
    # Generate adaptive bin edges for size histogram
    vel_bins = []
    # Fine bins for velocities under 2m/s
    vel_bins.extend(np.linspace(min_vel, min(2.0, max_vel), 10))
    # Coarser bins for larger velocity
    if max_vel > 2.0:
        vel_bins.extend(np.linspace(2, max_vel, 5)[1:])  # [1:] to avoid duplicate 2m/s bin
    vel_bins = np.array(vel_bins)  # Convert to numpy array
    
    ax[0, 0].hist(valid_df.avg_velocity_m_s, bins=15, color="navy", alpha=0.7)
    ax[0, 0].set_title(r"Velocity Distribution", fontsize=14)
    ax[0, 0].set_xlabel(r"Velocity (m/s)"); ax[0, 0].set_ylabel("Frequency")
    ax[0,0].set_xlim(0, max_vel)
    
    # Size histogram with adaptive bins
    min_size = max(0.01, valid_df.final_eq_diameter_mm.min())
    max_size = valid_df.final_eq_diameter_mm.max()
    
    # Generate adaptive bin edges for size histogram
    size_bins = []
    # Fine bins for diameters under 8mm
    size_bins.extend(np.linspace(min_size, min(8, max_size), 15))
    # Coarser bins for larger diameters if they exist
    if max_size > 8:
        size_bins.extend(np.linspace(8, max_size, 10)[1:])  # [1:] to avoid duplicate 8mm bin
    size_bins = np.array(size_bins)  # Convert to numpy array
    
    ax[0, 1].hist(valid_df.final_eq_diameter_mm, bins=size_bins, color="darkgreen", alpha=0.7)
    ax[0, 1].set_title(r"Size Distribution (Adaptive Bins)", fontsize=14)
    ax[0, 1].set_xlabel(r"Diameter (mm)"); ax[0, 1].set_ylabel(r"Frequency")
    
    # Size vs. Velocity scatter
    for cl in valid_df.cluster.unique():
        sub = valid_df[valid_df.cluster == cl]
        if sub.empty: continue 
        c_id = sub['cluster_id'].iloc[0]
        color = map_cluster_to_color(c_id)
        ax[1, 0].scatter(sub.final_eq_diameter_mm, sub.avg_velocity_m_s,
                       alpha=0.7,color=color, label=cl, s=30)
    
    # Plot Overall Area-Weighted Velocity Line ---
    xmin, xmax = ax[1, 0].get_xlim()
    if overall_area_weighted_velocity > 0:
        ax[1, 0].hlines(overall_area_weighted_velocity, xmin=xmin, xmax=xmax,
                      color='black', linestyle='-', linewidth=2.5, # Use a distinct style
                      label=f"Overall Area-weighted Avg. Velocity ({overall_area_weighted_velocity:.3f} m/s)")
        
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_title(r"Size vs. Velocity by Type", fontsize=14)
    ax[1, 0].set_xlabel(r"Diameter (mm)"); ax[1, 0].set_ylabel(r"Velocity (m/s)")
    ax[1, 0].legend(fontsize=10)
    
    # Area-weighted velocities
    if not cluster_df.empty:
        clusters = []; velocities = []; counts = []; colors = []; areas = []
        # Calculate total area of all bubbles from valid_df
        total_gas_area = valid_df['area_px'].sum() if not valid_df.empty else 1
    
        # Group valid_df by cluster_id to calculate area per cluster
        cluster_areas = valid_df.groupby('cluster_id')['area_px'].sum().to_dict()
        
        for _, row in cluster_df.iterrows():
            c_id = row['cluster_id']
            clusters.append(row['cluster'])
            velocities.append(row['area_weighted_velocity_m_s'])
            counts.append(row.get('count', 0))
            colors.append(map_cluster_to_color(c_id))
            # Get area for this cluster (default to 0 if not found)
            area = cluster_areas.get(c_id, 0)
            # Calculate percentage of total area
            area_percent = (area / total_gas_area) * 100 if total_gas_area > 0 else 0
            areas.append(area_percent)
        # Sort by velocity for better visualization
        sorted_idx = np.argsort(velocities)
        sorted_clusters = [clusters[i] for i in sorted_idx]
        sorted_velocities = [velocities[i] for i in sorted_idx]
        sorted_counts = [counts[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]
        sorted_areas = [areas[i] for i in sorted_idx]
        bars = ax[1, 1].barh(sorted_clusters, sorted_velocities, color=sorted_colors, alpha=0.7)
        
        for i, (bar, count, area_pct) in enumerate(zip(bars, sorted_counts, sorted_areas)):
            ax[1, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        rf"{area_pct:.1f}\%", va='center', fontsize=12)
            
        ax[1, 1].set_title(r"Mean Velocity by Cluster", fontsize=14)
        ax[1, 1].set_xlabel(r"Area-Weighted Velocity (m/s)", fontsize=12)
        ax[1, 1].set_ylabel(r"Gas Cluster", fontsize=12)
    
    plt.tight_layout()
    #plt.savefig(os.path.basename(results_case_dir) + "_stats.png", dpi=300)

def plot_size_distribution_cumulative(df):
    ARTIFACT_CLUSTER_ID = -1
    valid_df = df[(df.cluster_id != ARTIFACT_CLUSTER_ID) & 
                 (~df.bubble_id.astype(str).str.contains("CLUSTER_"))]
    
    # Create adaptive bins - finer under 8mm, coarser above 
    min_size = max(0.01, valid_df.final_eq_diameter_mm.min())
    max_size = valid_df.final_eq_diameter_mm.max()
    
    # Generate adaptive bin edges
    bins = []
    # Fine bins for diameters under 8mm
    bins.extend(np.linspace(min_size, min(8, max_size), 8))
    # Coarser bins for larger diameters if they exist
    if max_size > 8:
        bins.extend(np.linspace(8, max_size, 4)[1:])  # [1:] to avoid duplicate 8mm bin
    bins = np.array(bins)  # Convert to numpy array
    
    # Calculate bin centers (geometric mean)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    # Calculate distributions by cluster
    cluster_groups = valid_df.groupby('cluster')
    hist_data = {}; total_counts = np.zeros_like(bins[:-1])
    
    for name, group in cluster_groups:
        hist, _ = np.histogram(group.final_eq_diameter_mm, bins=bins)
        hist_data[name] = hist
        total_counts += hist
    
    # Convert to percentage and calculate cumulative
    total_counts = np.maximum(total_counts, 1)
    percentages = {k: (v/total_counts)*100 for k, v in hist_data.items()}
    cumulative = np.cumsum(total_counts)/total_counts.sum()*100
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bottom = np.zeros_like(bins[:-1])
    
    for name, perc in percentages.items():
        # Get the corresponding group from the groupby object
        group = cluster_groups.get_group(name)
        # Extract the cluster_id (it's the same for all rows in the group)
        c_id = group['cluster_id'].iloc[0]
        # Get the color using the cluster_id
        color = map_cluster_to_color(c_id)
        # Get the count for the label
        count = group.shape[0]

        # Plot the bar using the cluster_id based color
        plt.bar(bin_centers, perc, width=np.diff(bins), bottom=bottom, color=color,
            label=f"{name}")
        bottom += perc
    
    ax1 = plt.gca(); ax2 = ax1.twinx()
    ax2.plot(bin_centers, cumulative, 'k--', lw=2, label='Cumulative %')
    
    ax1.set_xscale('log')
    ax1.set_xlabel(r"Bubble Size (mm)", fontsize=12)
    ax1.set_ylabel(r"Size Distribution (\%)", fontsize=12)
    ax2.set_ylabel(r"Cumulative Percentage (\%)", fontsize=12)
    
    ax1.legend(loc='upper left', title="Bubble Type", fontsize=10)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.title(r"Bubble Size Distribution by Cluster Type", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig(os.path.basename(results_case_dir) + "_size_distribution.png", dpi=300)

def plot_spatial_distribution(gas_fractions):
    plt.figure(figsize=(10, 6))
    regions = ['L', 'M', 'R']
    values = [gas_fractions[r] * 100 for r in regions]
    
    # Calculate the percentage of total gas in each region
    total_gas = sum(gas_fractions[r] for r in regions)
    percentage_of_total = [(gas_fractions[r] / total_gas) * 100 for r in regions]
    
    bars = plt.bar(regions, percentage_of_total, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    
    #plt.axhline(gas_fractions['all'] * 100, color='black', linestyle='--', label=fr'Overall: {gas_fractions["all"]*100:.2f}\%')
    
    plt.ylabel(r'Gas Distribution (\% of total gas)', fontsize=12)
    plt.title(r'Spatial Distribution of Gas Phase', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add percentage labels in the middle of each bar
    for i, (bar, pct) in enumerate(zip(bars, percentage_of_total)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height/2, 
                rf'{pct:.1f}\%', ha='center', va='center', 
                color='black', fontweight='bold', fontsize=12)
    
    plt.ylim(0, max(percentage_of_total) * 1.15)  # Give some space for text
    
    #plt.savefig(os.path.basename(results_case_dir) + "_spatial.png", dpi=300)


def summarize_trajectories_updated(traj, APPLY_VELOCITY_FILTER=False):
    ARTIFACT_CLUSTER_ID = -1
    rows = []
    
    # Two separate dictionaries: one for all clusters and one for filtered bubbles
    # The all_cluster_stats will be used for plotting and regular cluster operations
    all_cluster_stats = defaultdict(lambda: {'total_vxa': 0.0, 'total_area': 0.0, 'count': 0})
    
    # The bubble_cluster_stats will store only bubble clusters with velocity filtering
    # We'll use this for the overall bubble velocity calculation
    bubble_cluster_stats = defaultdict(lambda: {'total_vxa': 0.0, 'total_area': 0.0, 'count': 0})
    
    for bid, rec in traj.items():
        if not rec or rec[-1][5] == ARTIFACT_CLUSTER_ID:
            continue
            
        start, end = rec[0][0], rec[-1][0]
        
        # Calculate both filtered and unfiltered velocities
        all_speeds = [r[3] for r in rec if r[3] > 0]
        
        # Filtered speeds (used only for bubble clusters if APPLY_VELOCITY_FILTER is True)
        if APPLY_VELOCITY_FILTER:
            filtered_speeds = [r[3] for r in rec if 0 < r[3] <= MAX_VELOCITY_THRESHOLD_PIX]
        else:
            filtered_speeds = all_speeds  # Use all speeds if filter is off
        
        # Calculate average velocities
        avg_v = (sum(all_speeds)/len(all_speeds))*PIXEL_TO_METER if all_speeds else 0.0
        
        # Calculate filtered average velocity (used only for bubble clusters)
        avg_v_filtered = (sum(filtered_speeds)/len(filtered_speeds))*PIXEL_TO_METER if filtered_speeds else 0.0
        
        # Get final metrics for area-weighted calculations
        final_diam = rec[-1][4]
        final_area_px = rec[-1][6]  
        cluster_id = rec[-1][5]
        cluster_name = map_shape_name(cluster_id)
        
        # Always update all_cluster_stats (used for plots and regular operations)
        all_cluster_stats[cluster_id]['total_vxa'] += avg_v * final_area_px
        all_cluster_stats[cluster_id]['total_area'] += final_area_px
        all_cluster_stats[cluster_id]['count'] += 1
        
        # Update bubble_cluster_stats only for bubble clusters with filtered velocity
        if cluster_id in BUBBLE_CLUSTER_IDS:
            bubble_cluster_stats[cluster_id]['total_vxa'] += avg_v_filtered * final_area_px
            bubble_cluster_stats[cluster_id]['total_area'] += final_area_px
            bubble_cluster_stats[cluster_id]['count'] += 1
        
        rows.append({
            "bubble_id": bid,
            "start_frame": start,
            "end_frame": end,
            "track_length": len(rec),
            "avg_velocity_m_s": avg_v,
            "final_eq_diameter_mm": final_diam,
            "area_px": final_area_px,
            "cluster": cluster_name,
            "cluster_id": cluster_id
        })
    
    # Add all cluster summaries with area-weighted velocities for plotting
    for cluster_id, stats in all_cluster_stats.items():
        if stats['total_area'] > 0:
            weighted_v = stats['total_vxa'] / stats['total_area']
            rows.append({
                "bubble_id": f"CLUSTER_{cluster_id}_SUMMARY",
                "cluster": map_shape_name(cluster_id),
                "cluster_id": cluster_id,
                "area_weighted_velocity_m_s": weighted_v,
                "count": stats['count']
            })
    
    # Create a DataFrame
    df = pd.DataFrame(rows)
    
    # Calculate overall bubble velocity (using filtered data for bubble clusters only)
    if APPLY_VELOCITY_FILTER:
        total_bubble_vxa = sum(stats['total_vxa'] for stats in bubble_cluster_stats.values())
        total_bubble_area = sum(stats['total_area'] for stats in bubble_cluster_stats.values())
        
        if total_bubble_area > 0:
            bubble_avg_vel = total_bubble_vxa / total_bubble_area
            # Add an additional row to the DataFrame for the overall bubble velocity
            df = pd.concat([df, pd.DataFrame({
                "bubble_id": ["OVERALL_BUBBLE_VELOCITY"],
                "cluster": ["Overall Bubble Clusters"],
                "cluster_id": [-2],  # Use a special ID for this summary
                "area_weighted_velocity_m_s": [bubble_avg_vel],
                "bubble_filtered_velocity": [True]  # Flag to identify this special row
            })], ignore_index=True)
    
    return df
