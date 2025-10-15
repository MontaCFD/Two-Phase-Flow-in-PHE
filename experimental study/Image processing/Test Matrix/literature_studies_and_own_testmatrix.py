import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ===== LANGUAGE SELECTOR =====
LANGUAGE = "deutsch"  # Options: "english", "deutsch"
LANGUAGE = "english"

# Language dictionary for all text elements
TEXT = {
    "english": {
        "data_points": "Measurement Points",
        "zero_points": "Measurement Points ($U_{s,G} = 0$)",
        "present_work": "Present Work",
        "x_label": r"Superficial Gas Velocity $U_{s,G} \;\left[ \frac{m}{s} \right]$",
        "y_label": r"Superficial Liquid Velocity $U_{s,L} \;\left[ \frac{m}{s} \right]$",
        "title": r"Investigation Range of This Work Compared to Other Studies",
        "max_gas_error": "Max. Gas Error",
        "max_liquid_error": "Max. Liquid Error",
        "zero_annotation": r"\textbf{$U_{s,G} = 0$}",
        "file_suffix": "_en"
    },
    "deutsch": {
        "data_points": "Messpunkte",
        "zero_points": "Messpunkte ($U_{s,G} = 0$)",
        "present_work": "Diese Arbeit",
        "x_label": r"Scheinbare Gasgeschwindigkeit $U_{s,G} \;\left[ \frac{m}{s} \right]$",
        "y_label": r"Scheinbare Flüssigkeitsgeschwindigkeit $U_{s,L} \;\left[ \frac{m}{s} \right]$",
        "title": r"Untersuchungsbereich dieser Arbeit im Vergleich zu anderen Studien",
        "max_gas_error": "Max. Fehler Gas",
        "max_liquid_error": "Max. Fehler Flüssigkeit",
        "zero_annotation": r"\textbf{$U_{s,G} = 0$}",
        "file_suffix": "_de"
    }
}

# Get current language texts
t = TEXT[LANGUAGE]

# Style settings
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "text.usetex": True
})

# === Read Excel data ===
filename = "results_Testmatrix.xlsx"
df = pd.read_excel(filename, sheet_name=0)
df.columns = df.columns.str.strip()

# === Geometry parameters ===
B_p = 0.08        # Plate width [m]
a = 0.001         # Sine wave amplitude [m]
b_p = 2 * a       # Plate depth [m]
A_f = B_p * b_p   # Flow cross-section [m²]

# === Geometry uncertainties ===
e_B_p = 0.0005    # [m]
e_b_p = 0.00005   # [m]
lambda_ = 0.00866 # [m]
phi_deg = 60
phi_rad = np.deg2rad(phi_deg)

# Approximation error (e_approx)
e_approx = (lambda_ * b_p) / (2 * np.pi * np.cos(phi_rad))

# Error of area A_f
e_A_f = np.sqrt((b_p * e_B_p)**2 + (B_p * e_b_p)**2 + e_approx**2)

# === Convert volume flows to [m³/s] ===
df["V_dot_liquid"] = df["V_punkt_L [m^3/h]"] / 3600
df["V_dot_gas"] = df["V_punkt_G [ml/min]"] * 1e-6 / 60

# === Superficial velocities [m/s] ===
df["U_s_liquid"] = df["V_dot_liquid"] / A_f
df["U_s_gas"] = df["V_dot_gas"] / A_f

# === Device-specific error specifications ===
# Liquid (Picomag IO-Link)
Q_l_max = 150 / 60 / 1000  # [m³/s] → 150 L/min
# Gas (GFC-17)
Q_g_max = 10 / 60 / 1000   # [m³/s] → 10 L/min

e_Q_l = 0.008 * df["V_dot_liquid"] + 0.002 * df["V_dot_liquid"] + 0.001 * Q_l_max
e_Q_g = (0.01 + 0.0025) * Q_g_max

# === Calculate flow errors ===
df["e_Q_l"] = e_Q_l
df["e_Q_g"] = e_Q_g

# === Errors of superficial velocities ===
df["e_U_s_liquid"] = df["U_s_liquid"] * np.sqrt((e_A_f / A_f)**2 + (df["e_Q_l"] / df["V_dot_liquid"])**2)
df["e_U_s_gas"] = df["U_s_gas"] * np.sqrt((e_A_f / A_f)**2 + (df["e_Q_g"] / df["V_dot_gas"])**2)

# === Masks for data selection ===
zero_mask = df["U_s_gas"] == 0
nonzero_mask = ~zero_mask

# === Helper function for study region boxes ===
def create_box(xmin, xmax, ymin, ymax):
    return [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]

# === Study regions definition ===
regions = {
    r"Tribbe \& Müller-Steinhagen": {"x": (1.8, 55), "y": (0.04, 0.6), "color": "green", "alpha": 0.2},
    r"Vlasogiannis et al.": {"x": (0.3, 10), "y": (0.01, 0.25), "color": "purple", "alpha": 0.2},
    r"Shiomi et al.": {"x": (0.14, 1.2), "y": (0.01, 0.56), "color": "orange", "alpha": 0.2},
    r"Asano et al.": {"x": (0.8, 8), "y": (0.02, 0.04), "color": "cyan", "alpha": 0.2},
    r"Nilpueng \& Wongwises": {"x": (0.06, 4), "y": (0.03, 0.3), "color": "magenta", "alpha": 0.2},
    r"Grabenstein et al.": {"x": (0.7, 15), "y": (0.03, 0.7), "color": "blue", "alpha": 0.2},
    r"Buscher": {"x": (0.007, 30), "y": (0.04, 0.4), "color": "red", "alpha": 0.2},
    r"Passoni et al.": {"x": (0.006, 3.3), "y": (0.006, 0.5), "color": "brown", "alpha": 0.2},
}

# === Define our own study region based on data bounds ===
# Calculate min/max of our experimental data (non-zero gas points)
x_min = df.loc[nonzero_mask, "U_s_gas"].min() * 0.8  # Add some margin
x_max = df.loc[nonzero_mask, "U_s_gas"].max() * 1.2
y_min = df.loc[nonzero_mask, "U_s_liquid"].min() * 0.8
y_max = df.loc[nonzero_mask, "U_s_liquid"].max() * 1.2

# Add our own study to the regions dictionary
regions[t["present_work"]] = {"x": (x_min, x_max), "y": (y_min, y_max), "color": "black", "alpha": 0.3}

# === Create single plot ===
plt.figure(figsize=(12, 9))

# First plot all study regions
for label, data in regions.items():
    x_vals, y_vals = create_box(*data["x"], *data["y"])
    plt.fill(x_vals, y_vals, alpha=data["alpha"], label=label, color=data["color"])
    plt.plot(x_vals + [x_vals[0]], y_vals + [y_vals[0]], linestyle='--', color=data["color"])

# Then plot our experimental points with error bars (non-zero gas)
plt.errorbar(
    df.loc[nonzero_mask, "U_s_gas"],
    df.loc[nonzero_mask, "U_s_liquid"],
    xerr=df.loc[nonzero_mask, "e_U_s_gas"],
    yerr=df.loc[nonzero_mask, "e_U_s_liquid"],
    fmt='o', markersize=6, ecolor='blue', capsize=3, color='red',
    label=t["data_points"]
)

# Plot zero gas points at the left edge of the plot
min_x = 0.004  # Position slightly to the right of y-axis for visibility
plt.errorbar(
    [min_x] * sum(zero_mask),
    df.loc[zero_mask, "U_s_liquid"],
    yerr=df.loc[zero_mask, "e_U_s_liquid"],
    fmt='o', markersize=6, ecolor='blue', capsize=3, color='darkred',
    label=t["zero_points"]
)

# Add annotation for zero gas points
plt.annotate(t["zero_annotation"], xy=(min_x*1, df.loc[zero_mask, "U_s_liquid"].mean()), 
             xytext=(min_x*1.2, df.loc[zero_mask, "U_s_liquid"].mean()))

# Configure axes
plt.xscale("log")
plt.yscale("log")
plt.xlim(0.003, 100)
plt.ylim(0.005, 3)

# Set labels and title using language selector
plt.xlabel(t["x_label"], fontsize=14)
plt.ylabel(t["y_label"], fontsize=14)
# plt.title(t["title"], fontsize=16)  # Uncomment if you want the title
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Add legend with smaller font and outside the plot
plt.legend(loc='best', fontsize=11.5, ncol=2)

# Calculate and display error statistics
max_error_gas_percent = (df.loc[nonzero_mask, "e_U_s_gas"] / df.loc[nonzero_mask, "U_s_gas"]).max() * 100
max_error_liquid_percent = (df["e_U_s_liquid"] / df["U_s_liquid"]).max() * 100

print(f"{t['max_gas_error']}: {max_error_gas_percent:.2f}%")
print(f"{t['max_liquid_error']}: {max_error_liquid_percent:.2f}%")

# Save files with language-specific suffix
plt.tight_layout()
plt.savefig(f"test_matrix_and_previous_studies{t['file_suffix']}.pdf", format="pdf", dpi=400)
plt.savefig(f"test_matrix_and_previous_studies{t['file_suffix']}.svg", format="svg", dpi=400)
plt.show()
