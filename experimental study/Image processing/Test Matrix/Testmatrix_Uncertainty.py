import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterExponent

# Schriftstil
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "text.usetex": True
})

# === Excel einlesen ===
dateiname = "results_Testmatrix.xlsx"
df = pd.read_excel(dateiname, sheet_name=0)
df.columns = df.columns.str.strip()


# === Geometrieparameter ===
B_p = 0.08        # Plattenbreite [m]
a = 0.001         # Amplitude der Sinuswelle [m]
b_p = 2 * a       # Plattentiefe [m]
A_f = B_p * b_p   # Strömungsquerschnitt [m²]

# === Unsicherheiten der Geometrie ===
e_B_p = 0.0005    # [m]
e_b_p = 0.00005   # [m]
lambda_ = 0.00866   # [m]
phi_deg = 60
phi_rad = np.deg2rad(phi_deg)

# Approximationsfehler (e_approx)
e_approx = (lambda_ * b_p) / (2 * np.pi * np.cos(phi_rad))

# Fehler der Fläche A_f
e_A_f = np.sqrt((b_p * e_B_p)**2 + (B_p * e_b_p)**2 + e_approx**2)


# === Volumenströme umrechnen [m³/s] ===
df["V_dot_liquid"] = df["V_punkt_L [m^3/h]"] / 3600
df["V_dot_gas"] = df["V_punkt_G [ml/min]"] * 1e-6 / 60

# === Superficial velocities [m/s] ===
df["U_s_liquid"] = df["V_dot_liquid"] / A_f
df["U_s_gas"] = df["V_dot_gas"] / A_f


# === Gerätespezifische Fehlerangaben ===
# Flüssigkeit (Picomag IO-Link)
Q_l_max = 150 / 60 / 1000  # [m³/s] → 150 L/min
# Gas (GFC-17)
Q_g_max = 10 / 60 / 1000   # [m³/s] → 10 L/min

e_Q_l = 0.008 *  df["V_dot_liquid"] + 0.002 * df["V_dot_liquid"] + 0.001 * Q_l_max # 0.8% vom Messwert + 0.2% vom Messwert (Wiederholbarkeit) + 0,1% Endwert

e_Q_g = (0.01+0.0025) * Q_g_max  # 1 % v. Endwer + 0,25% Endwert (Wiederholbarkeir)

# === Durchflussfehler berechnen ===
df["e_Q_l"] = e_Q_l
df["e_Q_g"] = e_Q_g

# === Fehler der superficial velocities ===
df["e_U_s_liquid"] = df["U_s_liquid"] * np.sqrt((e_A_f / A_f)**2 + (df["e_Q_l"] / df["V_dot_liquid"])**2)
df["e_U_s_gas"] = df["U_s_gas"] * np.sqrt((e_A_f / A_f)**2 + (df["e_Q_g"] / df["V_dot_gas"])**2)

# === Plot vorbereiten ===
zero_mask = df["U_s_gas"] == 0
nonzero_mask = ~zero_mask

max_error_gas_percent = (df["e_U_s_gas"] / df["U_s_gas"]).max(skipna=True) * 100
max_error_liquid_percent = (df["e_U_s_liquid"] / df["U_s_liquid"]).max(skipna=True) * 100
print(df[["V_dot_liquid", "e_Q_l", "V_dot_gas", "e_Q_g", "U_s_liquid", "e_U_s_liquid", "U_s_gas", "e_U_s_gas"]])
print(f"Max. Fehler Gas: {max_error_gas_percent:.2f}%")
print(f"Max. Fehler Flüssigkeit: {max_error_liquid_percent:.2f}%")


# === Plot 1: Hauptplot (log-log, nur U_s,G > 0)
plt.figure(figsize=(8, 6))
plt.errorbar(
    df.loc[nonzero_mask, "U_s_gas"],
    df.loc[nonzero_mask, "U_s_liquid"],
    xerr=df.loc[nonzero_mask, "e_U_s_gas"],
    yerr=df.loc[nonzero_mask, "e_U_s_liquid"],
    fmt='o', markersize=4, ecolor='blue', capsize=2, color='red'
)

plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-3, 4e-1)
plt.ylim(1e-1, 2e0)
plt.xlabel(r"Scheinbare Gasgeschwindigkeit $U_{s,G} \;\left[ \frac{m}{s} \right]$")
plt.ylabel(r"Scheinbare Flüssigkeitsgeschwindigkeit $U_{s,L} \;\left[ \frac{m}{s} \right]$")
#plt.title("Testmatrix")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# === Plot 2: Nullpunkte (linear-log)
plt.figure(figsize=(2, 6))
plt.errorbar(
    np.zeros_like(df.loc[zero_mask, "U_s_liquid"]),
    df.loc[zero_mask, "U_s_liquid"],
    yerr=df.loc[zero_mask, "e_U_s_liquid"],
    fmt='o', markersize=2, ecolor='blue', capsize=3, color ='red' ,label=r"$U_{s,G} = 0$"
)
ax = plt.gca()
ax.set_xticks([0])
ax.set_xticklabels(["0"])
plt.xlim(0, 0)
plt.yscale("log")
plt.ylim(1e-1, 2e0)
plt.xlabel(r"$U_{s,G} = 0$")
plt.ylabel(r"$U_{s,L} \;\left[ \frac{m}{s} \right]$")
#plt.title("Nullpunkte")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#plt.legend()
plt.tight_layout()
plt.show()