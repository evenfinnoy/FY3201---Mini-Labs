#!/usr/bin/env python3
"""
Plot MODTRAN3 radiance spectra (nadir-viewing, observer at various altitudes)
with Planck blackbody curves in the background. All wavelengths in nm.

Radiance units: W cm^-2 sr^-1 nm^-1
"""
import re
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DATA_DIR = "/mnt/user-data/uploads"          # folder containing modtran*.txt
FILE_GLOB = "modtran*km.txt"
OUT_FILE = "modtran_spectra.png"
BB_TEMPS = [220, 240, 260, 280, 300]          # K
XLIM_NM = (4_000, 60_000)                    # plotted wavelength range (nm)

# Physical constants (SI)
H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def read_modtran(path):
    """Return (altitude_km, wavelength_nm, radiance W/cm2/sr/nm) from a MODTRAN3 file."""
    alt = None
    rows = []
    # Data lines: FREQ(cm-1)  WAVLEN(um)  then 10 radiance/integral values + TRANS
    num = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"
    data_re = re.compile(rf"^\s*(\d+\.?)\s+({num})\s+(?:{num}\s+){{9}}({num})\s*$")

    with open(path) as f:
        for line in f:
            m = re.match(r"\s*H1\s*=\s*([\d.]+)\s*KM", line)
            if m and alt is None:
                alt = float(m.group(1))
            tok = line.split()
            if len(tok) == 12 and re.fullmatch(r"\d+\.?", tok[0]):
                try:
                    vals = [float(t) for t in tok]
                except ValueError:
                    continue
                # columns: 0 freq, 1 wavelen(um), 2-3 path thermal, 4-5 surf emission,
                # 6-7 surf reflected, 8-9 total radiance (per cm-1, per um),
                # 10 integral, 11 transmittance
                rows.append((vals[0], vals[1], vals[9]))

    arr = np.array(rows)
    # Blocks overlap at their edges -> drop duplicate frequencies, sort ascending
    _, idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[idx]
    arr = arr[arr[:, 1] > 0]

    wl_nm = arr[:, 1] * 1e3            # um -> nm
    rad_nm = arr[:, 2] * 1e-3          # W/cm2/sr/um -> W/cm2/sr/nm
    order = np.argsort(wl_nm)
    return alt, wl_nm[order], rad_nm[order]


# ----------------------------------------------------------------------------
# Planck function
# ----------------------------------------------------------------------------
def planck_nm(wl_nm, T):
    """Blackbody spectral radiance in W cm^-2 sr^-1 nm^-1."""
    wl = wl_nm * 1e-9                                       # m
    B = 2 * H * C**2 / wl**5 / np.expm1(H * C / (wl * KB * T))   # W m^-2 sr^-1 m^-1
    return B * 1e-4 * 1e-9                                  # -> W cm^-2 sr^-1 nm^-1


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def main():
    files = glob.glob(os.path.join(DATA_DIR, FILE_GLOB))
    spectra = sorted((read_modtran(f) for f in files), key=lambda s: s[0])

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Blackbody curves (background)
    wl_bb = np.linspace(XLIM_NM[0], XLIM_NM[1], 2000)
    bb_colors = plt.cm.Greys(np.linspace(0.35, 0.75, len(BB_TEMPS)))
    for T, c in zip(BB_TEMPS, bb_colors):
        b = planck_nm(wl_bb, T)
        ax.plot(wl_bb, b, color=c, lw=1.4, ls="--", zorder=1)
        # label at the peak (Wien)
        wl_pk = 2.8977719e-3 / T * 1e9
        ax.annotate(f"{T} K", xy=(wl_pk, planck_nm(wl_pk, T)),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color="0.25", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.plot([], [], color="0.5", ls="--", lw=1.4, label="Blackbody")

    # MODTRAN spectra, coloured by altitude
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(spectra)))
    for (alt, wl, rad), col in zip(spectra, colors):
        m = (wl >= XLIM_NM[0]) & (wl <= XLIM_NM[1])
        ax.plot(wl[m], rad[m], color=col, lw=1.2,
                label=f"{alt:g} km" if alt > 0.01 else "0 km (surface)", zorder=3)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Spectral radiance (W cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    ax.set_title("MODTRAN upwelling radiance at different observer heights\n"
                 "(tropical atmosphere, nadir view) with blackbody curves")
    ax.set_xlim(*XLIM_NM)
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Observer altitude", loc="upper right",
              ncol=2, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=200)
    print(f"Saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
