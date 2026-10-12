#!/usr/bin/env python3
"""
Plot MODTRAN3 radiance spectra (nadir view from 100 km) for different CO2
concentrations, with Planck blackbody curves in the background. Wavelengths in nm.

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
DATA_DIR = "/mnt/user-data/uploads"          # folder containing the files
FILE_GLOB = "MOD_co2_*.txt"
OUT_FILE = "modtran_co2_spectra.png"
EXCLUDE_PPM = [200, 300, 400, 600, 700]       # CO2 concentrations to leave out
BB_TEMPS = [220, 240, 260, 280, 300]          # K
XLIM_NM = (4_000, 60_000)                     # plotted wavelength range (nm)

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23


def read_modtran(path):
    """Return (co2_ppm, wavelength_nm, radiance W/cm2/sr/nm)."""
    co2 = None
    rows = []
    with open(path) as f:
        for line in f:
            if co2 is None:
                m = re.search(r"scalefreon\s+([-+\d.Ee]+)", line)
                if m:
                    co2 = float(m.group(1))
            tok = line.split()
            if len(tok) == 12 and re.fullmatch(r"\d+\.?", tok[0]):
                try:
                    vals = [float(t) for t in tok]
                except ValueError:
                    continue
                # 0 freq, 1 wavelen(um), 8-9 total radiance (per cm-1, per um)
                rows.append((vals[0], vals[1], vals[9]))

    arr = np.array(rows)
    _, idx = np.unique(arr[:, 0], return_index=True)   # drop overlapping block edges
    arr = arr[idx]
    arr = arr[arr[:, 1] > 0]

    wl_nm = arr[:, 1] * 1e3            # um -> nm
    rad_nm = arr[:, 2] * 1e-3          # W/cm2/sr/um -> W/cm2/sr/nm
    order = np.argsort(wl_nm)
    return co2, wl_nm[order], rad_nm[order]


def planck_nm(wl_nm, T):
    """Blackbody spectral radiance in W cm^-2 sr^-1 nm^-1."""
    wl = wl_nm * 1e-9
    B = 2 * H * C**2 / wl**5 / np.expm1(H * C / (wl * KB * T))
    return B * 1e-4 * 1e-9


def main():
    files = glob.glob(os.path.join(DATA_DIR, FILE_GLOB))
    spectra = sorted((read_modtran(f) for f in files), key=lambda s: s[0])
    spectra = [s for s in spectra if s[0] not in EXCLUDE_PPM]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Blackbody curves (background)
    wl_bb = np.linspace(XLIM_NM[0], XLIM_NM[1], 2000)
    bb_colors = plt.cm.Greys(np.linspace(0.35, 0.75, len(BB_TEMPS)))
    for T, c in zip(BB_TEMPS, bb_colors):
        ax.plot(wl_bb, planck_nm(wl_bb, T), color=c, lw=1.4, ls="--", zorder=1)
        wl_pk = 2.8977719e-3 / T * 1e9
        ax.annotate(f"{T} K", xy=(wl_pk, planck_nm(wl_pk, T)),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color="0.25", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.plot([], [], color="0.5", ls="--", lw=1.4, label="Blackbody")

    # MODTRAN spectra, coloured by CO2 concentration
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(spectra)))
    for (co2, wl, rad), col in zip(spectra, colors):
        m = (wl >= XLIM_NM[0]) & (wl <= XLIM_NM[1])
        ax.plot(wl[m], rad[m], color=col, lw=1.2, label=f"{co2:g} ppm", zorder=3)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Spectral radiance (W cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    ax.set_title("MODTRAN upwelling radiance at 100 km for different CO$_2$ concentrations\n"
                 "(tropical atmosphere, nadir view) with blackbody curves")
    ax.set_xlim(*XLIM_NM)
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="CO$_2$ concentration", loc="upper right",
              ncol=2, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=200)
    print(f"Saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
