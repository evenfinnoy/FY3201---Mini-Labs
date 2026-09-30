"""
Mini-Lab 3, Part 1 - Solar spectrum versus blackbody spectrum.

1. Load kurudz_0.1nm.dat and convert irradiance from mW m^-2 nm^-1 to W m^-2 nm^-1.
2. Plot the measured spectrum together with a 5778 K blackbody scaled to
   the top of Earth's atmosphere (mean Sun-Earth distance).
3. Integrate the measured spectrum to estimate the solar constant.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
# givenCode.py lives one level up, in Mini-Lab3/
sys.path.insert(0, str(HERE.parent))
from givenCode import BlackBodySun, StefanBoltzmann

T_SUN = 5778      # K, effective temperature of the Sun
T_BB_2 = 6100     # K, extra blackbody curves in part1_solar_vs_blackbody.png
T_BB_3 = 6400     # K

KURUCZ_COLOR = "0.35"  # grey, so the data stays readable on top of the rainbow


def wavelength_to_rgb(wl):
    """Approximate RGB colour of a visible wavelength in nm (Dan Bruton's algorithm)."""
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif 440 <= wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif 490 <= wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif 580 <= wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # Fade out towards the edges of human vision
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / 40
    elif wl > 700:
        factor = 0.3 + 0.7 * (750 - wl) / 50
    else:
        factor = 1.0
    return (r * factor, g * factor, b * factor)


def add_visible_spectrum(ax, alpha=0.5, wl_min=380, wl_max=750, step=1):
    """Shade the visible band (380-750 nm) in rainbow colours behind the data."""
    for wl in np.arange(wl_min, wl_max, step):
        ax.axvspan(wl, wl + step, color=wavelength_to_rgb(wl + step / 2),
                   alpha=alpha, lw=0, zorder=0)


# ---------------------------------------------------------------------------
# 1. Load data and convert units
# ---------------------------------------------------------------------------
# File header: "converted to mW / (m2 nm) and averaged over 0.1nm intervals"
# Column 1: wavelength [nm], column 2: spectral irradiance [mW m^-2 nm^-1]
data = np.loadtxt(HERE / "kurudz_0.1nm.dat", comments="#")
wavelength_nm = data[:, 0]
irradiance = data[:, 1] / 1000.0  # mW -> W  =>  W m^-2 nm^-1

print(f"Wavelength range: {wavelength_nm.min():.1f} - {wavelength_nm.max():.1f} nm "
      f"({len(wavelength_nm)} points)")
print(f"Peak irradiance: {irradiance.max():.3f} W m^-2 nm^-1 "
      f"at {wavelength_nm[np.argmax(irradiance)]:.1f} nm")

# ---------------------------------------------------------------------------
# 2. Blackbody at 5778 K, scaled to Earth's distance
# ---------------------------------------------------------------------------
# BlackBodySun takes wavelength in metres and returns W m^-2 nm^-1 at 1 AU:
#   E = pi * B(lambda, T) * (R_sun / d)^2
bb_irradiance = BlackBodySun(wavelength_nm * 1e-9, T_SUN)
bb_irradiance_2 = BlackBodySun(wavelength_nm * 1e-9, T_BB_2)
bb_irradiance_3 = BlackBodySun(wavelength_nm * 1e-9, T_BB_3)

fig, ax = plt.subplots(figsize=(11, 5.5))
add_visible_spectrum(ax)
ax.plot(wavelength_nm, irradiance, lw=0.4, color=KURUCZ_COLOR,
        label="Kurucz solar spectrum (TOA)")
ax.plot(wavelength_nm, bb_irradiance, lw=2, color="k",
        label=f"Blackbody {T_SUN} K, scaled to 1 AU")
ax.plot(wavelength_nm, bb_irradiance_2, lw=2, color="tab:blue", ls="--",
        label=f"Blackbody {T_BB_2} K, scaled to 1 AU")
ax.plot(wavelength_nm, bb_irradiance_3, lw=2, color="tab:purple", ls="-.",
        label=f"Blackbody {T_BB_3} K, scaled to 1 AU")
ax.set_xlim(250, 3000)
ax.set_ylim(bottom=0)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel(r"Spectral irradiance [W m$^{-2}$ nm$^{-1}$]")
ax.set_title(f"Kurucz solar spectrum vs. {T_SUN} K, {T_BB_2} K and {T_BB_3} K blackbodies")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "part1_solar_vs_blackbody.png", dpi=200)

# Full range on log axes to show the IR tail as well
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
add_visible_spectrum(ax2)
ax2.loglog(wavelength_nm, irradiance, lw=0.4, color=KURUCZ_COLOR,
           label="Kurucz solar spectrum (TOA)")
ax2.loglog(wavelength_nm, bb_irradiance, lw=2, color="k",
           label=f"Blackbody {T_SUN} K, scaled to 1 AU")
ax2.set_xlabel("Wavelength [nm]")
ax2.set_ylabel(r"Spectral irradiance [W m$^{-2}$ nm$^{-1}$]")
ax2.set_title("Kurucz solar spectrum vs. blackbody, full range (log-log)")
ax2.grid(alpha=0.3, which="both")
ax2.legend()
fig2.tight_layout()
fig2.savefig(HERE / "part1_solar_vs_blackbody_loglog.png", dpi=200)

# Relative deviation between Kurucz and blackbody. The 0.1 nm data is dominated
# by absorption lines, so a 10 nm running mean is overlaid to show the trend.
window = 100  # points = 10 nm at 0.1 nm resolution
kernel = np.ones(window) / window

rel_deviation = 100 * (irradiance - bb_irradiance) / bb_irradiance  # %
rel_deviation_smooth = np.convolve(rel_deviation, kernel, mode="same")

fig3, ax3 = plt.subplots(figsize=(11, 5.5))
add_visible_spectrum(ax3)
ax3.plot(wavelength_nm, rel_deviation, lw=0.4, color=KURUCZ_COLOR, alpha=0.6,
         label="Relative deviation (0.1 nm)")
ax3.plot(wavelength_nm, rel_deviation_smooth, lw=1.5, color="k",
         label="10 nm running mean")
ax3.axhline(0, color="grey", lw=0.8)
ax3.set_ylim(-100, 60)
ax3.set_xlim(250, 3000)
ax3.set_xlabel("Wavelength [nm]")
ax3.set_ylabel("Relative deviation [%]")
ax3.set_title(f"Relative deviation between Kurucz spectrum and {T_SUN} K blackbody")
ax3.grid(alpha=0.3)
ax3.legend()
fig3.tight_layout()
fig3.savefig(HERE / "part1_deviation.png", dpi=200)

# Effective emissivity: Kurucz / blackbody. A perfect 5778 K blackbody gives 1.
emissivity = irradiance / bb_irradiance
emissivity_smooth = np.convolve(emissivity, kernel, mode="same")

fig4, ax4 = plt.subplots(figsize=(11, 5.5))
add_visible_spectrum(ax4)
ax4.plot(wavelength_nm, emissivity, lw=0.4, color=KURUCZ_COLOR, alpha=0.6,
         label="Kurucz / blackbody (0.1 nm)")
ax4.plot(wavelength_nm, emissivity_smooth, lw=1.5, color="k",
         label="10 nm running mean")
ax4.axhline(1, color="grey", lw=0.8)
ax4.set_ylim(0, 1.6)
ax4.set_xlim(250, 3000)
ax4.set_xlabel("Wavelength [nm]")
ax4.set_ylabel(r"Emissivity $\varepsilon(\lambda)$ [-]")
ax4.set_title(f"Effective emissivity of the Sun relative to a {T_SUN} K blackbody")
ax4.grid(alpha=0.3)
ax4.legend()
fig4.tight_layout()
fig4.savefig(HERE / "part1_emissivity.png", dpi=200)

# ---------------------------------------------------------------------------
# 3. Solar constant by trapezoid integration
# ---------------------------------------------------------------------------
S_measured = np.trapezoid(irradiance, wavelength_nm)       # W m^-2
S_bb_range = np.trapezoid(bb_irradiance, wavelength_nm)    # same 250-10000 nm range

# Total blackbody irradiance at 1 AU (all wavelengths) for reference
R_sun, d_sun_earth = 696340.0, 149600000.0  # km
S_bb_total = StefanBoltzmann(T_SUN) * (R_sun / d_sun_earth) ** 2



print()
print(f"Solar constant from Kurucz data (250-10000 nm):   {S_measured:7.1f} W m^-2")
print(f"Blackbody 5778 K, same wavelength range:         {S_bb_range:7.1f} W m^-2")
print(f"Blackbody 5778 K, all wavelengths (sigma T^4):   {S_bb_total:7.1f} W m^-2")
print(f"Fraction of blackbody energy outside 250-10000 nm: "
      f"{100 * (1 - S_bb_range / S_bb_total):.1f} %")

plt.show()
