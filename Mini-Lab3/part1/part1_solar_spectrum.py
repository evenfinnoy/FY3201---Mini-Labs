"""
Mini-Lab 3, Part 1 - Solar spectrum versus blackbody spectrum.

1. Load kurudz_0.1nm.dat and convert irradiance from mW m^-2 nm^-1 to W m^-2 nm^-1.
2. Plot the measured spectrum together with a 5778 K blackbody scaled to
   the top of Earth's atmosphere (mean Sun-Earth distance).
3. Integrate the measured spectrum to estimate the solar constant.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from givenCode import BlackBodySun, StefanBoltzmann

HERE = Path(__file__).parent
T_SUN = 5778  # K

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

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(wavelength_nm, irradiance, lw=0.4, color="tab:orange",
        label="Kurucz solar spectrum (TOA)")
ax.plot(wavelength_nm, bb_irradiance, lw=2, color="k",
        label=f"Blackbody {T_SUN} K, scaled to 1 AU")
ax.set_xlim(250, 3000)
ax.set_ylim(bottom=0)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel(r"Spectral irradiance [W m$^{-2}$ nm$^{-1}$]")
ax.set_title("Solar spectrum at top of atmosphere vs. 5778 K blackbody")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "part1_solar_vs_blackbody.png", dpi=200)

# Full range on log axes to show the IR tail as well
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
ax2.loglog(wavelength_nm, irradiance, lw=0.4, color="tab:orange",
           label="Kurucz solar spectrum (TOA)")
ax2.loglog(wavelength_nm, bb_irradiance, lw=2, color="k",
           label=f"Blackbody {T_SUN} K, scaled to 1 AU")
ax2.set_xlabel("Wavelength [nm]")
ax2.set_ylabel(r"Spectral irradiance [W m$^{-2}$ nm$^{-1}$]")
ax2.set_title("Solar spectrum vs. blackbody, full range (log-log)")
ax2.grid(alpha=0.3, which="both")
ax2.legend()
fig2.tight_layout()
fig2.savefig(HERE / "part1_solar_vs_blackbody_loglog.png", dpi=200)

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
