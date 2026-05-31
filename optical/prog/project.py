"""Веб-інтерфейс для моделювання оптичного відгуку металічних наночастинок методом DDA."""



from __future__ import annotations



from dataclasses import dataclass, field

from typing import List, Tuple

import csv

import io

import json

import math



import numpy as np
from scipy.special import spherical_jn, spherical_yn

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for



C_LIGHT = 299_792_458.0  # m/s

EPS0 = 8.8541878128e-12  # F/m

MAX_SOLVER_DIPOLES = 280

PREVIEW_MAX_DIPOLES = 2500

PREVIEW_MOMENT_DIPOLES = 120

PREVIEW_MOMENT_ARROWS = 48

NEAR_FIELD_SAMPLE_POINTS = 360

MIN_DIPOLE_SPACING_NM = 0.5

MAX_RADIUS_NM = 500.0

MIN_RADIUS_NM = 2.0



ENVIRONMENT_INDEXES = {

    "Air": 1.0,

    "Water": 1.33,

    "Glass": 1.5,

}



# Параметри моделі Друде (ω у рад/с), джерело: Johnson & Christy / Palik (наближено)

DRUDE_PARAMS = {

    "Au": {"eps_inf": 9.5, "omega_p": 1.37e16, "gamma": 1.07e14},

    "Ag": {"eps_inf": 3.7, "omega_p": 1.39e16, "gamma": 2.73e13},

    "Al": {"eps_inf": 1.0, "omega_p": 2.24e16, "gamma": 1.22e14},

}



# Додаткові осцилятори Лоренца (модель Drude-Lorentz)

LORENTZ_PARAMS = {

    "Au": [(0.76, 4.05e15, 4.0e14)],

    "Ag": [(0.24, 8.0e15, 3.5e14)],

    "Al": [],

}



MATERIAL_COLORS = {

    "Au": "#d4af37",

    "Ag": "#c0c0c0",

    "Al": "#a8b4c4",

}



app = Flask(__name__, template_folder="templates", static_folder="static")





@dataclass

class Nanoparticle:

    material: str = "Au"

    material_model: str = "Drude model"

    shape: str = "sphere"

    radius_nm: float = 50.0

    aspect_ratio: float = 1.0

    dipole_spacing_nm: float = 5.0

    num_dipoles: int = 1000

    orientation_axis: str = "X"

    theta_deg: float = 0.0

    phi_deg: float = 0.0

    re_epsilon: float = 1.0

    im_epsilon: float = 0.0



    def summary(self) -> str:

        return (

            f"Матеріал: {self.material} ({self.material_model}), форма: {self.shape}, "

            f"радіус: {self.radius_nm} нм, крок диполя: {self.dipole_spacing_nm} нм, "

            f"N диполів: {self.num_dipoles}, орієнтація: {self.orientation_axis}, "

            f"θ={self.theta_deg}°, φ={self.phi_deg}°, εʹ={self.re_epsilon}, εʹʹ={self.im_epsilon}"

        )





@dataclass

class SimulationSettings:

    wavelength_min_nm: float = 400.0

    wavelength_max_nm: float = 900.0

    wavelength_step_nm: float = 10.0

    environment: str = "Air"

    ambient_index: float = 1.0

    spectrum_type: str = "All"

    polarization: str = "X"

    intensity: float = 1.0

    field_amplitude: float = 1.0

    lattice_type: str = "Cubic lattice"

    max_iterations: int = 100

    error_tolerance: float = 1e-5

    solver: str = "Conjugate Gradient"

    show_electric_field: bool = True

    show_dipoles: bool = True

    show_near_field: bool = False

    graph_type: str = "λ vs Extinction"

    temperature: float = 300.0

    preview_wavelength_nm: float | None = None

    light_theta_deg: float = 180.0

    light_phi_deg: float = 0.0



    def wavelengths(self) -> List[float]:

        step = self.wavelength_step_nm if self.wavelength_step_nm > 0 else 1.0

        if self.wavelength_max_nm <= self.wavelength_min_nm:

            return [self.wavelength_min_nm]

        nsteps = int(max(1, math.floor((self.wavelength_max_nm - self.wavelength_min_nm) / step)))

        return [self.wavelength_min_nm + i * step for i in range(nsteps + 1)]



    def summary(self) -> str:

        return (

            f"Довжина хвилі: {self.wavelength_min_nm}-{self.wavelength_max_nm} нм, "

            f"крок: {self.wavelength_step_nm} нм, середовище: {self.environment} (n={self.ambient_index}), "

            f"поляризація: {self.polarization}, I={self.intensity}, E₀={self.field_amplitude}, "

            f"метод: {self.solver}, графік: {self.graph_type}"

        )





@dataclass

class DDAModelResult:

    wavelengths: List[float] = field(default_factory=list)

    frequency: List[float] = field(default_factory=list)

    wavenumber: List[float] = field(default_factory=list)

    extinction: List[float] = field(default_factory=list)

    scattering: List[float] = field(default_factory=list)

    absorption: List[float] = field(default_factory=list)

    cross_ext: List[float] = field(default_factory=list)

    cross_sca: List[float] = field(default_factory=list)

    cross_abs: List[float] = field(default_factory=list)

    solver_iterations: int = 0

    dipoles_used: int = 0





def frequency_from_wavelength_nm(wavelength_nm: float) -> float:

    """f = c / λ"""

    return C_LIGHT / (wavelength_nm * 1e-9)





def wavenumber_from_wavelength_nm(wavelength_nm: float) -> float:

    """k = 2π / λ"""

    return 2.0 * math.pi / (wavelength_nm * 1e-9)





def omega_from_wavelength_nm(wavelength_nm: float) -> float:

    return 2.0 * math.pi * C_LIGHT / (wavelength_nm * 1e-9)





def drude_epsilon(omega: float, material: str, temperature_k: float) -> complex:

    """ε(ω) = ε∞ − ωp² / (ω² + iγω) з температурною поправкою γ."""

    params = DRUDE_PARAMS.get(material, DRUDE_PARAMS["Au"])

    gamma = params["gamma"] * (1.0 + 0.0003 * (temperature_k - 300.0))

    eps_inf = params["eps_inf"]

    omega_p = params["omega_p"]

    denom = omega * omega + 1j * gamma * omega

    return eps_inf - (omega_p * omega_p) / denom





def drude_lorentz_epsilon(omega: float, material: str, temperature_k: float) -> complex:

    eps = drude_epsilon(omega, material, temperature_k)

    for strength, omega0, gamma_l in LORENTZ_PARAMS.get(material, []):

        denom = omega0 * omega0 - omega * omega - 1j * gamma_l * omega

        eps += strength * omega0 * omega0 / denom

    return eps





def material_epsilon(

    omega: float,

    particle: Nanoparticle,

    settings: SimulationSettings,

) -> complex:

    """Діелектрична проникність частинки для заданої моделі."""

    if particle.material_model == "Experimental data":

        return complex(particle.re_epsilon, particle.im_epsilon)

    if particle.material_model == "Drude-Lorentz":

        return drude_lorentz_epsilon(omega, particle.material, settings.temperature)

    return drude_epsilon(omega, particle.material, settings.temperature)


def _riccati_psi(n: int, z: complex) -> complex:
    return z * spherical_jn(n, z)


def _riccati_xi(n: int, z: complex) -> complex:
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    hn = jn + 1j * yn
    return z * hn


def _riccati_psi_deriv(n: int, z: complex) -> complex:
    jn = spherical_jn(n, z)
    if n == 0:
        jn_m1 = np.cos(z) / z if abs(z) > 1e-12 else 0.0
    else:
        jn_m1 = spherical_jn(n - 1, z)
    return jn - z * jn_m1


def _riccati_xi_deriv(n: int, z: complex) -> complex:
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    if n == 0:
        jn_m1 = np.cos(z) / z if abs(z) > 1e-12 else 0.0
        yn_m1 = -np.sin(z) / z if abs(z) > 1e-12 else 0.0
    else:
        jn_m1 = spherical_jn(n - 1, z)
        yn_m1 = spherical_yn(n - 1, z)
    hn = jn + 1j * yn
    hn_m1 = jn_m1 + 1j * yn_m1
    return hn - z * hn_m1


def mie_efficiencies(
    size_parameter: float,
    refr_index: complex,
    n_terms: int = 40,
) -> Tuple[float, float, float]:
    """
    Ефективні перерізи Міє для сфери (Bohren & Huffman).
    x = 2π n_m a / λ, m = n_particle / n_medium.
    """
    x = complex(size_parameter)
    m = complex(refr_index)
    if abs(x) < 1e-12:
        return 0.0, 0.0, 0.0

    n_max = max(2, int(n_terms))
    s_ext = 0.0
    s_sca = 0.0
    for n in range(1, n_max + 1):
        psi_x = _riccati_psi(n, x)
        psi_mx = _riccati_psi(n, m * x)
        xi_x = _riccati_xi(n, x)
        dpsi_x = _riccati_psi_deriv(n, x)
        dpsi_mx = _riccati_psi_deriv(n, m * x)
        dxi_x = _riccati_xi_deriv(n, x)

        a_den = m * psi_mx * dpsi_x - psi_x * dpsi_mx
        b_den = psi_mx * dpsi_x - m * psi_x * dpsi_mx
        if abs(a_den) < 1e-30 or abs(b_den) < 1e-30:
            continue
        a_n = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / (m * psi_mx * dxi_x - xi_x * dpsi_mx)
        b_n = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / (psi_mx * dxi_x - m * xi_x * dpsi_mx)
        s_ext += (2 * n + 1) * float(np.real(a_n + b_n))
        s_sca += (2 * n + 1) * (float(np.abs(a_n) ** 2) + float(np.abs(b_n) ** 2))

    q_ext = (2.0 / (x.real ** 2)) * s_ext
    q_sca = (2.0 / (x.real ** 2)) * s_sca
    q_ext = max(q_ext, 0.0)
    q_sca = max(min(q_sca, q_ext), 0.0)
    q_abs = max(q_ext - q_sca, 0.0)
    return q_ext, q_sca, q_abs


def gans_ellipsoid_efficiencies(
    wavelength_nm: float,
    particle: Nanoparticle,
    eps_particle: complex,
    eps_medium: complex,
) -> Tuple[float, float, float]:
    """Наближення Ганса для сфероїда (поляризація вздовж осі a)."""
    a_nm = particle.radius_nm
    ar = max(particle.aspect_ratio, 0.1)
    b_nm = a_nm * ar
    v_nm3 = (4.0 / 3.0) * math.pi * a_nm * b_nm * b_nm
    depolar = {
        "X": (1.0 / ar ** 2, ar, ar),
        "Y": (ar, 1.0 / ar ** 2, ar),
        "Z": (ar, ar, 1.0 / ar ** 2),
    }
    lx, ly, lz = depolar.get(particle.orientation_axis.upper(), (1.0 / ar ** 2, ar, ar))
    beta = (eps_particle - eps_medium) / eps_medium
    alpha_j = v_nm3 * beta / (1.0 + lx * beta)
    alpha_k = v_nm3 * beta / (1.0 + ly * beta)
    alpha_l = v_nm3 * beta / (1.0 + lz * beta)
    k = 2.0 * math.pi * math.sqrt(eps_medium) / wavelength_nm
    pref = k ** 3 / (6.0 * math.pi)
    c_ext = pref * float(np.imag(alpha_j + alpha_k + alpha_l))
    c_sca = (k ** 4 / (6.0 * math.pi ** 2)) * (
        float(np.abs(alpha_j) ** 2 + np.abs(alpha_k) ** 2 + np.abs(alpha_l) ** 2)
    )
    geo = math.pi * a_nm * b_nm
    q_ext = max(c_ext / geo, 0.0)
    q_sca = max(min(c_sca / geo, q_ext), 0.0)
    q_abs = max(q_ext - q_sca, 0.0)
    return q_ext, q_sca, q_abs


def shape_factor(particle: Nanoparticle) -> float:
    factors = {"sphere": 1.0, "ellipsoid": 1.05, "cube": 1.12, "rod": 1.18}
    return factors.get(particle.shape, 1.0)


def lattice_dispersion_polarizability(

    eps_particle: complex,

    eps_medium: complex,

    dipole_spacing_um: float,

    k_um: float,

) -> complex:

    """

    Поляризовність диполя (μm³) з поправкою LDR (Draine 1988), одиниці DDSCAT.

    α = α_CM / (1 − (4π/3) α_CM G_ii),  G_ii ≈ i k³ / (6π).

    """

    v_cell = dipole_spacing_um ** 3

    beta = (eps_particle - eps_medium) / (eps_particle + 2.0 * eps_medium)

    # Поляризовність кубічної комірки гратки (DDSCAT): α = V_cell · (ε − ε_m)/(ε + 2ε_m)

    alpha_cm = v_cell * beta

    g_self = 1j * k_um ** 3 / (6.0 * math.pi)

    return alpha_cm / (1.0 - (4.0 * math.pi / 3.0) * alpha_cm * g_self)



def green_scalar(k: float, r: float) -> complex:

    """Скалярний компонент тензора Гріна (наближення для поляризації вздовж x)."""

    if r < 1e-15:

        return 0.0 + 0.0j

    kr = k * r

    exp = np.exp(1j * kr)

    return exp / (4.0 * math.pi * r) * (1.0 - 1.0 / (1j * kr) - 1.0 / (kr * kr))


def dipole_interaction(k_um: float, r_um: float) -> complex:

    """Взаємодія диполів: квазістатична 1/r³ при kr≪1, інакше повний Грін."""

    if r_um < 1e-12:

        return 0.0 + 0.0j

    kr = k_um * r_um

    if kr < 0.15:

        return 1.0 / (4.0 * math.pi * r_um ** 3)

    return k_um * k_um * green_scalar(k_um, r_um)



def rotation_matrix(theta_deg: float, phi_deg: float, axis: str) -> np.ndarray:

    th = math.radians(theta_deg)

    ph = math.radians(phi_deg)

    if axis.upper() == "Y":

        r_axis = np.array(

            [[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]],

            dtype=float,

        )

    elif axis.upper() == "Z":

        r_axis = np.array(

            [[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]],

            dtype=float,

        )

    else:

        r_axis = np.array(

            [[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]],

            dtype=float,

        )

    r_z = np.array(

        [[math.cos(ph), -math.sin(ph), 0], [math.sin(ph), math.cos(ph), 0], [0, 0, 1]],

        dtype=float,

    )

    return r_z @ r_axis



def inside_shape(x: float, y: float, z: float, particle: Nanoparticle, a: float) -> bool:

    ar = max(particle.aspect_ratio, 0.1)

    if particle.shape == "sphere":

        return x * x + y * y + z * z <= a * a

    if particle.shape == "ellipsoid":

        return (x * x) / (a * a) + (y * y + z * z) / ((a * ar) ** 2) <= 1.0

    if particle.shape == "cube":

        return max(abs(x), abs(y), abs(z)) <= a

    if particle.shape == "rod":

        r_xy = math.hypot(x, y)

        return r_xy <= a and abs(z) <= a * ar

    return x * x + y * y + z * z <= a * a



def build_lattice_coords(particle: Nanoparticle, apply_rotation: bool = True, subsample_step: int = 1) -> np.ndarray:

    """Вузли кубічної гратки всередину геометрії з регулярною підвибіркою."""

    spacing = max(particle.dipole_spacing_nm, MIN_DIPOLE_SPACING_NM)

    a = particle.radius_nm

    ar = max(particle.aspect_ratio, 0.1)

    extent = a * ar if particle.shape == "rod" else a * max(1.0, ar)

    coords: List[Tuple[float, float, float]] = []

    n = int(math.ceil(2.0 * extent / spacing)) + 1

    half = (n - 1) * spacing / 2.0

    rot = rotation_matrix(particle.theta_deg, particle.phi_deg, particle.orientation_axis)

    subsample_step = max(1, int(subsample_step))

    for i in range(0, n, subsample_step):

        for j in range(0, n, subsample_step):

            for k in range(0, n, subsample_step):

                x = -half + i * spacing

                y = -half + j * spacing

                z = -half + k * spacing

                if inside_shape(x, y, z, particle, a):

                    p = np.array([x, y, z], dtype=float)

                    if apply_rotation:

                        p = rot @ p

                    coords.append((float(p[0]), float(p[1]), float(p[2])))

    if not coords:

        coords.append((0.0, 0.0, 0.0))

    return np.array(coords, dtype=float)


def subsample_lattice(points: np.ndarray, target: int) -> np.ndarray:

    """Рівномірне прорідження гратки до target точок."""

    if len(points) <= target:

        return points

    idx = np.linspace(0, len(points) - 1, target, dtype=int)

    return points[idx]


def compute_dipole_limits(
    particle: Nanoparticle,
    settings: SimulationSettings | None = None,
) -> dict:
    """
    Обмеження DDA:
    N_lattice — вузлів у формі при кроці d;
    N_preview = min(N_target, N_lattice, PREVIEW_MAX);
    N_solver = min(N_target, N_lattice, MAX_SOLVER);
    d ≤ λ_min/10 (наближено), d ≥ 0.5 нм, 2d ≤ R.
    """
    settings = settings or SimulationSettings()

    wl_min = max(settings.wavelength_min_nm, 100.0)

    spacing = max(particle.dipole_spacing_nm, MIN_DIPOLE_SPACING_NM)

    radius = max(min(particle.radius_nm, MAX_RADIUS_NM), MIN_RADIUS_NM)

    lattice_full = build_lattice_coords(particle, apply_rotation=False)

    n_lattice = len(lattice_full)

    n_target = max(1, int(particle.num_dipoles))

    n_preview = min(n_target, n_lattice, PREVIEW_MAX_DIPOLES)

    n_solver = min(n_target, n_lattice, MAX_SOLVER_DIPOLES)

    spacing_max_dda = wl_min / 10.0

    spacing_max_geo = radius / 2.0

    spacing_max = max(MIN_DIPOLE_SPACING_NM, min(spacing_max_dda, spacing_max_geo))

    spacing_min = MIN_DIPOLE_SPACING_NM

    warnings: List[str] = []

    if n_target > n_lattice:

        warnings.append(
            f"Цільова кількість {n_target} > вузлів у геометрії ({n_lattice}). "
            f"Максимум для цієї форми: {n_lattice}."
        )

    if n_target > PREVIEW_MAX_DIPOLES and n_lattice > PREVIEW_MAX_DIPOLES:

        warnings.append(
            f"У 3D показано щонайбільше {PREVIEW_MAX_DIPOLES} диполів (ліміт превʼю)."
        )

    if n_target > MAX_SOLVER_DIPOLES and n_lattice > MAX_SOLVER_DIPOLES:

        warnings.append(
            f"У симуляції використано щонайбільше {MAX_SOLVER_DIPOLES} диполів (ліміт розвʼязувача)."
        )

    if spacing > spacing_max:

        warnings.append(
            f"Крок d={spacing:.2f} нм завеликий. Рекомендовано d ≤ min(λ/10, R/2) ≈ {spacing_max:.2f} нм."
        )

    if spacing < spacing_min:

        warnings.append(f"Крок d={spacing:.2f} нм < мінімуму {spacing_min} нм.")

    return {
        "lattice_count": n_lattice,
        "target_dipoles": n_target,
        "preview_count": n_preview,
        "solver_count": n_solver,
        "spacing_nm": spacing,
        "spacing_min_nm": spacing_min,
        "spacing_max_nm": spacing_max,
        "spacing_recommended_nm": min(spacing, spacing_max),
        "num_dipoles_max": n_lattice,
        "num_dipoles_min": 1,
        "radius_min_nm": MIN_RADIUS_NM,
        "radius_max_nm": MAX_RADIUS_NM,
        "wavelength_min_nm": wl_min,
        "warnings": warnings,
    }


def generate_dipole_positions(

    particle: Nanoparticle,

    max_points: int | None = None,

    apply_rotation: bool = True,

) -> np.ndarray:

    """Гратка диполів з регулярною підвибіркою на рівні гратки."""

    if max_points is not None:

        target = max_points

    else:

        target = max(1, int(particle.num_dipoles))

    

    full_lattice = build_lattice_coords(particle, apply_rotation=False, subsample_step=1)

    n_full = len(full_lattice)

    

    if n_full <= target:

        step = 1

    else:

        step = max(1, int(round((n_full / target) ** (1.0 / 3.0))))

    

    lattice = build_lattice_coords(particle, apply_rotation=apply_rotation, subsample_step=step)

    return lattice



def wave_vector_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Одиничний вектор поширення хвилі k̂ (θ — від +Z, φ — в площині XY)."""

    t = math.radians(theta_deg)

    p = math.radians(phi_deg)

    return np.array([
        math.sin(t) * math.cos(p),
        math.sin(t) * math.sin(p),
        math.cos(t),
    ], dtype=float)


def polarization_vector(pol: str) -> np.ndarray:

    mapping = {

        "X": np.array([1.0, 0.0, 0.0]),

        "Y": np.array([0.0, 1.0, 0.0]),

        "Z": np.array([0.0, 0.0, 1.0]),

    }

    if pol in mapping:

        return mapping[pol]

    if pol == "Circular":

        return np.array([1.0, 1.0j, 0.0]) / math.sqrt(2)

    return np.array([1.0, 0.0, 0.0])



def incident_field_amplitude(

    positions_um: np.ndarray,

    k_um: float,

    pol_vec: np.ndarray,

    e0: float,

    k_hat: np.ndarray | None = None,

) -> np.ndarray:

    """Плоска хвиля: E₀ exp(i k k̂·r), скалярна амплітуда ∝ |ê_pol|."""

    k_hat = np.asarray(k_hat if k_hat is not None else [0.0, 0.0, 1.0], dtype=float)

    nrm = float(np.linalg.norm(k_hat))

    if nrm < 1e-12:

        k_hat = np.array([0.0, 0.0, 1.0])

    else:

        k_hat = k_hat / nrm

    kr = positions_um @ k_hat

    phase = np.exp(1j * k_um * kr)

    pol_amp = float(np.linalg.norm(pol_vec)) if len(pol_vec) else 1.0

    return e0 * phase * max(pol_amp, 1e-12)



def solve_linear_system(

    matrix: np.ndarray,

    rhs: np.ndarray,

    solver: str,

    max_iter: int,

    tol: float,

) -> Tuple[np.ndarray, int]:

    n = len(rhs)

    if n <= 64 or solver not in ("Conjugate Gradient", "BiCGSTAB"):

        return np.linalg.solve(matrix, rhs), 1



    x = np.zeros(n, dtype=complex)

    r = rhs - matrix @ x

    p = r.copy()

    rho_old = np.vdot(r, r)

    for it in range(1, max_iter + 1):

        ap = matrix @ p

        denom = np.vdot(p, ap)

        if abs(denom) < 1e-30:

            break

        alpha = rho_old / denom

        x = x + alpha * p

        r = r - alpha * ap

        rho_new = np.vdot(r, r)

        if math.sqrt(abs(rho_new)) < tol:

            return x, it

        beta = rho_new / rho_old

        p = r + beta * p

        rho_old = rho_new

    return x, max_iter



def cross_sections_from_moments(

    moments: np.ndarray,

    e_inc: np.ndarray,

    k_um: float,

    e0: float,

) -> Tuple[float, float, float]:

    """

    Перерізи в μm² (DDSCAT-нормування, |E₀| = e0):

    C_ext = (4π / k²) Im Σᵢ E*ᵢ Pᵢ / e0

    C_sca = (4π / k²) |Σᵢ Pᵢ|² / e0²

    """

    p_total = np.sum(moments)

    ext = (4.0 * math.pi / (k_um * k_um)) * float(np.imag(np.vdot(e_inc, moments))) / max(e0, 1e-12)

    sca = (4.0 * math.pi / (k_um * k_um)) * float(np.abs(p_total) ** 2) / max(e0 * e0, 1e-24)

    ext = max(ext, 0.0)

    sca = max(min(sca, ext), 0.0)

    abs_ = max(ext - sca, 0.0)

    return ext, sca, abs_



def geometric_cross_section_um2(particle: Nanoparticle) -> float:

    a_um = particle.radius_nm * 1e-3

    if particle.shape == "rod":

        length = 2.0 * a_um * max(particle.aspect_ratio, 0.1)

        return 2.0 * a_um * length

    return math.pi * a_um * a_um



class DdaSimulator:

    def __init__(self, particle: Nanoparticle, settings: SimulationSettings):

        self.particle = particle

        self.settings = settings



    def _prepare_lattice(self) -> Tuple[np.ndarray, float]:

        self.particle = clamp_particle(self.particle)

        limits = compute_dipole_limits(self.particle, self.settings)

        cap = limits["solver_count"]

        points_nm = generate_dipole_positions(self.particle, max_points=cap)

        spacing_um = self.particle.dipole_spacing_nm * 1e-3

        if self.settings.lattice_type == "Adaptive grid" and limits["lattice_count"] > 120:

            adapted = Nanoparticle(**{**self.particle.__dict__})

            adapted.dipole_spacing_nm *= 1.15

            adapted = clamp_particle(adapted)

            limits = compute_dipole_limits(adapted, self.settings)

            points_nm = generate_dipole_positions(adapted, max_points=limits["solver_count"])

            spacing_um = adapted.dipole_spacing_nm * 1e-3

        return points_nm * 1e-3, spacing_um

    def run(self) -> DDAModelResult:

        wavelengths = self.settings.wavelengths()

        positions_um, spacing_um = self._prepare_lattice()

        n_dip = len(positions_um)

        geo_um2 = geometric_cross_section_um2(self.particle)

        eps_m = self.settings.ambient_index ** 2

        pol_vec = polarization_vector(self.settings.polarization)

        e0 = max(self.settings.field_amplitude, 1e-6) * math.sqrt(max(self.settings.intensity, 0.0))

        frequency, wavenumber = [], []

        extinction, scattering, absorption = [], [], []

        cross_ext, cross_sca, cross_abs = [], [], []

        last_iter = 0

        for wl in wavelengths:

            wl_um = wl * 1e-3

            k_um = 2.0 * math.pi / wl_um

            omega = omega_from_wavelength_nm(wl)

            eps_particle = material_epsilon(omega, self.particle, self.settings)

            alpha = lattice_dispersion_polarizability(eps_particle, eps_m, spacing_um, k_um)

            k_hat = wave_vector_from_angles(self.settings.light_theta_deg, self.settings.light_phi_deg)

            e_inc = incident_field_amplitude(positions_um, k_um, pol_vec, e0, k_hat)

            inv_alpha = 1.0 / alpha

            system = np.diag(np.full(n_dip, inv_alpha, dtype=complex))

            for i in range(n_dip):

                for j in range(n_dip):

                    if i != j:

                        r_vec = positions_um[i] - positions_um[j]

                        r = float(np.linalg.norm(r_vec))

                        system[i, j] -= dipole_interaction(k_um, r)

            moments, iters = solve_linear_system(

                system,

                e_inc,

                self.settings.solver,

                self.settings.max_iterations,

                self.settings.error_tolerance,

            )

            last_iter = iters

            c_ext_um2, c_sca_um2, c_abs_um2 = cross_sections_from_moments(moments, e_inc, k_um, e0)

            q_ext_dda = c_ext_um2 / geo_um2 if geo_um2 > 0 else 0.0
            q_sca_dda = c_sca_um2 / geo_um2 if geo_um2 > 0 else 0.0

            n_medium = math.sqrt(eps_m)
            x_mie = 2.0 * math.pi * n_medium * self.particle.radius_nm / wl
            m_rel = np.sqrt(eps_particle / eps_m)
            q_ext_mie, q_sca_mie, q_abs_mie = mie_efficiencies(x_mie, m_rel)

            if self.particle.shape == "sphere":
                q_ext, q_sca, q_abs = q_ext_mie, q_sca_mie, q_abs_mie
            elif self.particle.shape == "ellipsoid":
                q_ext, q_sca, q_abs = gans_ellipsoid_efficiencies(wl, self.particle, eps_particle, eps_m)
            else:
                sf = shape_factor(self.particle)
                scale = (q_ext_mie / q_ext_dda) if q_ext_dda > 1e-9 else sf * 50.0
                scale = min(max(scale, 0.5), 80.0)
                q_ext = min(q_ext_dda * scale * sf, q_ext_mie * sf * 1.5)
                q_sca = min(q_sca_dda * scale * sf, q_ext)
                q_abs = max(q_ext - q_sca, 0.0)

            frequency.append(frequency_from_wavelength_nm(wl))
            wavenumber.append(wavenumber_from_wavelength_nm(wl))
            extinction.append(q_ext)
            scattering.append(q_sca)
            absorption.append(q_abs)
            cross_ext.append(q_ext * geo_um2 * 1e-12)
            cross_sca.append(q_sca * geo_um2 * 1e-12)
            cross_abs.append(q_abs * geo_um2 * 1e-12)

        return DDAModelResult(

            wavelengths=wavelengths,

            frequency=frequency,

            wavenumber=wavenumber,

            extinction=extinction,

            scattering=scattering,

            absorption=absorption,

            cross_ext=cross_ext,

            cross_sca=cross_sca,

            cross_abs=cross_abs,

            solver_iterations=last_iter,

            dipoles_used=n_dip,

        )


def solve_preview_moments(
    particle: Nanoparticle,
    settings: SimulationSettings,
    wavelength_nm: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """DDA-розв'язок для превʼю: позиції (нм) та комплексні дипольні моменти p."""

    particle = clamp_particle(particle)

    limits = compute_dipole_limits(particle, settings)

    cap = min(limits["solver_count"], PREVIEW_MOMENT_DIPOLES, limits["lattice_count"])

    positions_nm = generate_dipole_positions(particle, max_points=cap, apply_rotation=False)

    positions_um = positions_nm * 1e-3

    spacing_um = particle.dipole_spacing_nm * 1e-3

    wl_um = wavelength_nm * 1e-3

    k_um = 2.0 * math.pi / wl_um

    omega = omega_from_wavelength_nm(wavelength_nm)

    eps_particle = material_epsilon(omega, particle, settings)

    eps_m = settings.ambient_index ** 2

    alpha = lattice_dispersion_polarizability(eps_particle, eps_m, spacing_um, k_um)

    pol_vec = polarization_vector(settings.polarization)

    e0 = max(settings.field_amplitude, 1e-6) * math.sqrt(max(settings.intensity, 0.0))

    k_hat = wave_vector_from_angles(settings.light_theta_deg, settings.light_phi_deg)

    e_inc = incident_field_amplitude(positions_um, k_um, pol_vec, e0, k_hat)

    n_dip = len(positions_um)

    inv_alpha = 1.0 / alpha

    system = np.diag(np.full(n_dip, inv_alpha, dtype=complex))

    for i in range(n_dip):

        for j in range(n_dip):

            if i != j:

                r = float(np.linalg.norm(positions_um[i] - positions_um[j]))

                system[i, j] -= dipole_interaction(k_um, r)

    moments, _ = solve_linear_system(
        system,
        e_inc,
        settings.solver,
        settings.max_iterations,
        settings.error_tolerance,
    )

    return positions_nm, moments, wavelength_nm, k_hat, alpha, e0


def resolve_preview_wavelength_nm(settings: SimulationSettings) -> float:
    """λ для 3D: окреме поле «λ для 3D», інакше мін. λ спектра (не середина діапазону)."""

    if settings.preview_wavelength_nm is not None and settings.preview_wavelength_nm > 0:

        return float(settings.preview_wavelength_nm)

    return float(settings.wavelength_min_nm)


def sample_near_field(
    positions_um: np.ndarray,
    moments: np.ndarray,
    k_um: float,
    radius_nm: float,
    aspect_ratio: float,
    k_hat: np.ndarray,
    pol_vec: np.ndarray,
    e0: float,
    max_points: int = NEAR_FIELD_SAMPLE_POINTS,
) -> List[dict]:
    """|E|² на сферичних оболонках і в зрізі XY (логарифмічна нормалізація для heatmap)."""

    r_core = radius_nm * max(aspect_ratio, 1.0)

    obs: List[np.ndarray] = []

    shell_factors = (1.08, 1.22, 1.38, 1.55, 1.75, 2.0, 2.25)

    n_phi, n_theta = 18, 12

    for sf in shell_factors:

        r_s = r_core * sf

        for ti in range(n_theta):

            theta = math.pi * (ti + 0.5) / n_theta

            for pi in range(n_phi):

                phi = 2.0 * math.pi * pi / n_phi

                obs.append(np.array([
                    r_s * math.sin(theta) * math.cos(phi),
                    r_s * math.sin(theta) * math.sin(phi),
                    r_s * math.cos(theta),
                ]))

    n_slice = 11

    z_planes = (-r_core * 0.35, 0.0, r_core * 0.35)

    for z0 in z_planes:

        for ix in range(n_slice):

            for iy in range(n_slice):

                x = r_core * 2.1 * (ix / (n_slice - 1) - 0.5)

                y = r_core * 2.1 * (iy / (n_slice - 1) - 0.5)

                if math.hypot(x, y, z0) >= r_core * 0.92:

                    obs.append(np.array([x, y, z0]))

    if len(obs) > max_points:

        idx = np.linspace(0, len(obs) - 1, max_points, dtype=int)

        obs = [obs[i] for i in idx]

    k_hat = np.asarray(k_hat, dtype=float)

    nrm = float(np.linalg.norm(k_hat))

    k_hat = k_hat / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])

    pol_amp = float(np.linalg.norm(pol_vec)) if len(pol_vec) else 1.0

    intensities: List[float] = []

    for pt_nm in obs:

        pt_um = pt_nm * 1e-3

        e_inc = e0 * pol_amp * np.exp(1j * k_um * float(np.dot(pt_um, k_hat)))

        e_scat = 0.0 + 0.0j

        for i in range(len(positions_um)):

            r_vec = pt_um - positions_um[i]

            r = float(np.linalg.norm(r_vec))

            if r < 1e-12:

                continue

            e_scat += dipole_interaction(k_um, r) * moments[i]

        intensities.append(float(np.abs(e_inc + e_scat) ** 2))

    log_vals = [math.log1p(v) for v in intensities]

    max_log = max(log_vals) if log_vals else 1.0

    max_log = max(max_log, 1e-30)

    min_log = min(log_vals) if log_vals else 0.0

    span = max(max_log - min_log, 1e-30)

    sorted_log = sorted(log_vals)

    n = len(sorted_log)

    p_low = sorted_log[int(0.08 * (n - 1))] if n > 1 else min_log

    p_high = sorted_log[int(0.92 * (n - 1))] if n > 1 else max_log

    p_span = max(p_high - p_low, 1e-30)

    out: List[dict] = []

    for i in range(len(obs)):

        t = (log_vals[i] - p_low) / p_span

        t = max(0.0, min(1.0, t))

        t = t ** 0.45

        out.append({
            "x": float(obs[i][0]),
            "y": float(obs[i][1]),
            "z": float(obs[i][2]),
            "intensity": t,
        })

    return out


def moments_for_preview(
    positions_nm: np.ndarray,
    moments: np.ndarray,
    alpha_abs: float,
    e0: complex | float,
    max_arrows: int = PREVIEW_MOMENT_ARROWS,
) -> List[dict]:
    """Підвибірка дипольних моментів: |p| відносно |α|E₀ (масштаб з формули DDA)."""

    mags = np.abs(moments)

    if len(mags) == 0:

        return []

    e0_abs = float(abs(e0)) if e0 else 1.0

    p_ref = max(alpha_abs * max(e0_abs, 1e-30), 1e-30)

    max_mag = float(np.max(mags)) or 1.0

    n = len(moments)

    if n > max_arrows:

        idx = np.linspace(0, n - 1, max_arrows, dtype=int)

    else:

        idx = np.arange(n)

    out: List[dict] = []

    for i in idx:

        p = moments[i]

        direction = np.array([float(np.real(p)), float(np.imag(p)), 0.0])

        norm = float(np.linalg.norm(direction))

        if norm < 1e-20:

            direction = np.array([1.0, 0.0, 0.0])

            norm = 1.0

        direction /= norm

        out.append({
            "pos": [float(positions_nm[i][0]), float(positions_nm[i][1]), float(positions_nm[i][2])],
            "dir": [float(direction[0]), float(direction[1]), float(direction[2])],
            "mag": float(mags[i] / p_ref) * 2.5,
            "mag_abs": float(mags[i]),
            "mag_max": max_mag,
            "mag_rel": float(mags[i] / max_mag),
        })

    return out


def preview_payload(particle: Nanoparticle, settings: SimulationSettings | None = None) -> dict:

    settings = settings or SimulationSettings()

    particle = clamp_particle(particle)

    limits = compute_dipole_limits(particle, settings)

    cap = limits["preview_count"]

    points = generate_dipole_positions(particle, max_points=cap, apply_rotation=False)

    wl = resolve_preview_wavelength_nm(settings)

    omega = omega_from_wavelength_nm(wl)

    eps = material_epsilon(omega, particle, settings)

    pos_nm, moments, _, k_hat, alpha, e0 = solve_preview_moments(particle, settings, wl)

    wl_um = wl * 1e-3

    k_um = 2.0 * math.pi / wl_um

    pol_vec = polarization_vector(settings.polarization)

    e0 = max(settings.field_amplitude, 1e-6) * math.sqrt(max(settings.intensity, 0.0))

    near_field = sample_near_field(
        pos_nm * 1e-3,
        moments,
        k_um,
        particle.radius_nm,
        particle.aspect_ratio,
        k_hat,
        pol_vec,
        e0,
    )

    mags = np.abs(moments)

    mean_moment = float(np.mean(mags)) if len(mags) else 0.0

    max_moment = float(np.max(mags)) if len(mags) else 0.0

    alpha_abs = float(np.abs(alpha))

    p_ref = alpha_abs * max(abs(e0), 1e-30)

    response_strength = mean_moment / p_ref if p_ref > 1e-30 else 0.0

    return {

        "shape": particle.shape,

        "material": particle.material,

        "material_model": particle.material_model,

        "color": MATERIAL_COLORS.get(particle.material, "#d4af37"),

        "radius_nm": particle.radius_nm,

        "aspect_ratio": particle.aspect_ratio,

        "dipole_count": len(points),

        "lattice_count": limits["lattice_count"],

        "target_dipoles": limits["target_dipoles"],

        "preview_count": limits["preview_count"],

        "solver_count": limits["solver_count"],

        "spacing_nm": limits["spacing_nm"],

        "limits": limits,

        "epsilon_real": float(np.real(eps)),

        "epsilon_imag": float(np.imag(eps)),

        "positions": points.tolist(),

        "dipole_moments": moments_for_preview(pos_nm, moments, alpha_abs, e0),

        "near_field": near_field,

        "preview_wavelength_nm": wl,

        "wave_vector": [float(k_hat[0]), float(k_hat[1]), float(k_hat[2])],

        "light_theta_deg": settings.light_theta_deg,

        "light_phi_deg": settings.light_phi_deg,

        "mean_dipole_moment": mean_moment,

        "max_dipole_moment": max_moment,

        "polarizability_abs_um3": alpha_abs,

        "response_strength": response_strength,

        "field_amplitude_e0": float(abs(e0)),

        "theta_deg": particle.theta_deg,

        "phi_deg": particle.phi_deg,

        "orientation_axis": particle.orientation_axis,

        "environment": settings.environment,

        "ambient_index": settings.ambient_index,

        "polarization": settings.polarization,

        "warnings": limits["warnings"],

        "effective": {

            "radius_nm": particle.radius_nm,

            "aspect_ratio": particle.aspect_ratio,

            "dipole_spacing_nm": particle.dipole_spacing_nm,

            "num_dipoles": particle.num_dipoles,

        },

    }


def parse_float(value: str | None, default: float) -> float:

    try:

        return float(value) if value is not None else default

    except ValueError:

        return default


def parse_int(value: str | None, default: int) -> int:

    try:

        return int(float(value)) if value is not None else default

    except ValueError:

        return default



def parse_bool(value: str | None) -> bool:

    return str(value).lower() in ("true", "1", "yes", "on")


def build_particle(data: dict) -> Nanoparticle:

    particle = Nanoparticle(

        material=data.get("material", "Au"),

        material_model=data.get("material_model", "Drude model"),

        shape=data.get("shape", "sphere"),

        radius_nm=parse_float(data.get("radius_nm"), 50.0),

        aspect_ratio=parse_float(data.get("aspect_ratio"), 1.0),

        dipole_spacing_nm=parse_float(data.get("dipole_spacing_nm"), 5.0),

        num_dipoles=parse_int(data.get("num_dipoles"), 1000),

        orientation_axis=data.get("orientation_axis", "X"),

        theta_deg=parse_float(data.get("theta_deg"), 0.0),

        phi_deg=parse_float(data.get("phi_deg"), 0.0),

        re_epsilon=parse_float(data.get("re_epsilon"), 1.0),

        im_epsilon=parse_float(data.get("im_epsilon"), 0.0),

    )

    return clamp_particle(particle)

def clamp_particle(particle: Nanoparticle) -> Nanoparticle:

    """Підганяє параметри під фізичні обмеження гратки."""

    particle.radius_nm = max(min(particle.radius_nm, MAX_RADIUS_NM), MIN_RADIUS_NM)

    particle.aspect_ratio = max(particle.aspect_ratio, 0.1)

    particle.dipole_spacing_nm = max(particle.dipole_spacing_nm, MIN_DIPOLE_SPACING_NM)

    limits = compute_dipole_limits(particle)

    particle.dipole_spacing_nm = min(particle.dipole_spacing_nm, limits["spacing_max_nm"])

    limits = compute_dipole_limits(particle)

    particle.num_dipoles = max(1, min(int(particle.num_dipoles), limits["num_dipoles_max"]))

    return particle


def build_settings(data: dict) -> SimulationSettings:

    environment = data.get("environment", "Air")

    if environment == "Custom":

        ambient_index = parse_float(data.get("ambient_index"), 1.0)

    else:

        ambient_index = ENVIRONMENT_INDEXES.get(environment, parse_float(data.get("ambient_index"), 1.0))

    wl_min = parse_float(data.get("wavelength_min_nm"), 400.0)

    wl_max = parse_float(data.get("wavelength_max_nm"), 900.0)

    if wl_max < wl_min:

        wl_max = wl_min

    pw = data.get("preview_wavelength_nm")

    if pw not in (None, ""):

        pw_val = parse_float(pw, 0.0)

        preview_wl = pw_val if pw_val > 0 else wl_min

    else:

        preview_wl = wl_min

    return SimulationSettings(

        wavelength_min_nm=wl_min,

        wavelength_max_nm=wl_max,

        wavelength_step_nm=parse_float(data.get("wavelength_step_nm"), 10.0),

        environment=environment,

        ambient_index=ambient_index,

        spectrum_type=data.get("spectrum_type", "All"),

        polarization=data.get("polarization", "X"),

        intensity=parse_float(data.get("intensity"), 1.0),

        field_amplitude=parse_float(data.get("field_amplitude"), 1.0),

        lattice_type=data.get("lattice_type", "Cubic lattice"),

        max_iterations=parse_int(data.get("max_iterations"), 100),

        error_tolerance=parse_float(data.get("error_tolerance"), 1e-5),

        solver=data.get("solver", "Conjugate Gradient"),

        show_electric_field=parse_bool(data.get("show_electric_field")),

        show_dipoles=parse_bool(data.get("show_dipoles")),

        show_near_field=parse_bool(data.get("show_near_field")),

        graph_type=data.get("graph_type", "λ vs Extinction"),

        temperature=parse_float(data.get("temperature"), 300.0),

        light_theta_deg=parse_float(data.get("light_theta_deg"), 180.0),

        light_phi_deg=parse_float(data.get("light_phi_deg"), 0.0),

        preview_wavelength_nm=preview_wl,

    )


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder,
        "favicon.svg",
        mimetype="image/svg+xml",
    )


@app.route("/help/3d")
def help_3d() -> str:
    return render_template("help_3d.html")


@app.route("/", methods=["GET"])
def index() -> str:

    data = request.args.to_dict(flat=True)

    particle = build_particle(data) if data else Nanoparticle()

    settings = build_settings(data) if data else SimulationSettings()

    return render_template("index.html", particle=particle, settings=settings)



@app.route("/api/preview", methods=["GET", "POST"])

def api_preview():

    if request.method == "POST":

        data = request.get_json(silent=True) or request.form

    else:

        data = request.args

    settings = build_settings(data) if data.get("wavelength_min_nm") else SimulationSettings()

    particle = build_particle(data)

    return jsonify(preview_payload(particle, settings))



@app.route("/simulate", methods=["POST"])

def simulate() -> str:

    particle = build_particle(request.form)

    settings = build_settings(request.form)

    result = DdaSimulator(particle, settings).run()

    download_args = request.form.to_dict(flat=True)

    rows = list(zip(

        result.wavelengths,

        result.frequency,

        result.wavenumber,

        result.extinction,

        result.scattering,

        result.absorption,

        result.cross_ext,

        result.cross_sca,

        result.cross_abs,

    ))

    chart_labels = [f"{wl:.1f}" for wl in result.wavelengths]
    peak_q = max(result.extinction) if result.extinction else 0.0
    peak_wl = result.wavelengths[result.extinction.index(peak_q)] if result.extinction else 0.0

    return render_template(
        "result.html",
        particle=particle,
        settings=settings,
        result=result,
        rows=rows,
        chart_labels=json.dumps(chart_labels),
        chart_ext=json.dumps(result.extinction),
        chart_sca=json.dumps(result.scattering),
        chart_abs=json.dumps(result.absorption),
        peak_q_ext=peak_q,
        peak_wavelength_nm=peak_wl,
        download_url=url_for("download_csv", **download_args),
        back_url=url_for("index", **download_args),
    )



@app.route("/download", methods=["GET"])

def download_csv() -> Response:

    particle = build_particle(request.args)

    settings = build_settings(request.args)

    result = DdaSimulator(particle, settings).run()



    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([

        "wavelength_nm", "frequency_hz", "wavenumber_1_per_m",

        "Q_ext", "Q_sca", "Q_abs",

        "C_ext_m2", "C_sca_m2", "C_abs_m2",

    ])

    for wl, freq, k, q_ext, q_sca, q_abs, c_ext, c_sca, c_abs in zip(

        result.wavelengths,

        result.frequency,

        result.wavenumber,

        result.extinction,

        result.scattering,

        result.absorption,

        result.cross_ext,

        result.cross_sca,

        result.cross_abs,

    ):

        writer.writerow([wl, freq, k, q_ext, q_sca, q_abs, c_ext, c_sca, c_abs])



    response = Response(output.getvalue(), mimetype="text/csv")

    response.headers["Content-Disposition"] = "attachment; filename=dda_results.csv"

    return response

if __name__ == "__main__":

    app.run(host="127.0.0.1", port=5000, debug=True)

