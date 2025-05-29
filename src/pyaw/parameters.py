"""Physical parameters and calculations for plasma physics and Alfvén waves.

This module contains physical constants and functions for calculating various
plasma physics parameters, particularly those related to Alfvén waves.

Constants:
    - Vacuum permeability/permittivity
    - Light speed
    - Boltzmann constant
    - Elementary charge
    - Atomic masses
    - Particle masses

Functions:
    - Plasma density calculations
    - Alfvén wave parameters
    - Plasma frequencies
    - Gyroradius calculations
"""

import math
from math import pi

import numpy as np

# Physical constants (with unified unit) (SI) (refer to wiki.com)
VACUUM_PERMEABILITY = 4 * pi * 1e-7  # (H/m SI)  $\mu_0$
VACUUM_PERMITTIVITY = 8.8541878188e-12  # (F/m SI) $\epsilon_0$
LIGHT_SPEED = 299792458  # (m/s SI) c

BOLTZMANN_CONSTANT = 1.380649e-23  # (J/K SI) $k_B$

ELEMENTARY_CHARGE = 1.602176634e-19  # (C SI) e

# dalton (unifed atom mass unit). symbol is Da or u.
# 1 Da or u is equal to:
DA = 1.66053906892e-27  # (kg)

ELECTRON_MASS = 9.1093837139e-31  # (kg) m_e
PROTON_MASS = 1.67262192595e-27  # (kg) m_p

HYDROGEN_RELATIVE_ATOMIC_MASS = 1.00784  # (u)
HELIUM_RELATIVE_ATOMIC_MASS = 4.002602  # (u)
OXYGEN_RELATIVE_ATOMIC_MASS = 15.999  # (u)


def calculate_atomic_mass(relative_atomic_mass, one_dalton=DA):
    return one_dalton * relative_atomic_mass


HYDROGEN_ATOMIC_MASS = calculate_atomic_mass(
    HYDROGEN_RELATIVE_ATOMIC_MASS, one_dalton=DA
)
HELIUM_ATOMIC_MASS = calculate_atomic_mass(HELIUM_RELATIVE_ATOMIC_MASS, one_dalton=DA)
OXYGEN_ATOMIC_MASS = calculate_atomic_mass(OXYGEN_RELATIVE_ATOMIC_MASS, one_dalton=DA)


def calculate_plasma_density(
    hydrogen_ion_number_density,
    helium_ion_number_density,
    oxygen_ion_number_density,
    hydrogen_ion_mass=HYDROGEN_ATOMIC_MASS,
    helium_ion_mass=HELIUM_ATOMIC_MASS,
    oxygen_ion_mass=OXYGEN_ATOMIC_MASS,
):
    """calculate plasma density. ions types: hydrogen, helium, oxygen. don't consider electrons"""
    return (
        hydrogen_ion_number_density * hydrogen_ion_mass
        + helium_ion_number_density * helium_ion_mass
        + oxygen_ion_number_density * oxygen_ion_mass
    )


class Alfven:
    """Class for calculating Alfvén wave related parameters.

    Contains methods for calculating Alfvén velocities, impedance, reflection
    coefficients and boundaries for dynamic/static regimes.

    Attributes:
        vacuum_permeability: Permeability of vacuum in H/m
    """

    vacuum_permeability = VACUUM_PERMEABILITY

    def calculate_alfven_velocity(self, background_magnetic_field, plasma_density):
        """calculate local alfven velocity"""
        return background_magnetic_field / np.sqrt(
            self.vacuum_permeability * plasma_density
        )

    def calculate_alfven_impedance(self, alfven_velocity):
        return self.vacuum_permeability * alfven_velocity

    def calculate_alfven_admittance(self, alfven_impedance):
        return 1 / alfven_impedance

    def calculate_ionospheric_reflection_coefficient(
        self, alfven_admittance, pedersen_impedance
    ):
        return (alfven_admittance - pedersen_impedance) / (
            alfven_admittance + pedersen_impedance
        )

    def calculate_electric_magnetic_field_phase_difference_range(
        self, ionospheric_reflection_coefficient
    ):
        result_same = np.degrees(
            math.atan(
                (2 * ionospheric_reflection_coefficient)
                / (1 - ionospheric_reflection_coefficient**2)
            )
        )
        return -result_same, result_same

    def calculate_upper_boundary(self, alfven_velocity, pedersen_conductance):
        return self.vacuum_permeability * alfven_velocity**2 * pedersen_conductance

    def calculate_lower_boundary(self, pedersen_conductance):
        return 1 / (self.vacuum_permeability * pedersen_conductance)

    def general_dynamic_static_boundary(self):
        """refer to miles 2018: Alfvénic Dynamics and Fine Structuring of Discrete Auroral Arcs: Swarm and e-POP Observations"""
        general_dynamic_alfven_velocity = 1.4e6
        general_dynamic_pedersen_conductance = 3.0
        general_dynamic_lower_boundary = self.calculate_lower_boundary(
            general_dynamic_pedersen_conductance
        )
        general_dynamic_upper_boundary = self.calculate_upper_boundary(
            general_dynamic_alfven_velocity, general_dynamic_pedersen_conductance
        )

        general_static_alfven_velocity = 1.3e6
        general_static_pedersen_conductance = 0.5
        general_static_lower_boundary = self.calculate_lower_boundary(
            general_static_pedersen_conductance
        )
        general_static_upper_boundary = self.calculate_upper_boundary(
            general_static_alfven_velocity, general_static_pedersen_conductance
        )

        return (
            general_dynamic_lower_boundary,
            general_dynamic_upper_boundary,
            general_static_lower_boundary,
            general_static_upper_boundary,
        )


def calculate_electron_plasma_frequency(
    electron_number_density,
    elementary_charge=ELEMENTARY_CHARGE,
    vacuum_permittivity=VACUUM_PERMITTIVITY,
    electron_mass=ELECTRON_MASS,
):
    """rad/s"""
    return np.sqrt(
        (electron_number_density * elementary_charge**2)
        / (vacuum_permittivity * electron_mass)
    )


def calculate_electron_inertial_length(
    electron_plasma_frequency, light_speed=LIGHT_SPEED
):
    """

    Args:
        electron_plasma_frequency: rad/s
        light_speed:

    Returns:

    """
    return light_speed / electron_plasma_frequency


def calculate_ion_gyrofrequency(
    background_magnetic_field, ion_mass, ion_charge=ELEMENTARY_CHARGE
):
    """ion_charge is how many elementary_charge"""
    return (np.abs(ion_charge) * background_magnetic_field) / ion_mass


def calculate_ion_thermal_gyroradius(
    ion_temperature, ion_mass, ion_gyrofrequency, boltzmann_constant=BOLTZMANN_CONSTANT
):
    """one type of ion dominates"""
    return (
        np.sqrt((boltzmann_constant * ion_temperature) / ion_mass) / ion_gyrofrequency
    )


def calculate_ion_acoustic_gyroradius(
    electron_temperature,
    ion_mass,
    ion_gyrofrequency,
    boltzmann_constant=BOLTZMANN_CONSTANT,
):
    """one type of ion dominates"""
    return (
        np.sqrt((boltzmann_constant * electron_temperature) / ion_mass)
        / ion_gyrofrequency
    )


def calculate_inertial_alfven_wave_electric_magnetic_field_ratio(
    alfven_velocity,
    perpendicular_wavenumber,
    electron_inertial_length,
    ion_thermal_gyroradius=None,
):
    """
    calculate inertial alfven wave electric and disturb magnetic field ratio.
    Args:
        alfven_velocity:
        perpendicular_wavenumber:
        electron_inertial_length:
        ion_thermal_gyroradius: whether to consider ion thermal gyroradius effect or not. None means not consider.

    Returns:

    """
    if ion_thermal_gyroradius is not None:
        return alfven_velocity * np.sqrt(
            (1 + perpendicular_wavenumber**2 * electron_inertial_length**2)
            * (1 + perpendicular_wavenumber**2 * ion_thermal_gyroradius**2)
        )
    return alfven_velocity * np.sqrt(
        1 + perpendicular_wavenumber**2 * electron_inertial_length**2
    )


def calculate_kinetic_alfven_wave_electric_magnetic_field_ratio(
    alfven_velocity,
    perpendicular_wavenumber,
    ion_acoustic_gyroradius,
    ion_thermal_gyroradius=None,
):
    if ion_thermal_gyroradius is not None:
        return (
            alfven_velocity
            * (1 + perpendicular_wavenumber**2 * ion_thermal_gyroradius**2)
        ) / np.sqrt(
            1
            + perpendicular_wavenumber**2
            * (ion_acoustic_gyroradius**2 + ion_thermal_gyroradius**2)
        )
    return alfven_velocity / np.sqrt(
        1 + perpendicular_wavenumber**2 * ion_acoustic_gyroradius**2
    )


def calculate_plasma_thermal_pressure(
    electron_number_density,
    electron_temperature,
    ion_number_density,
    ion_temperature,
    boltzmann_constant=BOLTZMANN_CONSTANT,
):
    return (
        electron_number_density * boltzmann_constant * electron_temperature
        + ion_number_density * boltzmann_constant * ion_temperature
    )


def calculate_magnetic_pressure(
    magnetic_field, vacuum_permeability=VACUUM_PERMEABILITY
):
    return magnetic_field**2 / (2 * vacuum_permeability)


def calculate_plasma_beta(plasma_thermal_pressure, magnetic_pressure):
    return plasma_thermal_pressure / magnetic_pressure


def calculate_approx_perpendicular_wavenumber(wave_frequency, spacecraft_speed):
    """Taylor's Hypothesis approximation"""
    return 2 * pi * wave_frequency / spacecraft_speed
