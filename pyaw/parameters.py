class PhysicalParameters:
    # unified unit

    mu0 = 1.256637061e-6  # (H/m SI) vacuum permeability
    epsilon0 = 8.8541878188e-12  # (F/m SI) vacuum permittivity
    kB = 1.380649e-23  # (J/K SI) Boltzmann constant
    e = 1.602176634e-19  # (C SI) elementary charge: https://en.wikipedia.org/wiki/Electric_charge#Unit

    me = 9.11e-31  # Electron mass in kg
    mp = 1.673e-27  # Proton mass in kg
    rH = 1.0080  # Hydrogen relative atomic mass
    rHE = 4.0026  # Helium relative atomic mass
    rO = 15.999  # Oxygen relative atomic mass
    mH = mp * rH
    mHe = mp * rHE
    mO = mp * rO
    # refer to
    # MU0: https://en.wikipedia.org/wiki/Permeability_(electromagnetism)
    # EPSILON0: https://en.wikipedia.org/wiki/Vacuum_permittivity
    # K_B: https://en.wikipedia.org/wiki/Boltzmann_constant#
    # M_E: https://en.wikipedia.org/wiki/Orders_of_magnitude_(mass)#The_least_massive_things:_below_10%E2%88%9224_kg
    # M_P: https://en.wikipedia.org/wiki/Orders_of_magnitude_(mass)#The_least_massive_things:_below_10%E2%88%9224_kg
    # R_O: relative atomic mass or standard atomic weight: https://en.wikipedia.org/wiki/List_of_chemical_elements


class AlfvenWaveParameters:

    general_dynamic_va = 1.4e6  # General dynamic Alfven speed
    general_dynamic_Sigma_P = 3.0  # General dynamic Sigma_P

    general_static_va = 1.3e6  # General static Alfven speed
    general_static_Sigma_P = 0.5  # General static Sigma_P

    def __init__(self):
        self.general_dynamic_lower_bound = self.calculate_lower_bound(self.general_dynamic_Sigma_P)
        self.general_dynamic_upper_bound = self.calculate_upper_bound(self.general_dynamic_va,self.general_dynamic_Sigma_P)

        self.general_static_lower_bound = self.calculate_lower_bound(self.general_static_Sigma_P)
        self.general_static_upper_bound = self.calculate_upper_bound(self.general_static_va,self.general_static_Sigma_P)

    def calculate_lower_bound(self,Sigma_P):
        mu0 = PhysicalParameters.mu0
        return 1 / (mu0 * Sigma_P)

    def calculate_upper_bound(self,va,Sigma_P):
        mu0 = PhysicalParameters.mu0
        return mu0 * va ** 2 * Sigma_P
