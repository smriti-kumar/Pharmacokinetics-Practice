import numpy as np


"""
Parameters
"""

class UnitConversion:
    @staticmethod
    def liter_to_mL(liter):
        return liter * 1e3

    @staticmethod
    def mL_to_L(mL):
        return mL/1e3

    @staticmethod
    def g_glucose_to_mol(g_glucose):
        return (g_glucose / 180.156)

    @staticmethod
    def g_glucose_to_mmol(g_glucose):
        return (g_glucose / 180.156) * 1000

    @staticmethod
    def mmol_glcuose_to_g(mmol_glucose):
        return (mmol_glucose / 1000) * 180.156

    @staticmethod
    def concentration_mmolL_to_mgdL(mmolL):
        return mmolL*18

    @staticmethod
    def concentration_mgdl_to_mmolL(mgdL):
        return mgdL/18

    @staticmethod
    def energy_g_glucose_to_kkcal(g):
        return g*4

    @staticmethod
    def energy_kkcal_to_g_glucose_equiv(kcal):
        return kcal/4

    @staticmethod
    def convert_hour_to_min(hours):
        return hours * 60

    @staticmethod
    def convert_inverse_hour_inverse_minute(inverse_hour):
        """ 1/h to 1/min"""
        return inverse_hour / 60

    @staticmethod
    def convert_minutes_to_days(minutes):
        return minutes / 60 / 24


    @staticmethod
    def Uhr_to_uUmin(Uhr):
        return Uhr/ 60.0 * 1E6

    @staticmethod
    def Uhr_to_mUmin(Uhr):
        return Uhr*1000/60

    @staticmethod
    def U_to_mU(U):
        return U*1000

    @staticmethod
    def U_to_uU(U):
        return U * 1e6


NOMINAL_BW0: float = 100 # Initial BW [kg]

NOMINAL_RTG0: float = UnitConversion.concentration_mgdl_to_mmolL(240) # Initial renal threshold for glucose [mmol/L]

NOMINAL_GFR0: float = UnitConversion.mL_to_L(90) # Initial glomerular filtration rate [L/min]

NOMINAL_RTGMax: float = 0.64 # Maximal effect parameter for SGLT2I PD model [-]

NOMINAL_EC50: float = 32.1 # SGLT2I PD EC50 parameter [ng/mL]

NOMINAL_KSA: float = UnitConversion.convert_inverse_hour_inverse_minute(2.38241575) # SGLT2I absorption rate [1/min]

NOMINAL_KE: float = UnitConversion.convert_inverse_hour_inverse_minute(4.65463322e-02) # SGLT2I elimination rate [1/min]

NOMINAL_VD: float = 1e5 # SGLT2I volume of distribution [L]

NOMINAL_KA: float = UnitConversion.convert_inverse_hour_inverse_minute(0.906) # carbohydrate absorption rate [1/min]

NOMINAL_Q: float = UnitConversion.convert_inverse_hour_inverse_minute(26.52) # Blood flow rate between compartments [L/min]

NOMINAL_VP: float = 8.56 # Peripheral volume of distribution [L]

NOMINAL_VG: float = 9.33 # Central glucose volume of distribution [L]

NOMINAL_CLG: float = UnitConversion.convert_inverse_hour_inverse_minute(1.722) # Glucose clearance rate [L/min]

NOMINAL_CLGI: float = 0.0024 # Clearance of glucose due to insulin [(L/min) / (mIU/L)]

NOMINAL_KGE: float = UnitConversion.convert_inverse_hour_inverse_minute(0.738) # glucose elimination rate [1/min]

NOMINAL_GSS: float = 11.3 # Glucose steady state value [mmol/L]

NOMINAL_ISS: float = 13.5 # Insulin steady state [mIU/L]

NOMINAL_IPRG: float = 1.42 # Insulin secretion parameter [-]

NOMINAL_SINCR: float = 0.076 # Incretin effect parameters [L/mmol]

NOMINAL_CLI: float = UnitConversion.convert_inverse_hour_inverse_minute(63.2) # Insulin clearance [L/min]

NOMINAL_VI: float = 6.09 # Insulin volume of distribution [L]

NOMINAL_KIE: float = 0.0077 # Insulin elimination rate [1/min]

NOMINAL_ISEC0: float = NOMINAL_ISS * NOMINAL_CLI # Baseline insulin secretion rate (mIU/min)

NOMINAL_GPROD0: float = NOMINAL_GSS * (NOMINAL_CLG + NOMINAL_CLGI * NOMINAL_ISS) # Baseline glucose secretion rate (mmol/min)



def generate_parameters(n_subjects):
    
    BW = np.full(n_subjects, NOMINAL_BW0)

    GFR0 = np.full(n_subjects, NOMINAL_GFR0)

    GSS_mean = NOMINAL_GSS  # mmol/L
    ISS_mean = NOMINAL_ISS  # mIU/L
    RTG0_mean = NOMINAL_RTG0  # mmol/L
    RTGmax_mean = NOMINAL_RTGMax
    EC50_mean = NOMINAL_EC50  # ng/mL

    # TODO: Update Time
    ksa_mean = NOMINAL_KSA
    ke_mean = NOMINAL_KE
    Vd_mean = NOMINAL_VD

    ka_mean = NOMINAL_KA
    Q_mean = NOMINAL_Q
    Vp_mean = NOMINAL_VP
    Vg_mean = NOMINAL_VG
    ClG_mean = NOMINAL_CLG
    ClGI_mean = NOMINAL_CLGI
    kGE_mean = NOMINAL_KGE
    IPRG_mean = NOMINAL_IPRG
    Sincr_mean = NOMINAL_SINCR
    ClI_mean = NOMINAL_CLI
    Vi_mean = NOMINAL_VI
    kIE_mean = NOMINAL_KIE

    ### RSE
    GSS_IIV_percent = 0.14
    ISS_IIV_percent = 0.49

    ka_IIV_percent = 0.19
    Vg_IIV_percent = 0.30
    Vp_IIV_percent = 0.30
    ClG_IIV_percent = 0.59
    ClGI_IIV_percent = 0.48
    Q_IIV_percent = 0.85
    kGE_IIV_percent = 0.53
    Sincr_IIV_percent = 0.35

    VI_IIV_percent = 0.41
    ClI_IIV_percent = 0.29
    kIE_IIV_percent = 0.45
    IPRG_IIV_percent = 0.35

    """ Generate parameter values """

    factor = 2

    RTG0 = np.full(BW.size, RTG0_mean)
    RTG_Max = np.full(BW.size, RTGmax_mean)
    EC50 = np.full(BW.size, EC50_mean)
    ksa = np.full(BW.size, ksa_mean)
    ke = np.full(BW.size, ke_mean)
    Vd = np.full(BW.size, Vd_mean)

    Vg = np.full(BW.size, Vg_mean)
    Q = np.full(BW.size, Q_mean)
    Vi = np.full(BW.size, Vi_mean)

    ka = np.random.normal(loc=ka_mean, scale=ka_mean * ka_IIV_percent / factor, size=BW.size)
    Vp = np.random.normal(loc=Vp_mean, scale=Vp_mean * Vp_IIV_percent / factor, size=BW.size)
    ClG = np.random.normal(loc=ClG_mean, scale=ClG_mean * ClG_IIV_percent / factor, size=BW.size)
    ClGI = np.random.normal(loc=ClGI_mean, scale=ClGI_mean * ClGI_IIV_percent / factor, size=BW.size)
    kGE = np.random.normal(loc=kGE_mean, scale=kGE_mean * kGE_IIV_percent / factor, size=BW.size)
    IPRG = np.random.normal(loc=IPRG_mean, scale=IPRG_mean * IPRG_IIV_percent / factor, size=BW.size)
    Sincr = np.random.normal(loc=Sincr_mean, scale=Sincr_mean * Sincr_IIV_percent / factor, size=BW.size)
    ClI = np.random.normal(loc=ClI_mean, scale=ClI_mean * ClI_IIV_percent / factor, size=BW.size)
    kIE = np.random.normal(loc=kIE_mean, scale=kIE_mean * kIE_IIV_percent / factor, size=BW.size)

    GSS = np.random.normal(loc=GSS_mean, scale=GSS_mean * GSS_IIV_percent / factor, size=BW.size)
    ISS = np.random.normal(loc=ISS_mean, scale=ISS_mean * ISS_IIV_percent / factor, size=BW.size)

    Isec0 = ISS * ClI

    Gprod0 = GSS * (ClG + ClGI * ISS)

    parameter_array = np.column_stack((BW, \
                                        RTG0, \
                                        GFR0, \
                                        RTG_Max, \
                                        EC50, \
                                        ksa, \
                                        ke, \
                                        Vd, \
                                        ka, \
                                        Gprod0, \
                                        Q, \
                                        Vp, \
                                        Vg, \
                                        ClG, \
                                        ClGI, \
                                        kGE, \
                                        GSS, \
                                        ISS, \
                                        IPRG, \
                                        Sincr, \
                                        Isec0, \
                                        ClI, \
                                        Vi, \
                                        kIE))

    return parameter_array



"""
Initial Cnditions
"""


def generate_initial_conditions(parameters: np.ndarray) -> np.ndarray:

    """
    This calculates the steady state values for each state variable
    """
    BW0, RTG0, GFR0, RTG_Max, EC50, ksa, ke, Vd, ka, Gprod0, Q, Vp, Vg, ClG, ClGI, kGE, GSS, ISS, IPRG, Sincr, Isec0, ClI, Vi, kIE = parameters.T

    N = parameters.shape[0]

    Ga0 = np.zeros(N)
    Gt0 = np.zeros(N)
    Gc0 = GSS * Vg
    Gp0 = GSS * Vp
    GE0 = GSS
    I0  = ISS * Vi
    IE0 = ISS

    return np.column_stack((Ga0, Gt0, Gc0, Gp0, GE0, I0, IE0))


"""
Mathematical Model
"""
def model(states, time, parameters, inputs) -> np.ndarray:
    
    #### States 
    Ga, Gt, Gc, Gp, GE, I, IE = states.T
    
    #### Parameters
    BW0, RTG0, GFR0, RTG_Max, EC50, ksa, ke, Vd, ka, Gprod0, Q, Vp, Vg, ClG, ClGI, kGE, GSS, ISS, IPRG, Sincr, Isec0, ClI, Vi, kIE = parameters.T
    
    #### Inputs
    # Carb inputs are in grams / minute
    uCarb = inputs.T
    # Carb inputs converted to mmol / min
    uCarb *= 1000 / 180

    """ Renal Equations """
    RC = RTG0*GFR0
    
    GluFR = GFR0 * (Gc/Vg)
        
    Rreab = np.minimum(GluFR, RC)
    
    RUGE = GluFR - Rreab
    
    
    """ Glucose Equations """
    # Ga
    dGadt = uCarb - ka*Ga
    
    # Gt
    dGtdt = ka * (Ga - Gt)

    # Central compartment
    # This is in mmol/min so the Gc solution output is in mmol. This should be divided by the Vg parameter (L) and then multipled by 18 to be in mg / dL
    # or use the UnitConversion class method concentration_mmolL_to_mgdL
    dGcdt = 0.78*ka*Gt + Gprod0 + (Q/Vp)*Gp - (ClG + ClGI*IE + Q) * Gc / Vg - RUGE
    
    # Peripheral compartment
    dGpdt = Q * (Gc/Vg - Gp/Vp)
    
    # Effect 1
    dGEdt = kGE*Gc/Vg - kGE*GE
                
    GCM   = (GE / GSS)**IPRG
        
    
    """ Insulin Equations """
    # Secretion
    Iabsg = 1 + Sincr*Gt
    
    Isec = Isec0 * GCM * Iabsg
    
    # Insulin
    dIdt = Isec - (ClI/Vi)*I
    
    # Effect
    dIEdt = kIE * (I / Vi  - IE)
            
    return np.column_stack((dGadt, dGtdt, dGcdt, dGpdt, dGEdt, dIdt, dIEdt))


"""
Generate Inputs
"""

def generate_inputs(time, time_step, meal_times, meal_amounts):

    assert all(np.isin(meal_times, time)), "All meal_times must be in time"
    assert np.isclose(time[1] - time[0], time_step), "Time step doesn't seem correct"

    inputs = np.zeros_like(time)

    inputs[meal_times] = meal_amounts / time_step

    return inputs