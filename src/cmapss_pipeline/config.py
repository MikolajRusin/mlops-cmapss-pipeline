COLUMNS = [
    'engine',
    'cycle',
    'setting_1',
    'setting_2',
    'setting_3',
    'Fan_Inlet_Temperature',
    'LPC_Outlet_Temperature',
    'HPC_Outlet_Temperature',
    'LPT_Outlet_Temperature',
    'Fan_Inlet_Pressure',
    'Bypass_Duct_Pressure',
    'HPC_Outlet_Pressure',
    'Physical_Fan_Speed',
    'Physical_Core_Speed',
    'Engine_Pressure_Ratio',
    'HPC_Outlet_Static_Pressure',
    'Ratio_of_Fuel_Flow_to_Ps30',
    'Corrected_Fan_Speed',
    'Corrected_Core_Speed',
    'Bypass_Ratio',
    'Burner_Fuel-Air_Ratio',
    'Bleed_Enthalpy',
    'Required_Fan_Speed',
    'Required_Fan_Conversion_Speed',
    'High-Pressure_Turbines_Cool_Air_Flow',
    'Low-Pressure_Turbines_Cool_Air_Flow',
]

# Zero-variance sensors in FD001
CONST_COLS = [
    'setting_3',
    'Fan_Inlet_Temperature',
    'Fan_Inlet_Pressure',
    'Engine_Pressure_Ratio',
    'Burner_Fuel-Air_Ratio',
    'Required_Fan_Speed',
    'Required_Fan_Conversion_Speed',
]

FEATURES = [
    col for col in COLUMNS
    if col not in ['engine', 'cycle'] + CONST_COLS
]
