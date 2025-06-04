"""
Created on Fri Mar 8 17:04:19 2019
Import datasets
@author: Stamatis
"""

""" parameters """
if need_to_update: crit_load = [0.8, 0.6, 0.4]
if need_to_update: number_hospitals, number_schools, number_homes = 2, 5, 300
if need_to_update: solar_panels, wind_turbines = 6000, 10
solar_efficiency, solar_cell_area, solar_cells_per_panel = 0.16, 0.0232258, 72 # solar_area is the area of the solar cell
wind_efficiency, wind_diameter, cut_in_speed, cut_out_speed = 0.48, 44, 3, 22
wind_area = math.pi * (wind_diameter / 2) ** 2 # wind_area is the rotor swept area
# using solar_efficiency = 0.16, solar_cell_area = 0.0232258, solar_cells_per_panel = 72, then 20 solar panels output approximately 1kW
# using wind_efficiency = 0.48, wind_diameter = 44, then 1 wind turbine outputs approximately 37 kW

""" read load datasets """
data_hospital, data_school, data_res = pd.read_csv('data_load_hospital.csv'), pd.read_csv('data_load_school.csv'), pd.read_csv('data_load_residential.csv')
data_np_hospital, data_np_school, data_np_res = data_hospital['Electricity:Facility [kW](Hourly)'].values, data_school['Electricity:Facility [kW](Hourly)'].values, data_res['Electricity:Facility [kW](Hourly)'].values
dem_hospital, dem_school, dem_res = data_np_hospital, data_np_school, data_np_res
for i in range(years_in_period):
    dem_hospital, dem_school, dem_res = np.append(dem_hospital, data_np_hospital), np.append(dem_school, data_np_school), np.append(dem_res, data_np_res)
dem = list()
for i in range(decision_periods):
    dem_hospital_orig, dem_school_orig, dem_res_orig = [(1 + years_in_period / 100) ** i * crit_load[0] * number_hospitals * a for a in dem_hospital], [(1 + years_in_period / 100) ** i * crit_load[1] * number_schools * b for b in dem_school], [(1 + years_in_period / 100) ** i * crit_load[2] * number_homes * c for c in dem_res] 
    dem.append([dem_hospital_orig, dem_school_orig, dem_res_orig])

""" read meteorological datasets """
data_meteo = pd.read_csv('data_meteorological.csv')
solar_ghi_orig, wind_speed_orig = data_meteo['GHI'].values, data_meteo['Wind Speed'].values
solar_ghi, wind_speed = solar_ghi_orig, wind_speed_orig
for i in range(years_in_period):
    solar_ghi, wind_speed = np.append(solar_ghi, solar_ghi_orig), np.append(wind_speed, wind_speed_orig)
    
""" calculate pv production """
prod_solar = [0.001 * solar_efficiency * solar_cell_area * solar_cells_per_panel * solar_panels * a for a in solar_ghi]

""" calculate wind production """
prod_wind = [0.001 * 0.5 * 1.25 * wind_efficiency * wind_area * wind_turbines * a ** 3 if a >= cut_in_speed and a <= cut_out_speed else 0 for a in wind_speed]

""" calculate total renewables production """
prod = [a + b for a, b in zip(prod_solar, prod_wind)]