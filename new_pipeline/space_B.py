import numpy as np

mu_earth = 3.986e14
R_earth = 6.378e6
AU = 1.496e11
mu_sun = 1.327e20
RSOI_earth = AU * (mu_earth / mu_sun)**(2/5)
mu_venus = 3.249e14
r_venus = 0.723*AU
RSOI_venus = r_venus * (mu_venus / mu_sun)**(2/5)
r_earth = AU
R_venus = 6051e3
r_mars = 1.524*AU
r_jupyter = 5.204*AU
g = 9.81

v_mars = np.sqrt(mu_sun/r_mars)
v_esc = np.sqrt(2)*v_mars
v_arrival = np.sqrt(2*mu_sun/r_mars-mu_sun*2/(r_mars+r_earth))
v_a_inf = v_arrival - v_mars
print('va inf:', v_a_inf)

R_mars = 3390e3
mu_mars = 4.269*1e13

rp = R_mars+400e3
vp = np.sqrt(v_a_inf**2+2*mu_mars/rp)
v_esc = np.sqrt(2*mu_mars/rp)
delta_v = np.sqrt(v_esc**2+v_a_inf**2)-np.sqrt(mu_mars/rp)
print('delta v:', delta_v)

fuel = 3000*(1-np.exp(-delta_v/(350*g)))
print('fuel:', fuel)


"""
python3 new_pipeline/space_B.py
"""
