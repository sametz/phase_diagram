#!/usr/bin/env python
# coding: utf-8

"""Modifying the isopropanol/isobutanol distillation example from *The Organic
Chem Lab Survival Manual* to create a phase diagram (.png) plus an animated .gif
for chemistry education.
"""

import math

from IPython.display import HTML
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import fsolve

"""
Color blind palette via 

https://jbloomlab.github.io/dms_struct/dms_struct.colorschemes.html

#000000  black
#E69F00  gold
#56B4E9  light blue
#009E73  green
#F0E442  yellow
#0072B2  dark blue
#D55E00  orange
#CC79A7  violet
"""

# Ideal behavior is calculated using the Clausius-Clapeyron equation, and
# the boiling points and heats of vaporization for the components of
# the binary mixture.
bp_isopropanol = 355.4  # K (82.6 °C)
bp_isobutanol = 381.2  # K (107.9 °C)
H_isopropanol = 9515.73  # cal/mol
H_isobutanol = 10039.70  # cal/mol


def p_from_t(h, t, p0, t0):
    """Calculate the vapor pressure of a liquid at a given temperature,
    using the Clausius-Clapeyron equation.
    
    The most convenient references p0 and t0 are from the boiling point of the
    compound, e.g. 760 torr and 355.4 K for isopropanol.
    
    Parameters
    ----------
    h : float
        The heat of vaporization in cal/mol
    t : float
        The temperature in K
    p0 : float
        Reference vapor pressure
    t0 : float
        Reference temperature in K
        
    Returns
    -------
    float
        The vapor pressure at temperature t.
    """
    r = 1.987204  # cal/mol·K
    e = - (h/r) * ((1/t) - (1/t0))
    return p0 * math.exp(e)


def isobutanol_pressure(t):
    """Calculate the vapor pressure (torr) of isobutanol,
    given a temperature in K.
    """
    return p_from_t(H_isobutanol, t, 760, bp_isobutanol)


def isopropanol_pressure(t):
    """Calculate the vapor pressure (torr) of isopropanol,
    given a temperature in K.
    """
    return p_from_t(H_isopropanol, t, 760, bp_isopropanol)


# From *OCLSM* p. 327:
# $$X_A = \frac{P_{atm}-P°_B}{P°_A-P°_B}$$
def chi_isopropanol_liquid(t):
    """Calculate the mol fraction of isopropanol in an isopropanol/isubutanol
    mixture, given its boiling point in K.
    """
    p_isopropanol = isopropanol_pressure(t)
    p_isobutanol = isobutanol_pressure(t)
    return (760 - p_isobutanol) / (p_isopropanol - p_isobutanol)


def chi_isobutanol_liquid(t):
    return 1.0 - chi_isopropanol_liquid(t)


# From *OCLSM p. 329:
# $$X^{vapor}_A = X^{liquid}_AP^°_A/760$$
def chi_isobutanol_vapor(t):
    """Calculate the mol fraction isobutanol in the vapor phase at a given
    temperature, given the temperature t in K.
    """
    chi_liquid = chi_isobutanol_liquid(t)
    p = isobutanol_pressure(t)
    return chi_liquid * p / 760


# create y coordinates
t_k = np.arange(bp_isopropanol, bp_isobutanol, 0.1)  # K
t_c = t_k - 273.15  # °C

# create vectorized versions of chi functions,
# then apply them to t_k to generate x coordinates 
f_liquid = np.vectorize(chi_isobutanol_liquid)
f_vapor = np.vectorize(chi_isobutanol_vapor)
chi_liquid = f_liquid(t_k)
chi_vapor = f_vapor(t_k)


# Before creating the animated gif, the basic plot is created and saved in PNG
# format. This can be used for student assignments.
fig, ax = plt.subplots()
plt.xlim(0.0, 1.0)
plt.ylim(80.0, 110.0)
liquid_curve = ax.plot(chi_liquid, t_c, color='#0072b2')[0]
vapor_curve = ax.plot(chi_vapor, t_c, color='#d55e00')[0]

ax.set(xlabel='mol fraction isobutanol', ylabel='T (°C)',
       title='Phase Diagram for Isopropanol/Isobutanol')

# Show the major grid lines with dark grey lines
ax.grid(b=True, which='major', color='#666666', linestyle='-')

# Show the minor grid lines with very faint and almost transparent grey lines
ax.minorticks_on()
ax.xaxis.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2,
              markevery=0.02)
ax.yaxis.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

x_major_spacing = 0.1
x_minor_spacing = 0.02
ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_minor_spacing))
ax.xaxis.set_major_locator(ticker.MultipleLocator(x_major_spacing))

# uncomment to view plot in notebook
# plt.show()

# When you want to save the plot, uncomment the two lines below:
fig.set_size_inches(9, 5.5)
plt.savefig('iPrOH-iBuOH-phase-diagram.png', dpi=300, orientation='landscape',
            transparent=False)


def solve_t(fn, chi):
    """Solve for t where fn(t) = chi.
    
    Parameters
    ----------
    chi : the target output value for the function
    fn : function taking one numerical argument and returnng one numerical value
    
    Returns
    -------
    [t] : [float] 
        Temperature in K. Should only have one answer. 
        TODO: consider returning t directly.
    """
    test_function = lambda t: fn(t) - chi
    # TODO consider making the first guess value an optional argument
    # using 90 °C or 363.15 K as first guess for fsolve
    return fsolve(test_function, 363.15)


def theoretical_plate(chi_initial):
    """Return the bp of a mixture in °C and its vapor composition,
    given the composition of the liquid.
    
    Parameters
    ----------
    chi_initial : float
        the mol fraction of isobutanol in the mixture
    
    Returns
    -------
    float, float
        tuple of (T in °C, mol fraction isobutanol in vapor)
    """
    t = solve_t(chi_isobutanol_liquid, chi_initial)[0]
    chi_vapor = chi_isobutanol_vapor(t)
    return t - 273.15, chi_vapor 


# Starting from an initial composition of 0.8 mol fraction isobutanol,
# calculate the t and chi data for the theoretical plates
chi0 = 0.8
t0, chi1 = theoretical_plate(chi0)
t1, chi2 = theoretical_plate(chi1)
t2, chi3 = theoretical_plate(chi2)
t3, chi4 = theoretical_plate(chi3)
t4, chi5 = theoretical_plate(chi4)
t5, chi6 = theoretical_plate(chi5)
t6, chi7 = theoretical_plate(chi6)

# line segments for theoretical plates
vertical1 = ax.plot([chi0, chi0], [0, t0],
                    color='#999999', linestyle='--')[0]
horizontal1 = ax.plot([chi0, chi1], [t0, t0],
                      color='#999999', linestyle='--')[0]
vertical2 = ax.plot([chi1, chi1], [t0, t1],
                    color='#999999', linestyle='--')[0]
horizontal2 = ax.plot([chi1, chi2], [t1, t1],
                      color='#999999', linestyle='--')[0]
vertical3 = ax.plot([chi2, chi2], [t1, t2],
                    color='#999999', linestyle='--')[0]
horizontal3 = ax.plot([chi2, chi3], [t2, t2],
                      color='#999999', linestyle='--')[0]
vertical4 = ax.plot([chi3, chi3], [t2, t3],
                    color='#999999', linestyle='--')[0]
horizontal4 = ax.plot([chi3, chi4], [t3, t3],
                      color='#999999', linestyle='--')[0]
vertical5 = ax.plot([chi4, chi4], [t3, t4],
                    color='#999999', linestyle='--')[0]
horizontal5 = ax.plot([chi4, chi5], [t4, t4],
                      color='#999999', linestyle='--')[0]
vertical6 = ax.plot([chi5, chi5], [t4, t5],
                    color='#999999', linestyle='--')[0]
horizontal6 = ax.plot([chi5, chi6], [t5, t5],
                      color='#999999', linestyle='--')[0]
vertical7 = ax.plot([chi6, chi6], [t5, t6],
                    color='#999999', linestyle='--')[0]

# points for bp of pure components
isopropanol_bp_point = plt.scatter(0, bp_isopropanol-273.15, 
                                   clip_on=False, color='#0072b2')
isobutanol_bp_point = plt.scatter(1, bp_isobutanol-273.15, 
                                  clip_on=False, color='#0072b2')

# points along the liquid and vapor curves that intersect with
# the line segments
l0 = plt.scatter([chi0], [t0], color='#0072b2')
v0 = plt.scatter([chi1], [t0], color='#d55e00')
l1 = plt.scatter([chi1], [t1], color='#0072b2')
v1 = plt.scatter([chi2], [t1], color='#d55e00')
l2 = plt.scatter([chi2], [t2], color='#0072b2')
v2 = plt.scatter([chi3], [t2], color='#d55e00')
l3 = plt.scatter([chi3], [t3], color='#0072b2')

# What follows are the annotations that will be added during the animation:
a_isopropanol_bp = plt.annotate(
    'boiling point of isopropanol\n(mol fraction isobutanol = 0)',
    xy=(0, bp_isopropanol-273.15),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -0),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a_isobutanol_bp = plt.annotate(
    'boiling point of isobutanol\n(mol fraction isobutanol = 1)',
    xy=(1, bp_isobutanol-273.15),
    xycoords='data',
    fontsize=12,
    xytext=(-200, -10),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a_liquid = plt.annotate(
    'boiling point of\niPrOH/iBuOH mixture',
    xy=(0.5, solve_t(chi_isobutanol_liquid, 0.5)[0]-273.15),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -30),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a_vapor = plt.annotate(
    'composition of vapor\n at boiling point T',
    xy=(chi2, t1),
    xycoords='data',
    fontsize=12,
    xytext=(-130, +20),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a1 = plt.annotate(
    f'bp of\n80 mol%\nisobutanol\n= {t0:.1f} °C',
    xy=(0.8, t0),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -30),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a2 = plt.annotate(
    f'mol% of vapor\nat {t0:.1f} °C\n= {chi1*100:.0f}%',
    xy=(chi1, t0),
    xycoords='data',
    fontsize=12,
    xytext=(-110, -10),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a3 = plt.annotate(
    f'cooling these vapors\ngives a condensate\nthat is {chi1*100:.0f}% iBuOH',
    xy=(chi1, t1),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -30),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a4 = plt.annotate(
    f'i.e. the amount of\nthe more volatile\niPrOH has increased\nfrom 20% to {(1-chi1)*100:.0f}%',
    xy=(chi1, t1),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -30),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a5 = plt.annotate(
    'a single\nvaporize/condense\ncycle is called a\n' + r'$\bf{theoretical\ plate}$',
    xy=(chi1, t1),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -30),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a6 = plt.annotate(
    '''So, if a simple distillation\n'''
    '''apparatus has an\n'''
    '''efficiency of ~ 1\n'''
    '''theoretical plate,\n'''
    '''the distillate will only be\n'''
    '''slightly enriched in iPrOH.''',
    xy=(chi1, t1),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -30),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a7 = plt.annotate(
    '''In fractional distillation,\n'''
    '''multiple evaporate/condense\n'''
    '''steps occur before the\n'''
    '''vapors reach the condenser.\n''',
    xy=(chi2, t1),
    xycoords='data',
    fontsize=12,
    xytext=(-130, +15),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a8 = plt.annotate(
    '''An efficiency of 2\n'''
    '''theoretical plates gives a\n'''
    '''distillate that is\n'''
    f'''{(1-chi2)*100:.0f}% iPrOH''',
    xy=(chi2, t2),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -50),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a9 = plt.annotate(
    '''An efficiency of 3\n'''
    '''theoretical plates gives a\n'''
    '''distillate that is\n'''
    f'''{(1-chi3)*100:.0f}% iPrOH''',
    xy=(chi3, t3),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -50),
    textcoords='offset points',
    arrowprops={
        'arrowstyle': '->'
    })

a10 = plt.annotate(
    '''The greater the number of theoretical plates,\n'''
    '''the more perfect the separation.''',
    xy=(chi4, t5),
    xycoords='data',
    fontsize=12,
    xytext=(+30, -5),
    textcoords='offset points'
    )

# The following define the frames for each step of the animation:
iproh_bp_frame = [isopropanol_bp_point] + [a_isopropanol_bp]
ibuoh_bp_frame = iproh_bp_frame[:-1] + [isobutanol_bp_point] + [a_isobutanol_bp]
frame_1 = ibuoh_bp_frame[:-1] + [liquid_curve] + [a_liquid]
frame_2 = frame_1[:-1] + [vapor_curve] + [a_vapor]
frame_3 = frame_2[:-1] + [vertical1, l0, a1]
frame_4 = frame_3[:-1] + [horizontal1, v0, a2]
frame_5 = frame_4[:-1] + [vertical2, l1, a3]
frame_5_2 = frame_5[:-1] + [a4]
frame_5_3 = frame_5_2[:-1] + [a5]
frame_5_4 = frame_5_3[:-1] + [a6]
frame_6 = frame_5_3[:-1] + [horizontal2, v1] + [a7]
frame_7 = frame_6[:-1] + [vertical3, l2] + [a8]
frame_8 = frame_7[:-1] + [horizontal3, v2]
frame_9 = frame_8 + [vertical4, l3] + [a9]
frame_10 = frame_9[:-1] + [horizontal4] + [a10]
frame_11 = frame_10 + [vertical5]
frame_12 = frame_11 + [horizontal5]
frame_13 = frame_12 + [vertical6]
frame_14 = frame_13 + [horizontal6]
frame_15 = frame_14 + [vertical7]
intro_frame = [liquid_curve, vapor_curve]

# The frames are put in order into the `artists` list
artists = [
           intro_frame, iproh_bp_frame, ibuoh_bp_frame, 
           frame_1, frame_2, frame_3, frame_4, 
           frame_5, frame_5_2, frame_5_3, frame_5_4,
           frame_6, frame_7, frame_8, frame_9, frame_10,
           frame_11, frame_12, frame_13, frame_14, frame_15]

# Adjust interval (ms) to suit your taste. Here: 5 s per frame
ani = ArtistAnimation(fig, artists, interval=5000)
# plt.show()

HTML(ani.to_jshtml())

# Uncomment the code below to save the animated gif
ani.save('distillation.gif', writer='imagemagick')
