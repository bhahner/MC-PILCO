# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import pickle as pkl

import numpy as np
import sympy as sym


def pend(y, t, u):
    """
    System of first order equations for a pendulum system
    The policy commands the torque applied to the joint
    (stable equilibrium point with the pole down at [0,0])
    """
    theta, theta_dot = y

    m = 1.0  # mass of the pendulum
    l = 1.0  # lenght of the pendulum
    b = 0.1  # friction coefficient
    g = 9.81  # acceleration of gravity
    I = 1 / 3 * m * l**2  # moment of inertia of a pendulum around extreme point

    dydt = [theta_dot, (u - b * theta_dot - 1 / 2 * m * l * g * np.sin(theta)) / I]
    return dydt


def cartpole(y, t, u):
    """
    System of first order equations for a cart-pole system
    The policy commands the force applied to the cart
    (stable equilibrium point with the pole down at [~,0,0,0])
    """

    x, x_dot, theta, theta_dot = y

    m1 = 0.5  # mass of the cart
    m2 = 0.5  # mass of the pendulum
    l = 0.5  # length of the pendulum
    b = 0.1  # friction coefficient
    g = 9.81  # acceleration of gravity

    den = 4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2

    dydt = [
        x_dot,
        (
            2 * m2 * l * theta_dot**2 * np.sin(theta)
            + 3 * m2 * g * np.sin(theta) * np.cos(theta)
            + 4 * u
            - 4 * b * x_dot
        )
        / den,
        theta_dot,
        (
            -3 * m2 * l * theta_dot**2 * np.sin(theta) * np.cos(theta)
            - 6 * (m1 + m2) * g * np.sin(theta)
            - 6 * (u - b * x_dot) * np.cos(theta)
        )
        / (l * den),
    ]
    return dydt

def cartpoleQuanser(y, t, u):
    """
    Predict change in state given current state and action in discrete time.
    Using a custom friction function
    """
    # from clients/quanser_robots/cartpole/base.py:CartpoleDynamics
    g = 9.81  # gravity
    #dt = 0.05
    x, x_dot, theta, theta_dot = y

    p = np.array(
            [
                1.00,  # eta_m, Motor efficiency  []
                1.00,  # eta_g, Planetary Gearbox Efficiency []
                3.71,  # Kg,  Planetary Gearbox Gear Ratio
                3.9e-7,  # Jm,  Rotor inertia [kg.m^2]
                6.35e-3,  # r_mp,  Motor Pinion radius [m]
                2.60,  # Rm,  Motor armature Resistance [Ohm]
                0.00767,  # Kt, Motor Torque Constant [N.zz/A]
                0.00767,  # Km, Motor Torque Constant [N.zz/A]
                0.370,  # mc,  Mass of the cart [kg]
                0.127,  # mp, Mass of the pole [kg]
                0.3365 / 2.0,  # pl, Half of the pole length [m]
                5.400,  # Beq, Equivalent Viscous damping Coefficient
                0.0048,  # Bp, Viscous coefficient at the pole (was 0.0024, but this could be too unstable)
            ]
        )
    eta_m, eta_g, Kg, Jm, r_mp, Rm, Kt, Km, mc, mp, pl, Beq, Bp = p.T

    Jp = pl**2 * mp / 3.0  # Pole inertia [kg.m^2]
    Jeq = mc + (eta_g * Kg**2 * Jm) / (r_mp**2)

    # Compute force acting on the cart:
    F = (
        (eta_g * Kg * eta_m * Kt)
        / (Rm * r_mp)
        * (-Kg * Km * x_dot / r_mp + eta_m * u)
    )

    # Compute acceleration:
    a, b = mp + Jeq, mp * pl * np.cos(theta)
    c, d = mp * pl * np.cos(theta), Jp + mp * pl**2
    dd = a * d - b * c

    f_x, f_theta = np.asarray([Beq, Bp]) * np.asarray([x_dot, theta_dot])

    x, y = F - f_x - mp * pl * np.sin(
        theta
    ) * theta_dot**2, 0.0 - f_theta - mp * pl * g * np.sin(theta)

    x_ddot, theta_ddot = (d * x - b * y) / dd, (-c * x + a * y) / dd

    return [x_dot, x_ddot, theta_dot, theta_ddot]


def cartpoleQuanserState(y, dt, u):
    """
    Predict change in state given current state and action in discrete time.
    Using a custom friction function
    """
    # from clients/quanser_robots/cartpole/base.py:CartpoleDynamics
    g = 9.81  # gravity
    x, x_dot, theta, theta_dot = y

    p = np.array(
            [
                1.00,  # eta_m, Motor efficiency  []
                1.00,  # eta_g, Planetary Gearbox Efficiency []
                3.71,  # Kg,  Planetary Gearbox Gear Ratio
                3.9e-7,  # Jm,  Rotor inertia [kg.m^2]
                6.35e-3,  # r_mp,  Motor Pinion radius [m]
                2.60,  # Rm,  Motor armature Resistance [Ohm]
                0.00767,  # Kt, Motor Torque Constant [N.zz/A]
                0.00767,  # Km, Motor Torque Constant [N.zz/A]
                0.370,  # mc,  Mass of the cart [kg]
                0.127,  # mp, Mass of the pole [kg]
                0.3365 / 2.0,  # pl, Half of the pole length [m]
                5.400,  # Beq, Equivalent Viscous damping Coefficient
                0.0048,  # Bp, Viscous coefficient at the pole (was 0.0024, but this could be too unstable)
            ]
        )
    eta_m, eta_g, Kg, Jm, r_mp, Rm, Kt, Km, mc, mp, pl, Beq, Bp = p.T

    Jp = pl**2 * mp / 3.0  # Pole inertia [kg.m^2]
    Jeq = mc + (eta_g * Kg**2 * Jm) / (r_mp**2)

    # Compute force acting on the cart:
    F = (
        (eta_g * Kg * eta_m * Kt)
        / (Rm * r_mp)
        * (-Kg * Km * x_dot / r_mp + eta_m * u)
    )

    # Compute acceleration:
    a, b = mp + Jeq, mp * pl * np.cos(theta)
    c, d = mp * pl * np.cos(theta), Jp + mp * pl**2
    dd = a * d - b * c

    f_x, f_theta = np.asarray([Beq, Bp]) * np.asarray([x_dot, theta_dot])

    x_, y_ = F - f_x - mp * pl * np.sin(
        theta
    ) * theta_dot**2, 0.0 - f_theta - mp * pl * g * np.sin(theta)

    x_ddot, theta_ddot = (d * x_ - b * y_) / dd, (-c * x_ + a * y_) / dd


    x_dot += dt * x_ddot
    theta_dot += dt * theta_ddot

    return y + dt * np.array( [x_dot, x_ddot, theta_dot, theta_ddot] )



def furutaQuanser(y, t, u):
    """
    Predict change in state given current state and action in discrete time.
    Using a custom friction function
    """
    # from clients/quanser_robots/qube/base.py:QubeDynamics
    g = 9.81  # gravity

    theta, theta_dot, alpha, alpha_dot = y

    p = np.array(
            [
                # Motor
                8.4,  # Rm, resistance
                0.042,  # km, back-emf constant (V-s/rad)
                # Rotary arm
                0.095,  # Mr  # mass (kg)
                0.085,  # Lr  # length (m)
                5e-4,  # Dr  # viscous damping (N-m-s/rad), original: 0.0015
                # Pendulum link
                0.024,  # Mp  # mass (kg)
                0.129,  # Lp  # length (m)
                1e-4,  # Dp  # viscous damping (N-m-s/rad), original: 0.0005
            ])
    Rm, km, Mr, Lr, Dr, Mp, Lp, Dp = p

    # Moments of inertia
    Jr = Mr * Lr**2 / 12  # inertia about COM (kg-m^2)
    Jp = Mp * Lp**2 / 12  # inertia about COM (kg-m^2)

    # Constants for equations of motion
    # _c = jnp.zeros(5)
    _c0 = Jr + Mp * Lr**2
    _c1 = 0.25 * Mp * Lp**2
    _c2 = 0.5 * Mp * Lp * Lr
    _c3 = Jp + _c1
    _c4 = 0.5 * Mp * Lp * g

    # Define mass matrix M = [[a, b], [b, c]]
    a = _c0 + _c1 * np.sin(alpha) ** 2
    b = _c2 * np.cos(alpha)
    c = _c3
    d = a * c - b * b

    # Calculate vector [x, y] = tau - C(q, qd)
    trq = km * (u - km * theta_dot) / Rm
    c0 = (
        _c1 * np.sin(2 * alpha) * theta_dot * alpha_dot
        - _c2 * np.sin(alpha) * alpha_dot * alpha_dot
    )
    c1 = -0.5 * _c1 * np.sin(2 * alpha) * theta_dot * theta_dot + _c4 * np.sin(
        alpha
    )

    f_theta, f_alpha = np.asarray([Dr, Dp]) * np.asarray([theta_dot, alpha_dot])


    x = trq - f_theta - c0
    y = -f_alpha - c1

    # Compute M^{-1} @ [x, y]
    theta_ddot = (c * x - b * y) / d
    alpha_ddot = (a * y - b * x) / d

    return [theta_dot, theta_ddot, alpha_dot, alpha_ddot]
