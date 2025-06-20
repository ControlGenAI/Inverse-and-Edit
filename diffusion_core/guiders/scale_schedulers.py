import numpy as np


def first_steps(g_scale, steps):
    g_scales = np.ones(50)
    g_scales[:steps] *= g_scale
    g_scales[steps:] = 0.
    return g_scales.tolist()


def last_steps(g_scale, steps):
    g_scales = np.ones(50)
    g_scales[-steps:] *= g_scale
    g_scales[:-steps] = 0.
    return g_scales.tolist()

