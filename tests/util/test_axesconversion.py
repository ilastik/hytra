from __future__ import unicode_literals
import numpy as np
import hytra.util.axesconversion as ax

def test_conversion_default():
    a = np.zeros([123, 64])
    b = ax.adjustOrder(a, 'yx')
    assert(b.ndim == 5)
    assert(b.shape[0] == 1)
    assert(b.shape[1] == 64)
    assert(b.shape[2] == 123)
    assert(b.shape[3] == 1)
    assert(b.shape[4] == 1)

def test_conversion_other():
    a = np.zeros([123, 64, 987])
    b = ax.adjustOrder(a, 'yxc', 'cxtzy')
    assert(b.ndim == 5)
    assert(b.shape[0] == 987)
    assert(b.shape[1] == 64)
    assert(b.shape[2] == 1)
    assert(b.shape[3] == 1)
    assert(b.shape[4] == 123)

def test_frame_slicing():
    a = np.zeros([123, 64, 987])
    b = a[ax.getFrameSlicing('xyt', 12)]
    assert(b.ndim == 2)
    assert(b.shape[0] == 123)
    assert(b.shape[1] == 64)

    b = a[ax.getFrameSlicing('xyt', 12, 'y')]
    assert(b.ndim == 2)
    assert(b.shape[0] == 123)
    assert(b.shape[1] == 987)
