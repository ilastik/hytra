"""
This module provides some helper methods to deal with multidimensional arrays of different axes order.
"""
from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import numpy as np

def adjustOrder(volume, inputAxes, outputAxes='txyzc'):
    """
    This method allows to convert a given `volume` (with given `inputAxes` ordering)
    into a different axis ordering, specified as `outputAxes` string (e.g. "xyzt").

    Allowed axes are `t`, `x`, `y`, `z`, `c`.

    The default format volumes are converted to is "txyzc", axes that are missing in the input
    volume are created with size 1.
    """
    assert(isinstance(volume, np.ndarray))
    assert(len(volume.shape) == len(inputAxes))
    assert(len(outputAxes) >= len(inputAxes))
    assert(not any(a not in 'txyzc' for a in outputAxes))
    assert(not any(a not in 'txyzc' for a in inputAxes))

    outVolume = volume

    # find present and missing axes
    positions = {}
    missingAxes = []
    for axis in outputAxes:
        try:
            positions[axis] = inputAxes.index(axis)
        except ValueError:
            missingAxes.append(axis)

    # insert missing axes at the end
    for m in missingAxes:
        outVolume = np.expand_dims(outVolume, axis=-1)
        positions[m] = outVolume.ndim - 1

    # transpose
    axesRemapping = [positions[a] for a in outputAxes]
    outVolume = np.transpose(outVolume, axes=axesRemapping)

    return outVolume

def getFrameSlicing(inputAxes, selectValue, selectAxis='t'):
    """
    This methods helps to get a slice of a multidimensional array of the specified `inputAxes`,
    where only for one specific axis (`selectAxis`) an index (or a list of indices, or a slicing object) is given.

    Example: `myarray[getFrameSlicing('xzt', 3, t)]`
    Example: `myarray[getFrameSlicing('xzt', [3,7,9], t)]`
    """
    assert(len(selectAxis) == 1)
    assert(inputAxes.count(selectAxis) == 1)
    slicing = tuple()
    for a in inputAxes:
        if a == selectAxis:
            slicing += (selectValue,)
        else:
            slicing += (slice(None),)
    return slicing
