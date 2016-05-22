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


