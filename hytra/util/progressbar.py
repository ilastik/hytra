from __future__ import unicode_literals
from __future__ import division
from builtins import object
from past.utils import old_div
import sys

class ProgressBar(object):
    def __init__(self, start=0, stop=100):
        self._state = 0
        self._start = start
        self._stop = stop

    def reset(self, val=0):
        self._state = val

    def show(self, increase=1):
        self._state += increase
        if self._state > self._stop:
            self._state = self._stop

        # show
        pos = old_div(float(self._state - self._start),(self._stop - self._start))
        try:
            sys.stdout.write("\r[%-20s] %d%%" % ('='*int(20*pos), (100*pos)))

            if self._state == self._stop:
                sys.stdout.write('\n')
                sys.stdout.flush()
        except IOError:
            pass
