#!/usr/bin/ipython -wthread
#
# (c) Bernhard X. Kausler, 2010
#

'''Embryo Laboratory

Elab is a high level interface to the emproynic package intended to be
used from an interactive shell such as ipython or as a domain specific
language for scripting.

Elab may be imported or called as a script. In the latter case, it
will start an interactive python shell with elab alreadey imported.

The following empryonic modules are imported:
- segment
- io
- plot
- training
'''

import segment
from segment import msa

import io
from io import loadRaw, loadSegmentation

import plot
from plot import cutPlanes

import training
