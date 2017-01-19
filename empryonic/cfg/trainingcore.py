from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import configparser
import io

sample_config = """
[display]
colormap3d: gray
"""

cfg = configparser.SafeConfigParser()
cfg.readfp(io.BytesIO(sample_config))
