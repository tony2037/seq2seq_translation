"""
**Reference**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

::

    [KEY: > input, = target, < output]

    > how are glacier caves formed .
    = A partly submerged glacier cave on Perito Moreno Glacier .
    < A partly submerged glacier cave on Perito Moreno Glacier .

... Attention mechanism
    Attention mechanism <https://arxiv.org/abs/1409.0473>`__, which lets the decoder learn to focus over a specific range of the input sequence.

"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
