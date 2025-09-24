# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "..", "sam2_configs")
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

from hydra import initialize_config_module

initialize_config_module("sam2_configs", version_base="1.2")
