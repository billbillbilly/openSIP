# opensip/imagebind — vendored from facebookresearch/ImageBind
#
# Original source: https://github.com/facebookresearch/ImageBind
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under CC-BY-NC-SA 4.0
#
# Modifications:
#   - Removed text, video, depth, thermal, IMU loaders (vision + audio only)
#   - Replaced pytorchvideo dependency with inline clip sampler
#   - Shimmed iopath / ftfy / regex with stdlib equivalents
#   - Updated intra-package imports to relative paths
