"""Contains utility functions."""

from pkg_resources import get_distribution, packaging

TORCH_VERSION = packaging.version.parse(get_distribution("torch").version)
TORCH_VERSION_AT_LEAST_1_9_1 = packaging.version.parse("1.9.1") <= TORCH_VERSION
TORCH_VERSION_AT_LEAST_2_0_0 = packaging.version.parse("2.0.0") <= TORCH_VERSION
TORCH_VERSION_AT_LEAST_1_13 = packaging.version.parse("1.13") <= TORCH_VERSION

ADAPTIVE_AVG_POOL_BUG: bool = not TORCH_VERSION_AT_LEAST_2_0_0
