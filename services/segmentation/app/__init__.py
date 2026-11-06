"""Segmentation service package.

Compat shim: the CUDA base image ships Python 3.11.0rc1, which lacks
sys.get_int_max_str_digits / sys.set_int_max_str_digits (added in 3.11.0
final). torch._dynamo polyfills require them at import time.
"""

import sys

if not hasattr(sys, "get_int_max_str_digits"):
    _int_max_str_digits = 4300  # CPython default

    def _get_int_max_str_digits() -> int:
        return _int_max_str_digits

    def _set_int_max_str_digits(maxdigits: int) -> None:
        global _int_max_str_digits
        _int_max_str_digits = maxdigits

    sys.get_int_max_str_digits = _get_int_max_str_digits
    sys.set_int_max_str_digits = _set_int_max_str_digits
