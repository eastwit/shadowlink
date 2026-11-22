"""Phone attitude input for macro hand orientation.

The only supported transport is the third-party phyphox app's Remote Access
HTTP interface.  :class:`PhyphoxService` implements the same small source
contract used by the application: ``start``, ``close``, ``latest``, ``health``.
"""

from shadowlink.phone.service import (
    NullPhoneSource,
    PhoneHealth,
    PhoneSample,
    PhoneSource,
    PhyphoxService,
)

__all__ = [
    "NullPhoneSource",
    "PhoneHealth",
    "PhoneSample",
    "PhoneSource",
    "PhyphoxService",
]
