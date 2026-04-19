"""Base classes for Python wrappers around Rust objects."""

from abc import ABC, abstractmethod


class RustWrapperObject(ABC):
    """Abstract base class for Python wrappers around PyO3 Rust objects.

    Following the rustims/imspy pattern: each Python wrapper holds a pointer
    to a PyO3 class instance (the Rust object) and provides a Pythonic API.
    """

    @classmethod
    @abstractmethod
    def from_py_ptr(cls, obj):
        """Create a Python wrapper from a PyO3 pointer."""
        pass

    @abstractmethod
    def get_py_ptr(self):
        """Get the underlying PyO3 Rust object."""
        pass
