"""Public API for ILLUME."""

__version__ = "0.1.0"
__all__ = ["ILLUME"]


def __getattr__(name):
    if name == "ILLUME":
        try:
            from .core import ILLUME
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "torchvision"}:
                raise ImportError(
                    'ILLUME needs the model extra: pip install "mimosa-illume[model]"'
                ) from exc
            raise
        return ILLUME
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
