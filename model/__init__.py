from .version import __version__

def __getattr__(name):
    if name in ("KANamav1", "KANamav2", "KANamav3", "KANamav4"):
        from model import KANamav1, KANamav2, KANamav3, KANamav4
        return locals()[name]

__all__ = ["KANamav1", "KANamav2", "KANamav3", "KANamav4"]