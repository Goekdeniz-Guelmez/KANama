def __getattr__(name):
    if name in ("KANLinear", "KAN"):
        from .kan import KANLinear, KAN
        return locals()[name]
    if name in ("ModelArgs"):
        from .args import ModelArgs
        return locals()[name]
    if name in ("KANamav1", "KANamav2", "KANamav3", "KANamav4"):
        from model import KANamav1, KANamav2, KANamav3, KANamav4
        return locals()[name]

__all__ = ["KANLinear", "KAN", "KANamav1", "KANamav2", "KANamav3", "KANamav4", "ModelArgs"]