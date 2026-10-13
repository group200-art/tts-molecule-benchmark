import sys, math, io


def finite_or_none(value):
    try:
        x = float(value)
        return x if not isinstance(value, bool) and math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def rounded_csv(fields):
    """Exactly pandas float DataFrame.round(2) -> to_csv -> read_csv.

    Raw values are returned separately by the caller. Only this normalization
    operation imports pandas/NumPy; it computes no molecular properties.
    """
    import numpy as np
    import pandas as pd
    if sys.version_info[:2] != (3, 6) or pd.__version__ != '1.0.5' or np.__version__ != '1.18.5':
        raise RuntimeError('Compatibility normalization requires gellmo_eval_py36 pandas1.0.5/numpy1.18.5')
    names = list(fields)
    if not names:
        return {}, '', {'pandas':pd.__version__, 'numpy':np.__version__}
    values = {p: finite_or_none(v) for p, v in fields.items()}
    frame = pd.DataFrame([values], columns=names, dtype=float).round(2)
    stream = io.StringIO()
    frame.to_csv(stream, index=False)
    serialized = stream.getvalue()
    restored = pd.read_csv(io.StringIO(serialized))
    return {p: finite_or_none(restored.iloc[0][p]) for p in names}, serialized, {'pandas':pd.__version__, 'numpy':np.__version__}

