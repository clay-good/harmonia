"""Deterministic projections of the dataset into runnable / interoperable forms.

Every builder takes a loaded ``Dataset`` and returns text (or writes files). The
reference kernel (``reference.py``) is the validation oracle every model export
is checked against.
"""


def default_dataset_version() -> str:
    """The version stamped into an export's ``harmonia:datasetVersion`` RDF when a
    caller doesn't supply one. Code and dataset ship as one version, so this is the
    package ``__version__``. Resolved lazily (at call time, not import time) to
    avoid a circular import — by the time any builder runs, ``harmonia`` is fully
    imported."""
    from .. import __version__
    return __version__
