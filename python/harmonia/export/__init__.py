"""Deterministic projections of the dataset into runnable / interoperable forms.

Every builder takes a loaded ``Dataset`` and returns text (or writes files). The
reference kernel (``reference.py``) is the validation oracle every model export
is checked against.
"""
