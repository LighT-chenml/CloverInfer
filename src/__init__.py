"""CloverInfer source package.

This file intentionally makes ``src`` a regular Python package instead of a
namespace package. That avoids mixed-module imports when multiple repositories
or overlays add different ``src`` directories to ``PYTHONPATH`` at runtime.
"""
