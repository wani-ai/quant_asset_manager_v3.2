# /scripts/__init__.py

"""
The scripts package contains standalone, executable Python scripts for system
maintenance, setup, and periodic data generation tasks.

Unlike other packages, the modules in this package are not typically imported
by the main application logic. Instead, they are designed to be run
independently from the command line, often on a schedule (e.g., via cron jobs).

This includes tasks such as:
- Generating the absolute quality benchmark.
- Updating the dynamic peer groups using machine learning clustering.
- Setting up and indexing the knowledge base for RAG.

This __init__.py file marks the directory as a Python package, allowing for
cleaner execution paths (e.g., 'python -m scripts.update_dynamic_peer_groups').
It is intentionally kept minimal as it does not need to expose any functions
or classes to the rest of the application.
"""

# This file is intentionally left blank.

