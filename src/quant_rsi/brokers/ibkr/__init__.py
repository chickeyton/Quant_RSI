"""Interactive Brokers TWS/IB Gateway adapter.

Only `ibapi_transport` imports the GPL `ibapi` package, and only when a real connection is
made. Everything else talks to the project-owned `Transport` protocol so tests stay offline.
"""
