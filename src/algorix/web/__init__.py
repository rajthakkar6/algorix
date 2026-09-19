"""Local web UI for Algorix.

The FastAPI instance is exported as `app`; the module that builds it is
`server`, so `algorix.web.app` unambiguously means the application object
rather than shadowing a module of the same name.
"""

from algorix.web.server import app, configure, main

__all__ = ["app", "configure", "main"]
