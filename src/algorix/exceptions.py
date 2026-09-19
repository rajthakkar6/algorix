"""Exception hierarchy for Algorix.

The distinction between "no data exists" and "could not reach the source" is
deliberate and load-bearing. CLAUDE.md forbids silently swallowing a
data-source failure, which is only enforceable if callers can tell the two
apart:

    - DataUnavailableError  -> expected on holidays, pre-listing dates, or a
                               symbol with no history. Usually not an error
                               worth alerting on.
    - SourceUnreachableError -> the feed is broken or we are rate-limited.
                               Always worth surfacing.

Collapsing both into one exception (or worse, into a `None` return) is how a
scan silently produces scores from missing data.
"""


class AlgorixError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(AlgorixError):
    """Configuration is missing or invalid."""


class DataSourceError(AlgorixError):
    """Base class for anything that goes wrong fetching external data."""


class DataUnavailableError(DataSourceError):
    """The source was reached, but holds no data for the request.

    Expected and usually benign: a non-trading day, a symbol before its
    listing date, or a series that does not publish on this date.
    """


class SourceUnreachableError(DataSourceError):
    """The source could not be reached, or refused the request.

    Network failure, timeout, HTTP 5xx, or rate limiting. Never treat this as
    "no data" -- doing so turns an outage into a silent gap.
    """


class DataIntegrityError(DataSourceError):
    """Data was returned but failed validation.

    Negative prices, a high below its low, zero volume on a trading day,
    dates outside the requested range. A signal computed from data like this
    is worse than no signal, because it looks valid.
    """


class StorageError(AlgorixError):
    """Persistence layer failure."""
