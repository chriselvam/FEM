import logging

from aso.logging import enable_logging

logger = logging.getLogger(__name__)


def pytest_sessionstart(session):
    """This function is called once before any tests are run."""
    enable_logging(
        console_level=logging.ERROR,
        file_level=logging.DEBUG,
    )
    logger.info("Logging enabled for pytest session.")
