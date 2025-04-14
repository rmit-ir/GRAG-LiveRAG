from dataclasses import dataclass
import os
import logging
import structlog

# Get log level from environment variable
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Configure structlog, so we have logger names
structlog.stdlib.recreate_defaults(log_level=log_level)


def get_logger(module: str) -> structlog.BoundLogger:
    return structlog.get_logger(module)


get_logger("logging_utils").info(
    "Logger initialized", log_level=log_level_name)

################################################################################
# below is just an example
if __name__ == "__main__":
    """
    Basic usage example of logging_utils.

    Run with: uv run src/utils/logging_utils.py
    Set log level: LOG_LEVEL=DEBUG uv run src/utils/logging_utils.py
    """
    # Initialize logger with component name
    log = get_logger("example")
    log.debug("debugging is hard", a_list=[1, 2, 3])
    log.info("informative!", some_key="some_value")
    log.warning("uh-uh!")
    log.error("omg", a_dict={"a": 42, "b": "foo"})

    @dataclass
    class SomeClass:
        x: int
        y: str
    log.critical("wtf", what=SomeClass(x=1, y="z"))

    try:
        d = {"x": 42}
        print(d["y"])
    except Exception:
        log.exception("poor me")
    log.info("all better now! but with stack info", stack_info=True)
