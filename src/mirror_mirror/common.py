from types import FunctionType
from typing import NoReturn

from mirror_mirror.camera_server import logger


def log_errors(func: FunctionType):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as ex:
            logger.exception("%s failed with error %s", func.__name__, ex, exc_info=ex)
            raise

    return wrapper


def assert_unreachable() -> NoReturn:
    raise RuntimeError("Unreachable code reached")
