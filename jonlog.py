import sys
import structlog
import logging
import threading
import inspect
import retrying
import traceback


def _add_info(logger, log_method, event_dict):
    """
    Add the thread ID to the event_dict if the current execution is within a thread.
    """
    if threading.current_thread() != threading.main_thread():
        event_dict["thread_id"] = threading.get_ident()
        event_dict["thread_name"] = threading.current_thread().name
    else:
        event_dict["thread_id"] = "[Main]"

    # Inspect the stack and find the calling function
    frame = inspect.currentframe()
    while frame:
        func_name = frame.f_code.co_name
        if func_name not in {"retry_with_logging", "decorator", "wrapper", "log_exception", "_add_info", "_process_event", "_proxy_to_logger", "info", "warning", "debug", "critical", "exception"}:
            event_dict["calling_func"] = func_name
            break
        frame = frame.f_back
    return event_dict


def getLogger():
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            _add_info,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.getLogger('moviepy').setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    return structlog.get_logger(level=logging.INFO)

def log_exception(exception, func_name):
    """Log the exception with its stack trace and the function name."""
    stack_trace = traceback.format_exc()
    getLogger().error(f"Exception in {func_name}: {exception}\n{stack_trace}")
    return True

def retry_with_logging(stop_max_attempt_number=5, wait_fixed=2000):
    def decorator(func):
        @retrying.retry(stop_max_attempt_number=stop_max_attempt_number, wait_fixed=wait_fixed,
               retry_on_exception=lambda exception: log_exception(exception, func.__name__))
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
