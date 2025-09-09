import logging
import io
import os
import datetime
import time as time_module
from typing import List, Union

class MemoryHandler(logging.Handler):
    """Custom handler that stores log messages in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.logs = []

    def emit(self, record) -> None:
        msg = self.format(record)
        self.logs.append(msg)

    def get_logs(self) -> List[str]:
        return self.logs

    def clear_logs(self) -> None:
        self.logs = []

class NullHandler(logging.Handler):
    """Handler that does nothing with log records."""

    def emit(self, record) -> None:
        pass

class EnhancedLogger(logging.Logger):
    """Enhanced logger with custom info method to include metadata."""

    def __init__(self, name, level=logging.NOTSET) -> None:
        super().__init__(name, level)
        self.start_time = time_module.time()
        self.memory_handler = MemoryHandler()
        formatter = logging.Formatter('%(message)s')
        self.memory_handler.setFormatter(formatter)
        self.addHandler(self.memory_handler)

    def info(self, msg, *args, **kwargs) -> None:
        """Enhanced info method that adds pid, time, and elapsed time to the message."""
        pid = os.getpid()
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_seconds = time_module.time() - self.start_time
        elapsed_h = int(elapsed_seconds // 3600)
        elapsed_m = int((elapsed_seconds % 3600) // 60)
        elapsed_s = int(elapsed_seconds % 60)
        elapsed_ms = int((elapsed_seconds % 1) * 1000)
        elapsed = f"{elapsed_h:02}:{elapsed_m:02}:{elapsed_s:02}.{elapsed_ms:03}"

        formatted_msg = f"[pid={pid}] time=({current_time}) elapsed=({elapsed})] {msg}"
        super().info(formatted_msg, *args, **kwargs)

    def get_logs(self) -> List[str]:
        """Get all logs stored in memory.

        Returns:
            List of log messages
        """
        return self.memory_handler.get_logs()

    def get_logs_text(self) -> str:
        """Get all logs as a single string.
        
        Returns:
            String containing all logs
        """
        return "\n".join(self.memory_handler.get_logs())

    def clear_logs(self) -> None:
        """clear all stored logs from memory."""
        self.memory_handler.clear_logs()

class LogHandler(logging.Handler):
    def __init__(self, file_path: str, flush_every: int=5) -> None:
        super().__init__()
        self.file_path = file_path
        self.buffer = io.StringIO()
        self.flush_every = flush_every
        self.counter = 0
        self.start_time = datetime.datetime.now()

    def emit(self, record) -> None:
        msg = self.format(record)
        self.buffer.write(msg + '\n')
        self.counter += 1
        if self.counter >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self.buffer.tell() == 0:
            return
        with open(self.file_path, 'a') as f:
            f.write(self.buffer.getvalue())
        self.buffer = io.StringIO()
        self.counter = 0

def get_logger(name: str = "DefaultLogger", log_file: Union[str, List[str], None] = None,
               flush_every: int = 1, console_output: bool = True) -> logging.Logger:
    """
    Get an enhanced logger with metadata-rich info method.

    Args:
        name: logger name
        log_file: Optional path **or list of paths** to log file(s)
        flush_every: How often to flush to file
        console_output: Whether to output logs to console (default: True)

    Returns:
        EnhancedLogger instance
    """

    # Register the EnhancedLogger class
    logging.setLoggerClass(EnhancedLogger)

    # Get a logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates (but keep memory handler)
    memory_handler = None
    for handler in logger.handlers:
        if isinstance(handler, MemoryHandler):
            memory_handler = handler
        else:
            logger.removeHandler(handler)

    # Re-add memory handler if it was removed
    if memory_handler is None:
        memory_handler = MemoryHandler()
        formatter = logging.Formatter('%(message)s')
        memory_handler.setFormatter(formatter)
        logger.addHandler(memory_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # normalize to a list
        paths = log_file if isinstance(log_file, (list, tuple)) else [log_file]
        for path in paths:
            if not path:
                continue
            # ensure directory exists (if any)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            file_handler = LogHandler(path, flush_every)
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
    init_msg = (
        "\n" +
        "=" * 60 + "\n" +
        " EnhancedLogger initialized\n" +
        f" Logger Name: {name}\n" +
        f" Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" +
        "=" * 60
    )
    logger.info(init_msg)

    return logger