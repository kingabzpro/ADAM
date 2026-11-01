"""Bounded retry for transient network failures from Yahoo data requests."""
import logging
import math
import random
import threading
import time

from curl_cffi.const import CurlECode
from curl_cffi.requests.exceptions import RequestException

TRANSIENT_CODES = {CurlECode.COULDNT_RESOLVE_HOST, CurlECode.COULDNT_CONNECT,
                   CurlECode.OPERATION_TIMEDOUT, CurlECode.SEND_ERROR,
                   CurlECode.RECV_ERROR, CurlECode.GOT_NOTHING}


class YahooPacer:
    def __init__(self, interval=.25):
        self._lock = threading.Lock()
        self._next = 0.0
        self.configure(interval)

    def configure(self, interval):
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or not 0 <= interval <= 60):
            raise ValueError('Yahoo interval must be between 0 and 60 seconds')
        with self._lock:
            self._interval = float(interval)
            self._next = 0.0

    def wait(self):
        with self._lock:
            delay = self._next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next = time.monotonic() + self._interval


_pacer = YahooPacer()


def configure_yahoo(interval):
    _pacer.configure(interval)


def yahoo_request(operation):
    _pacer.wait()
    try:
        return operation()
    except RequestException as error:
        if error.code not in TRANSIENT_CODES:
            raise
        logging.info('Yahoo request failed transiently; retrying once')
        time.sleep(random.uniform(.25, .75))
        _pacer.wait()
        return operation()
