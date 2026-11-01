"""Bounded retry for transient network failures from Yahoo data requests."""
import logging
import random
import time

from curl_cffi.const import CurlECode
from curl_cffi.requests.exceptions import RequestException

TRANSIENT_CODES = {CurlECode.COULDNT_RESOLVE_HOST, CurlECode.COULDNT_CONNECT,
                   CurlECode.OPERATION_TIMEDOUT, CurlECode.SEND_ERROR,
                   CurlECode.RECV_ERROR, CurlECode.GOT_NOTHING}


def yahoo_request(operation):
    try:
        return operation()
    except RequestException as error:
        if error.code not in TRANSIENT_CODES:
            raise
        logging.info('Yahoo request failed transiently; retrying once')
        time.sleep(random.uniform(.25, .75))
        return operation()
