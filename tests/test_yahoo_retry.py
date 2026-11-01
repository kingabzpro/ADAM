import unittest
from unittest.mock import patch, sentinel

from curl_cffi.const import CurlECode
from curl_cffi.requests.exceptions import ConnectionError, Timeout, CertificateVerifyError
from yfinance.exceptions import YFRateLimitError

from adam.market import fetch_history
from adam.fundamentals import fetch_fundamentals


class YahooRetryTests(unittest.TestCase):
    def setUp(self):
        self.pacer_patch = patch('adam.yahoo._pacer.wait')
        self.wait_mock = self.pacer_patch.start()
        self.addCleanup(self.pacer_patch.stop)

    def test_pacer_spaces_requests_and_allows_configuration(self):
        from adam.yahoo import YahooPacer
        pacer = YahooPacer(.25)
        with patch('adam.yahoo.time.monotonic', side_effect=[10., 10., 10.05, 10.25]), \
             patch('adam.yahoo.time.sleep') as sleep:
            pacer.wait()
            pacer.wait()
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], .2)
        for invalid in (-1, float('nan'), True, 61):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pacer.configure(invalid)

    @patch('adam.yahoo.time.sleep')
    @patch('adam.market.yf.Ticker')
    def test_each_retry_is_paced(self, ticker, sleep):
        ticker.return_value.history.side_effect = [Timeout('timeout', code=CurlECode.OPERATION_TIMEDOUT), sentinel.prices]
        self.assertIs(fetch_history('AAPL'), sentinel.prices)
        self.assertEqual(self.wait_mock.call_count, 2)

    @patch('adam.yahoo.time.sleep')
    @patch('adam.market.yf.Ticker')
    def test_history_retries_one_transient_timeout(self, ticker, sleep):
        ticker.return_value.history.side_effect = [Timeout('timeout', code=CurlECode.OPERATION_TIMEDOUT), sentinel.prices]
        self.assertIs(fetch_history('AAPL'), sentinel.prices)
        self.assertEqual(ticker.return_value.history.call_count, 2)
        sleep.assert_called_once()

    @patch('adam.yahoo.time.sleep')
    @patch('adam.fundamentals.yf.Ticker')
    def test_fundamentals_retries_connection_error_once(self, ticker, sleep):
        ticker.return_value.get_info.side_effect = [ConnectionError('connect', code=CurlECode.COULDNT_CONNECT), sentinel.info]
        self.assertIs(fetch_fundamentals('AAPL'), sentinel.info)
        self.assertEqual(ticker.return_value.get_info.call_count, 2)

    @patch('adam.yahoo.time.sleep')
    @patch('adam.market.yf.Ticker')
    def test_rate_limits_certificates_and_invalid_tickers_are_not_retried(self, ticker, sleep):
        for error in (YFRateLimitError(), CertificateVerifyError('bad cert', code=CurlECode.PEER_FAILED_VERIFICATION), ValueError('invalid')):
            ticker.return_value.history.reset_mock()
            ticker.return_value.history.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                fetch_history('AAPL')
            self.assertEqual(ticker.return_value.history.call_count, 1)
        sleep.assert_not_called()

    @patch('adam.yahoo.time.sleep')
    @patch('adam.market.yf.Ticker')
    def test_exhausted_retry_propagates_error(self, ticker, sleep):
        ticker.return_value.history.side_effect = Timeout('timeout', code=CurlECode.OPERATION_TIMEDOUT)
        with self.assertRaises(Timeout):
            fetch_history('AAPL')
        self.assertEqual(ticker.return_value.history.call_count, 2)

    def test_real_http_disconnect_recovers_on_second_attempt(self):
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from types import SimpleNamespace
        from curl_cffi import requests

        class Handler(BaseHTTPRequestHandler):
            calls = 0
            def do_GET(self):
                type(self).calls += 1
                if type(self).calls == 1:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ok')
            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/'
            source = SimpleNamespace(history=lambda **kwargs: requests.get(url, timeout=3).text)
            with patch('adam.market.yf.Ticker', return_value=source), patch('adam.yahoo.time.sleep'):
                self.assertEqual(fetch_history('AAPL'), 'ok')
            self.assertEqual(Handler.calls, 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
