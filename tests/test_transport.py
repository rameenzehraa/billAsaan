"""Offline transport/configuration regressions: no network requests."""
import os
import ssl
import unittest
from unittest.mock import patch

from config import load_settings, REQUEST_TIMEOUT_S
from extract import _client


class TransportTests(unittest.TestCase):
    def test_default_transport_unchanged(self):
        with patch.dict(os.environ, {"GENAI_IPV4_ONLY": "0"}), patch("extract.genai.Client") as client:
            _client("offline-placeholder")
            options = client.call_args.kwargs["http_options"]
            self.assertNotIn("transport", options.client_args)
            self.assertEqual(options.timeout, REQUEST_TIMEOUT_S * 1000)

    def test_ipv4_transport_keeps_certificate_validation(self):
        with patch.dict(os.environ, {"GENAI_IPV4_ONLY": "1"}), patch("extract.genai.Client") as client:
            _client("offline-placeholder")
            transport = client.call_args.kwargs["http_options"].client_args["transport"]
            try:
                self.assertEqual(transport._pool._local_address, "0.0.0.0")
                self.assertEqual(transport._pool._ssl_context.verify_mode, ssl.CERT_REQUIRED)
                self.assertTrue(transport._pool._ssl_context.check_hostname)
            finally:
                transport.close()

    def test_config_requires_explicit_valid_opt_in(self):
        for value, expected in [("0", False), ("false", False), ("1", True), ("true", True)]:
            with self.subTest(value=value), patch.dict(os.environ, {"GENAI_IPV4_ONLY": value}):
                self.assertEqual(load_settings().force_ipv4, expected)
        with patch.dict(os.environ, {"GENAI_IPV4_ONLY": "invalid"}):
            with self.assertRaisesRegex(ValueError, "GENAI_IPV4_ONLY"):
                load_settings()


if __name__ == "__main__":
    unittest.main()
