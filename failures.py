"""Allowlisted processing diagnostics: never copy exception messages, requests or bill values."""
import re

MESSAGES = {
    "configuration": "Bill processing is not configured correctly. Check the server's .env model/provider settings and API key.",
    "authentication": "The AI service rejected access. Check the server's API key and account permissions.",
    "quota": "The AI service reported a rate or quota limit. Check the account quota before retrying.",
    "network": "The connection to the AI service failed. Check the server's network, proxy and TLS settings, then retry.",
    "timeout": "The AI service request timed out. Please retry later.",
    "api": "The AI service could not complete this request. Check the diagnostic status and retry when the service is available.",
    "validation": "The extracted response could not be validated. Please retry; the previous result has not been replaced.",
    "image": "This image could not be decoded. Choose a valid JPG or PNG image.",
    "processing": "A local processing error occurred. Check the sanitized server diagnostic before retrying.",
}


def safe_failure(error: Exception, stage: str) -> dict:
    """Use only class names and numeric HTTP status; never str(error) or error payloads."""
    names, seen, item = [], set(), error
    while item is not None and id(item) not in seen and len(names) < 5:
        seen.add(id(item))
        name = type(item).__name__
        names.append(name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) else "Exception")
        item = item.__cause__ or item.__context__
    status = getattr(error, "code", None)
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    category = stage if stage in MESSAGES else "processing"
    if status in (401, 403):
        category = "authentication"
    elif status == 429:
        category = "quota"
    elif status is not None:
        category = "api"
    elif any("timeout" in name.lower() for name in names):
        category = "timeout"
    elif any(any(part in name.lower() for part in ("connect", "ssl", "proxy")) for name in names):
        category = "network"
    elif stage == "processing" and any(name in ("ValidationError", "JSONDecodeError") for name in names):
        category = "validation"
    return {"category": category, "exception_type": names[0], "status": status, "cause_types": names}


class ProcessingFailure(Exception):
    def __init__(self, diagnostic: dict):
        # Accept only this closed set of fields; never relay an arbitrary metadata object.
        self.diagnostic = {key: diagnostic[key] for key in ("category", "exception_type", "status", "cause_types")}
        super().__init__(MESSAGES.get(self.diagnostic["category"], MESSAGES["processing"]))
