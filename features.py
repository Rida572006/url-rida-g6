import re
from urllib.parse import urlparse


def _ensure_scheme(url):
    """Many dataset URLs have no http://, which makes urlparse return an empty hostname."""
    url = str(url)
    return url if "://" in url else "http://" + url
def _safe_parse(url):
    try:
        return urlparse(_ensure_scheme(url))
    except ValueError:
        try:
            return urlparse(str(url))
        except ValueError:
            return urlparse("")

feature_names = [
    "NumDots", "UrlLength", "NumDash", "AtSymbol", "IpAddress",
    "Https", "PathLevel", "PathLength", "NumNumericChars",
]


def extract_features(url):
    url = str(url)
    parsed = _safe_parse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    return [
        url.count("."),                                                    # Number of dots
        len(url),                                                          # URL length
        url.count("-"),                                                    # Number of dashes
        url.count("@"),                                                    # @ symbol
        1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0,        # IP address
        1 if "https" in url.lower() else 0,                                # HTTPS
        path.count("/"),                                                   # Path level
        len(path),                                                         # Path length
        len(re.findall(r"\d", url)),                                       # Numeric characters
    ]


better_feature_names = [
    "UrlLength", "HostnameLength", "PathLength", "NumDots", "NumDash",
    "NumSlash", "AtSymbol", "NumQuestion", "NumEqual", "NumAmpersand",
    "NumDigits", "Https", "HasAtSymbol", "IpAddress", "QueryLength",
]


def extract_better_features(url):
    url = str(url)
    parsed = _safe_parse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    return [
        len(url),                                                          # URL length
        len(hostname),                                                     # Hostname length
        len(path),                                                         # Path length
        url.count("."),                                                    # Number of dots
        url.count("-"),                                                    # Number of dashes
        url.count("/"),                                                    # Number of slashes
        url.count("@"),                                                    # @ symbol
        url.count("?"),                                                    # ? symbol
        url.count("="),                                                    # = symbol
        url.count("&"),                                                    # & symbol
        sum(c.isdigit() for c in url),                                     # Number of digits
        1 if "https" in url.lower() else 0,                                # HTTPS
        1 if "@" in url else 0,                                            # @ present
        1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0,        # IP address
        len(query),                                                        # Query length
    ]
