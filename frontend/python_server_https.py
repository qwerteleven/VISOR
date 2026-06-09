import ssl
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler


def create_secure_context(certfile, keyfile):
    """Create SSLContext with secure defaults."""
    # TLS 1.2+ only, secure ciphers
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load certificate and private key
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)

    # Disable insecure options
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1

    return context

with open("../config.json") as f:
    config = json.load(f)


host = config["connection"]["host"]
port = config["connection"]["port"]

context = create_secure_context(config["certified"], config["keyfile"])

httpd = HTTPServer((host, port), SimpleHTTPRequestHandler)
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving HTTPS on https://{host}:{port}")

httpd.serve_forever()