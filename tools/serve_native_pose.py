"""Start the native GoTrack bridge. Source delivery does not start this server."""
from __future__ import annotations

import argparse
import ipaddress
import logging
import os
from pathlib import Path
import secrets
import ssl
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.workbench_live_pose import ConfiguredGoTrackFactory
from tools.native_pose_bridge import BoundedNativeServer, NativePoseBridge, make_handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Authenticated native trained-GoTrack HTTPS bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--config", type=Path, help="trusted operator native_gotrack_operator_v1 JSON")
    parser.add_argument("--cert", type=Path, help="HTTPS certificate PEM trusted by the physical phone")
    parser.add_argument("--key", type=Path, help="HTTPS private key PEM")
    parser.add_argument("--token-env", help="environment variable containing a process-scoped operator Bearer token")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if bool(args.cert) != bool(args.key):
        parser.error("--cert and --key must be supplied together")
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback and args.cert is None:
        parser.error("non-loopback native serving requires HTTPS --cert and --key")
    token = os.environ.get(args.token_env, "") if args.token_env else secrets.token_urlsafe(32)
    if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        parser.error("operator token must contain at least 32 non-whitespace characters")
    context = None
    if args.cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(args.cert), str(args.key))
    factory = ConfiguredGoTrackFactory(args.config) if args.config else None
    bridge = NativePoseBridge(factory)
    server = None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        server = BoundedNativeServer((args.host, args.port), make_handler(bridge, token))
        if context is not None:
            server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        if not args.token_env:
            print("Native operator Bearer token (this process only): "+token, file=sys.stderr)
        print(f"Native bridge: {'https' if args.cert else 'http'}://{args.host}:{args.port}; live admission is reverified per frame",
              file=sys.stderr)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
