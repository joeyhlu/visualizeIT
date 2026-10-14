"""Create a workspace-local HTTPS certificate; trust installation is manual."""
import argparse
from datetime import datetime, timedelta, timezone
import ipaddress
from pathlib import Path


def main():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.cache/prototype-sprint/https'))
    parser.add_argument('--ip', action='append', default=[])
    parser.add_argument('--hostname', action='append', default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError('Certificate output must stay in the workspace')
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ('server-key.pem', 'server-cert.pem', 'server-cert.cer')]
    if any(path.exists() for path in paths):
        raise ValueError('Certificate files already exist; choose a fresh directory')
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'VisualizeIt local prototype')])
    names = [x509.DNSName('localhost'), x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]
    names.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in args.ip)
    names.extend(x509.DNSName(name) for name in args.hostname)
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=7))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
            key_encipherment=True, data_encipherment=False, key_agreement=False,
            key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256()))
    paths[0].write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    paths[1].write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    paths[2].write_bytes(cert.public_bytes(serialization.Encoding.DER))
    print('Certificate:', paths[2])
    print('SHA256 fingerprint:', cert.fingerprint(hashes.SHA256()).hex())
    print('Local seven-day certificate generated. No OS/phone trust settings were changed.')


if __name__ == '__main__':
    main()
