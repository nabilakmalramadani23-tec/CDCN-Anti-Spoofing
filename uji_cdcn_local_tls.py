import argparse
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def main():
    parser = argparse.ArgumentParser(description='Buat sertifikat HTTPS lokal Uji Coba CDCN Local')
    parser.add_argument('ip', type=ip_address)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / 'checkpoints' / 'tls'
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    def builder(subject, issuer, key):
        return (x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(days=1))
                .not_valid_after(now + timedelta(days=365)))

    authority = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Uji Coba CDCN Local CA')])
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca = (builder(authority, authority, ca_key)
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False),
                         critical=True)
          .sign(ca_key, hashes.SHA256()))

    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, str(args.ip))])
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    certificate = (builder(subject, authority, server_key)
                   .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                   .add_extension(x509.SubjectAlternativeName([x509.IPAddress(args.ip)]), critical=False)
                   .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                   .sign(ca_key, hashes.SHA256()))
    (folder / 'Uji-CDCN-Local-CA.cer').write_bytes(ca.public_bytes(serialization.Encoding.DER))
    (folder / 'ca.pem').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (folder / 'server.pem').write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    (folder / 'server-key.pem').write_bytes(server_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    print(folder / 'Uji-CDCN-Local-CA.cer')


if __name__ == '__main__':
    main()
