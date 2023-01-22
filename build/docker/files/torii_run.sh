cd /tmp
openssl req -nodes -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -subj "/C=DE"
mv key.pem /etc/torii/
mv cert.pem /etc/torii/
cd /
ToriiGateway
