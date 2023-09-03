cd /tmp
# HINT (kitsudaiki): I know this is a stupid unsafe solution and this will also changed in the 
#                    near future, but for the current prototypical state it should be enough
openssl req -nodes -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -subj "/C=DE"
mv key.pem /etc/hanami/
mv cert.pem /etc/hanami/
cd /
Hanami
