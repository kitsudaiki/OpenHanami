# Torii

!!! info

    Repository: [ToriiGateway](https://github.com/kitsudaiki/ToriiGateway)


## Basic

The Torii works as proxy for the complete infrastructure. Every incoming incoming connection must go trough the Torii to reach the other components. 

??? question "Why it is named `Torii`"

    The `Torii` it the only component, which has not a female japanese name, like the other ones. Torii's are in general the gates of shinto shrines in Japan. Because it is already different to the other components (has no database and no API-endpoints), as proxy it works also as gate for the infrastructure and because I like torii's a lot, this name was too perfect for this component here.
    
![Database-Layers](../img/Torii_basic.drawio)

It checks the token of the connection by requesting `Misaki` and sends an audit-log to `Shiori` for each incoming request. After this it converts the HTTP-message into a custom format, which is then send to the destination in the backend. Beside this the Torii also provides the files for the dashboard.


## Internal workflow

The following graphics should visualize the internal flow of an incoming HTTP-request within the Torii before forwarding it to the backend.

![Database-Layers](../img/Torii_internal.drawio)
