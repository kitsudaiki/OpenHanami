# libKitsunemimiSakuraNetwork

## Description

This library provides a simple session-layer-protocol, which I created for data-transfers in my projects. 

It suppots sessions, which can base on Unix-Domain-Sockets, TCP or TLS encrypted TCP.

The following messages-types are supported:

- stream-messages

	This are very simple and fast messages. For these messages, there is no additional memory allocation and the receiver of the messages gets only little parts. The data are only inside the message-ring-buffer of the socket, so the receiver have to process them instantly, or they will be overwritten. These messages are for cases, when the layer above should handle the data and wants as minimal overhead as possible.

- request-response-messages:

	This are a special case of the standalone-messages. The request-call sends a standalone-message to the receiver with an ID and blocks the thread, which has called the request-method, until the other side sends a response-message with the ID back. The request-message returns after its release the received data. This way it can force a synchronized communication to implement for example RPC-calls.

