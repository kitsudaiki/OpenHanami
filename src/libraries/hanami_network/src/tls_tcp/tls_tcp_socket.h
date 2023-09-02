/**
 *  @file    tls_tcp_socket.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef TLS_TCP_SOCKET_H
#define TLS_TCP_SOCKET_H

#include <netinet/in.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <openssl/ssl.h>
#include <openssl/bio.h>
#include <openssl/err.h>

#include <tcp/tcp_socket.h>

namespace Kitsunemimi
{
class UnixDomainServer;
class TcpServer;
class TlsTcpServer;

template <class>
class TemplateSocket;

template <class>
class TemplateServer;

class TlsTcpSocket
{
public:
    TlsTcpSocket(TcpSocket&& socket,
                 const std::string &certFile,
                 const std::string &keyFile,
                 const std::string &caFile = "");

    ~TlsTcpSocket();

private:
    friend TemplateSocket<TlsTcpSocket>;
    friend TemplateServer<UnixDomainServer>;
    friend TemplateServer<TcpServer>;
    friend TemplateServer<TlsTcpServer>;

    TlsTcpSocket();

    TcpSocket socket;
    uint32_t type = 3;
    std::string certFile = "";
    std::string keyFile = "";
    std::string caFile = "";

    int getSocketFd() const;
    bool initClientSide(ErrorContainer &error);
    bool initOpenssl(ErrorContainer &error);
    bool isClientSide() const;

    long recvData(int,
                  void* bufferPosition,
                  const size_t bufferSize,
                  int);

    ssize_t sendData(int,
                     const void* bufferPosition,
                     const size_t bufferSize,
                     int);

    bool cleanupOpenssl();

    SSL_CTX* m_ctx = nullptr;
    SSL* m_ssl = nullptr;
};

}

#endif // TLS_TCP_SOCKET_H
