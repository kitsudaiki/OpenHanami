/**
 *  @file    tls_tcp_socket.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef TLS_TCP_SERVER_H
#define TLS_TCP_SERVER_H

#include <arpa/inet.h>
#include <netinet/in.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <tcp/tcp_server.h>
#include <tls_tcp/tls_tcp_socket.h>
#include <unistd.h>

#include <string>

namespace Hanami
{

template <class>
class TemplateServer;

class TlsTcpServer
{
   public:
    TlsTcpServer(TcpServer&& server,
                 const std::string& certFile,
                 const std::string& keyFile,
                 const std::string& caFile = "");
    ~TlsTcpServer();

   private:
    friend TemplateServer<TlsTcpServer>;

    TlsTcpServer();

    bool initServer(ErrorContainer& error);
    int getServerFd() const;

    std::string caFile = "";
    std::string certFile = "";
    std::string keyFile = "";

    uint16_t port = 0;
    uint32_t type = 3;

    TcpServer server;
    struct sockaddr_in socketAddr;
};

}  // namespace Hanami

#endif  // TLS_TCP_SERVER_H
