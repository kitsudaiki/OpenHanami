/**
 *  @file    tcp_server.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef TCP_SERVER_H
#define TCP_SERVER_H

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <hanami_common/logger.h>

#include <tcp/tcp_socket.h>

namespace Kitsunemimi
{
class TlsTcpServer;

template<class>
class TemplateSocket;

template <class>
class TemplateServer;

class TcpServer
{
public:
    TcpServer(const uint16_t port);
    ~TcpServer();

private:
    friend TemplateServer<TcpServer>;
    friend TlsTcpServer;

    TcpServer();

    bool initServer(ErrorContainer &error);

    int getServerFd() const;
    uint16_t getPort() const;

    int serverFd = 0;
    uint32_t type = 0;;
    std::string caFile = "";
    std::string certFile = "";
    std::string keyFile = "";
    struct sockaddr_in socketAddr;

    uint16_t m_port = 0;
};

}

#endif // TCP_SERVER_H
