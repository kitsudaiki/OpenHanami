/**
 *  @file    unix_domain_socket.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef UNIX_DOMAIN_SOCKET_H
#define UNIX_DOMAIN_SOCKET_H

#include <template_socket.h>

namespace Hanami
{
class UnixDomainServer;
class TcpServer;
class TlsTcpServer;

template <class>
class TemplateSocket;

template <class>
class TemplateServer;

class UnixDomainSocket
{
   public:
    UnixDomainSocket(const std::string &socketFile);
    ~UnixDomainSocket();

   private:
    friend TemplateSocket<UnixDomainSocket>;
    friend TemplateServer<UnixDomainServer>;
    friend TemplateServer<TcpServer>;
    friend TemplateServer<TlsTcpServer>;

    UnixDomainSocket();
    UnixDomainSocket(const int socketFd);

    sockaddr_un socketAddr;
    bool isConnected = false;
    bool m_isClientSide = false;
    int socketFd = 0;
    uint32_t type = 0;
    ;

    bool initClientSide(ErrorContainer &error);
    bool initSocket(ErrorContainer &error);
    int getSocketFd() const;
    bool isClientSide() const;
    long recvData(int socket, void *bufferPosition, const size_t bufferSize, int flags);
    ssize_t sendData(int socket, const void *bufferPosition, const size_t bufferSize, int flags);

    std::string m_socketFile = "";
};

}  // namespace Hanami

#endif  // UNIX_DOMAIN_SOCKET_H
