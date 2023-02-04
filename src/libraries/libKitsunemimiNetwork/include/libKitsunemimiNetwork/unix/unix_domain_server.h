/**
 *  @file    unix_domain_server.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef UNIX_DOMAIN_SERVER_H
#define UNIX_DOMAIN_SERVER_H

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiNetwork/unix/unix_domain_socket.h>

namespace Kitsunemimi
{
template<class>
class TemplateSocket;

template <class>
class TemplateServer;

class UnixDomainServer
{
public:
    UnixDomainServer(const std::string &socketFile);
    ~UnixDomainServer();

private:
    friend TemplateServer<UnixDomainServer>;

    UnixDomainServer();

    int getServerFd() const;
    bool initServer(ErrorContainer &error);

    int serverFd = 0;
    uint32_t type = 0;

    uint16_t m_port = 0;
    struct sockaddr_un socketAddr;
    std::string caFile = "";
    std::string certFile = "";
    std::string keyFile = "";

    std::string m_socketFile = "";
};

} // namespace Kitsunemimi

#endif // UNIX_DOMAIN_SERVER_H
