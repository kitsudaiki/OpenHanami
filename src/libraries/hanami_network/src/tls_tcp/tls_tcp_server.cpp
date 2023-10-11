/**
 *  @file    tls_tcp_socket.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <hanami_common/logger.h>
#include <tls_tcp/tls_tcp_server.h>
#include <tls_tcp/tls_tcp_socket.h>

namespace Hanami
{

/**
 * @brief constructor
 */
TlsTcpServer::TlsTcpServer(TcpServer &&server,
                           const std::string &certFile,
                           const std::string &keyFile,
                           const std::string &caFile)
{
    this->server = std::move(server);
    this->port = server.getPort();
    this->certFile = certFile;
    this->keyFile = keyFile;
    this->caFile = caFile;
    this->socketAddr = server.socketAddr;
}

/**
 * @brief default-constructor
 */
TlsTcpServer::TlsTcpServer() {}

/**
 * @brief TlsTcpServer::initServer
 * @param error
 * @return
 */
bool
TlsTcpServer::initServer(ErrorContainer &error)
{
    return server.initServer(error);
}

/**
 * @brief destructor
 */
TlsTcpServer::~TlsTcpServer() {}

/**
 * @brief get file-descriptor
 *
 * @return file-descriptor
 */
int
TlsTcpServer::getServerFd() const
{
    return server.getServerFd();
}

}  // namespace Hanami
