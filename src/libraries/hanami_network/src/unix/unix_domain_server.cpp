/**
 *  @file    unix_domain_server.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <hanami_common/logger.h>
#include <unix/unix_domain_server.h>

namespace Hanami
{

/**
 * @brief constructor
 *
 * @param socketFile file for the unix-domain-socket
 */
UnixDomainServer::UnixDomainServer(const std::string& socketFile)
{
    this->m_socketFile = socketFile;
    this->type = 1;
}

/**
 * @brief default-constructor
 */
UnixDomainServer::UnixDomainServer() {}

/**
 * @brief destructor
 */
UnixDomainServer::~UnixDomainServer() {}

/**
 * @brief creates a server on a specific port
 *
 * @param error reference for error-output
 *
 * @return false, if server creation failed, else true
 */
bool
UnixDomainServer::initServer(ErrorContainer& error)
{
    // check file-path length to avoid conflics, when copy to the sockaddr_un-object
    if (m_socketFile.size() > 100) {
        error.addMeesage(
            "Failed to create a unix-server, "
            "because the filename is longer then 100 characters: \""
            + m_socketFile + "\"");
        error.addSolution("use a shorter name for the unix-domain-socket");
        return false;
    }

    // create socket
    serverFd = socket(AF_LOCAL, SOCK_STREAM, 0);
    if (serverFd < 0) {
        error.addMeesage("Failed to create a unix-socket");
        error.addSolution("Maybe no permissions to create a unix-socket on the system");
        return false;
    }

    unlink(m_socketFile.c_str());
    socketAddr.sun_family = AF_LOCAL;
    strncpy(socketAddr.sun_path, m_socketFile.c_str(), m_socketFile.size());
    socketAddr.sun_path[m_socketFile.size()] = '\0';

    // bind to port
    if (bind(serverFd, reinterpret_cast<struct sockaddr*>(&socketAddr), sizeof(socketAddr)) < 0) {
        error.addMeesage("Failed to bind unix-socket to addresse: \"" + m_socketFile + "\"");
        return false;
    }

    // start listening for incoming connections
    if (listen(serverFd, 5) == -1) {
        error.addMeesage("Failed listen on unix-socket on addresse: \"" + m_socketFile + "\"");
        return false;
    }

    LOG_INFO("Successfully initialized unix-socket server on targe: " + m_socketFile);

    return true;
}

/**
 * @brief get file-descriptor
 *
 * @return file-descriptor
 */
int
UnixDomainServer::getServerFd() const
{
    return serverFd;
}

}  // namespace Hanami
