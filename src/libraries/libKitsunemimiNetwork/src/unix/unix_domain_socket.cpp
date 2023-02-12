/**
 *  @file    unix_domain_socket.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <libKitsunemimiNetwork/unix/unix_domain_socket.h>
#include <libKitsunemimiCommon/threading/cleanup_thread.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{

/**
 * @brief constructor for the socket-side of the unix-socket-connection
 *
 * @param socketFile
 */
UnixDomainSocket::UnixDomainSocket(const std::string &socketFile)
{
    this->m_socketFile = socketFile;
    this->m_isClientSide = true;
    this->type = 1;
}

/**
 * @brief default-constructor
 */
UnixDomainSocket::UnixDomainSocket() {}

/**
 * @brief destructor
 */
UnixDomainSocket::~UnixDomainSocket() {}

/**
 * @brief init socket on client-side
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
UnixDomainSocket::initClientSide(ErrorContainer &error)
{
    if(socketFd > 0) {
        return true;
    }

    bool result = initSocket(error);
    if(result == false) {
        return false;
    }

    isConnected = true;
    LOG_INFO("Successfully initialized unix-socket client to targe: " + m_socketFile);

    return true;
}

/**
 * @brief constructor for the server-side of the unix-socket-connection, which is called by the
 *        unix-server for each incoming connection
 *
 * @param socketFd file-descriptor of the socket-socket
 */
UnixDomainSocket::UnixDomainSocket(const int socketFd)
{
    this->socketFd = socketFd;
    this->m_isClientSide = false;
    this->type = 1;
    this->isConnected = true;
}

/**
 * @brief get file-descriptor
 *
 * @return file-descriptor
 */
int
UnixDomainSocket::getSocketFd() const
{
    return socketFd;
}

/**
 * @brief init unix-socket and connect to the server
 *
 * @param error reference for error-output
 *
 * @return false, if socket-creation or connection to the server failed, else true
 */
bool
UnixDomainSocket::initSocket(ErrorContainer &error)
{
    struct sockaddr_un address;

    // check file-path length to avoid conflics, when copy to the address
    if(m_socketFile.size() > 100)
    {
        error.addMeesage("Failed to create a unix-socket, "
                         "because the filename is longer then 100 characters: \""
                         + m_socketFile
                         + "\"");
        error.addMeesage("use a shorter name for the unix-domain-socket.");
        return false;
    }

    // create socket
    socketFd = socket(PF_LOCAL, SOCK_STREAM, 0);
    if(socketFd < 0)
    {
        error.addMeesage("Failed to create a unix-socket");
        error.addSolution("Maybe no permissions to create a unix-socket on the system");
        return false;
    }

    // set other informations
    address.sun_family = AF_LOCAL;
    strncpy(address.sun_path, m_socketFile.c_str(), m_socketFile.size());
    address.sun_path[m_socketFile.size()] = '\0';

    // create connection
    if(connect(socketFd, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) < 0)
    {
        error.addMeesage("Failed to connect unix-socket to server with addresse: \""
                         + m_socketFile
                         + "\"");
        error.addSolution("check your write-permissions for the location \""
                          + m_socketFile
                          + "\"");
        return false;
    }

    socketAddr = address;
    isConnected = true;

    return true;
}

/**
 * @brief check if socket is on client-side of the connection
 *
 * @return true, if socket is client-side, else false
 */
bool
UnixDomainSocket::isClientSide() const
{
    return m_isClientSide;
}

/**
 * @brief receive data
 *
 * @return number of read bytes
 */
long
UnixDomainSocket::recvData(int socket,
                           void* bufferPosition,
                           const size_t bufferSize,
                           int flags)
{
    return recv(socket, bufferPosition, bufferSize, flags);
}

/**
 * @brief send data
 *
 * @return number of written bytes
 */
ssize_t
UnixDomainSocket::sendData(int socket,
                           const void* bufferPosition,
                           const size_t bufferSize,
                           int flags)
{
    return send(socket, bufferPosition, bufferSize, flags);
}

}
