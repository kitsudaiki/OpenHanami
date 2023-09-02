/**
 *  @file    abstract_server.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <abstract_server.h>
#include <abstract_socket.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{

/**
 * @brief AbstractServer::AbstractServer
 */
AbstractServer::AbstractServer(void* target,
                               void (*processConnection)(void*, AbstractSocket*),
                               const std::string &threadName)
    : Kitsunemimi::Thread(threadName)
{
    m_target = target;
    m_processConnection = processConnection;
}

/**
 * @brief AbstractServer::~AbstractServer
 */
AbstractServer::~AbstractServer()
{
}

}
