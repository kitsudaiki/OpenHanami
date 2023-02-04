/**
 *  @file    abstract_socket.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <libKitsunemimiNetwork/abstract_socket.h>
#include <libKitsunemimiCommon/buffer/ring_buffer.h>
#include <libKitsunemimiCommon/threading/cleanup_thread.h>

namespace Kitsunemimi
{

/**
 * @brief AbstractSocket::AbstractSocket
 */
AbstractSocket::AbstractSocket(const std::string &threadName)
    : Kitsunemimi::Thread(threadName)
{
}

/**
 * @brief destructor, which close the socket before deletion
 */
AbstractSocket::~AbstractSocket() {}

/**
 * @brief add new callback for incoming messages
 *
 * @param target
 * @param processMessage
 *
 * @return false, if object was nullptr, else true
 */
void
AbstractSocket::setMessageCallback(void* target,
                                   uint64_t (*processMessage)(void*,
                                                              Kitsunemimi::RingBuffer*,
                                                              AbstractSocket*))
{
    m_target = target;
    m_processMessage = processMessage;
}

} // namespace Kitsunemimi
