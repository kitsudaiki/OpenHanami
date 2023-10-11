/**
 *  @file    abstract_socket.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <abstract_socket.h>
#include <hanami_common/buffer/ring_buffer.h>
#include <hanami_common/threading/cleanup_thread.h>

namespace Hanami
{

/**
 * @brief AbstractSocket::AbstractSocket
 */
AbstractSocket::AbstractSocket(const std::string& threadName) : Hanami::Thread(threadName) {}

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
                                                              Hanami::RingBuffer*,
                                                              AbstractSocket*))
{
    m_target = target;
    m_processMessage = processMessage;
}

}  // namespace Hanami
