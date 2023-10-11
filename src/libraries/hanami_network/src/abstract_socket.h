/**
 *  @file    abstract_socket.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef KITSUNEMIMI_NETWORK_ABSTRACT_SOCKET_H
#define KITSUNEMIMI_NETWORK_ABSTRACT_SOCKET_H

#include <arpa/inet.h>
#include <errno.h>
#include <hanami_common/buffer/ring_buffer.h>
#include <hanami_common/logger.h>
#include <hanami_common/threading/thread.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <string>
#include <vector>

namespace Hanami
{
struct RingBuffer;

class AbstractSocket : public Hanami::Thread
{
   public:
    AbstractSocket(const std::string &threadName);
    ~AbstractSocket();

    void setMessageCallback(void *target,
                            uint64_t (*processMessage)(void *,
                                                       Hanami::RingBuffer *,
                                                       AbstractSocket *));

    virtual bool initConnection(Hanami::ErrorContainer &error) = 0;
    virtual bool isClientSide() = 0;
    virtual uint32_t getType() = 0;
    virtual bool sendMessage(const std::string &message, ErrorContainer &error) = 0;
    virtual bool sendMessage(const void *message,
                             const uint64_t numberOfBytes,
                             ErrorContainer &error)
        = 0;
    virtual bool closeSocket() = 0;

   protected:
    // callback-parameter
    void *m_target = nullptr;
    uint64_t (*m_processMessage)(void *, RingBuffer *, AbstractSocket *);
};

}  // namespace Hanami

#endif  // KITSUNEMIMI_NETWORK_ABSTRACT_SOCKET_H
