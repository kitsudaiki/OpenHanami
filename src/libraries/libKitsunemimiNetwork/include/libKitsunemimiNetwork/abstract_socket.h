/**
 *  @file    abstract_socket.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef KITSUNEMIMI_NETWORK_ABSTRACT_SOCKET_H
#define KITSUNEMIMI_NETWORK_ABSTRACT_SOCKET_H

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cinttypes>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <errno.h>
#include <atomic>

#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/buffer/ring_buffer.h>

namespace Kitsunemimi
{
struct RingBuffer;

class AbstractSocket
        : public Kitsunemimi::Thread
{
public:
    AbstractSocket(const std::string &threadName);
    ~AbstractSocket();

    void setMessageCallback(void *target,
                            uint64_t (*processMessage)(void*,
                                                       Kitsunemimi::RingBuffer*,
                                                       AbstractSocket*));

    virtual bool initConnection(Kitsunemimi::ErrorContainer &error) = 0;
    virtual bool isClientSide() = 0;
    virtual uint32_t getType() = 0;
    virtual bool sendMessage(const std::string &message, ErrorContainer &error) = 0;
    virtual bool sendMessage(const void* message,
                             const uint64_t numberOfBytes,
                             ErrorContainer &error) = 0;
    virtual bool closeSocket() = 0;

protected:
    // callback-parameter
    void* m_target = nullptr;
    uint64_t (*m_processMessage)(void*, RingBuffer*, AbstractSocket*);
};

}

#endif // KITSUNEMIMI_NETWORK_ABSTRACT_SOCKET_H
