/**
 *  @file    abstract_server.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef KITSUNEMIMI_NETWORK_ABSTRACT_SERVER_H
#define KITSUNEMIMI_NETWORK_ABSTRACT_SERVER_H

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <hanami_common/threading/thread.h>
#include <hanami_common/logger.h>

namespace Hanami
{
struct RingBuffer;

class AbstractSocket;

class AbstractServer
        : public Hanami::Thread
{
public:
    AbstractServer(void* target,
                   void (*processConnection)(void*, AbstractSocket*),
                   const std::string &threadName);
    ~AbstractServer();

    virtual bool closeServer() = 0;

protected:
    // callback-parameter
    void* m_target = nullptr;
    void (*m_processConnection)(void*, AbstractSocket*);
};

}

#endif // KITSUNEMIMI_NETWORK_ABSTRACT_SERVER_H
