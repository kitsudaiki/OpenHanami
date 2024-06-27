/**
 *  @file    unix_domain_socket_unix_domain_server_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "unix_domain_test.h"

#include <hanami_common/buffer/ring_buffer.h>
#include <template_server.h>
#include <template_socket.h>

namespace Hanami
{

/**
 * processMessageUnixDomain-callback
 */
uint64_t
processMessageUnixDomain(void* target, Hanami::RingBuffer* recvBuffer, AbstractSocket*)
{
    UnixDomain_Test* targetTest = static_cast<UnixDomain_Test*>(target);
    const uint8_t* dataPointer = getDataPointer_RingBuffer(*recvBuffer, recvBuffer->usedSize);
    if (dataPointer == nullptr) {
        return 0;
    }

    addData_DataBuffer(*targetTest->m_buffer, dataPointer, recvBuffer->usedSize);
    return recvBuffer->usedSize;
}

/**
 * processConnectionUnixDomain-callback
 */
void
processConnectionUnixDomain(void* target, AbstractSocket* socket)
{
    UnixDomain_Test* targetTest = static_cast<UnixDomain_Test*>(target);
    targetTest->m_socketServerSide = socket;
    socket->setMessageCallback(target, &processMessageUnixDomain);
    socket->startThread();
}

UnixDomain_Test::UnixDomain_Test() : Hanami::MemoryLeakTestHelpter("UnixDomain_Test")
{
    ErrorContainer* error = nullptr;

    // init for one-time-allocations
    error = new ErrorContainer();

    UnixDomainServer udsServer2("/tmp/sock.uds");
    m_server = new TemplateServer<UnixDomainServer>(
        std::move(udsServer2), this, &processConnectionUnixDomain, "UnixDomain_Test");
    m_server->initServer(*error);
    m_server->scheduleThreadForDeletion();
    sleep(2);

    // create new test-server
    REINIT_TEST();
    m_buffer = new DataBuffer(1000);
    error = new ErrorContainer();
    UnixDomainServer udsServer("/tmp/sock.uds");
    m_server = new TemplateServer<UnixDomainServer>(
        std::move(udsServer), this, &processConnectionUnixDomain, "UnixDomain_Test");
    m_server->initServer(*error);
    m_server->startThread();

    // test client create and delete
    UnixDomainSocket udsSocket("/tmp/sock.uds");
    m_socketClientSide
        = new TemplateSocket<UnixDomainSocket>(std::move(udsSocket), "UnixDomain_Test_client");
    m_socketClientSide->initConnection(*error);

    sleep(2);

    // send messages
    std::string sendMessage("poipoipoi");
    m_socketClientSide->sendMessage(sendMessage, *error);
    usleep(100000);

    std::string sendMessage2 = "poi";
    m_socketClientSide->sendMessage(sendMessage2, *error);
    for (uint32_t i = 0; i < 99999; i++) {
        m_socketClientSide->sendMessage(sendMessage2, *error);
    }

    m_socketServerSide->closeSocket();
    m_socketServerSide->scheduleThreadForDeletion();
    m_socketClientSide->closeSocket();
    m_socketClientSide->scheduleThreadForDeletion();
    sleep(2);

    // clear test-server
    m_server->closeServer();
    m_server->scheduleThreadForDeletion();
    sleep(2);
    delete m_buffer;
    delete error;
    CHECK_MEMORY();
}

}  // namespace Hanami
