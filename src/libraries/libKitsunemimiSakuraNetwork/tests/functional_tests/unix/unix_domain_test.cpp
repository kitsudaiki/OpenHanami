/**
 *  @file    unix_domain_socket_unix_domain_server_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "unix_domain_test.h"
#include <libKitsunemimiCommon/buffer/ring_buffer.h>

#include <template_socket.h>
#include <template_server.h>

namespace Kitsunemimi
{

/**
 * processMessageUnixDomain-callback
 */
uint64_t processMessageUnixDomain(void* target,
                                  Kitsunemimi::RingBuffer* recvBuffer,
                                  AbstractSocket*)
{
    UnixDomain_Test* targetTest = static_cast<UnixDomain_Test*>(target);
    const uint8_t* dataPointer = getDataPointer_RingBuffer(*recvBuffer, recvBuffer->usedSize);
    if(dataPointer == nullptr) {
        return 0;
    }

    addData_DataBuffer(*targetTest->m_buffer, dataPointer, recvBuffer->usedSize);
    return recvBuffer->usedSize;
}

/**
 * processConnectionUnixDomain-callback
 */
void processConnectionUnixDomain(void* target,
                                 AbstractSocket* socket)
{
    UnixDomain_Test* targetTest = static_cast<UnixDomain_Test*>(target);
    targetTest->m_socketServerSide = socket;
    socket->setMessageCallback(target, &processMessageUnixDomain);
    socket->startThread();
}


UnixDomain_Test::UnixDomain_Test() :
    Kitsunemimi::CompareTestHelper("UnixDomain_Test")
{
    initTestCase();
    checkConnectionInit();
    checkLittleDataTransfer();
    checkBigDataTransfer();
    cleanupTestCase();
}

/**
 * initTestCase
 */
void
UnixDomain_Test::initTestCase()
{
    m_buffer = new DataBuffer(1000);
}

/**
 * checkConnectionInit
 */
void
UnixDomain_Test::checkConnectionInit()
{
    ErrorContainer error;
    // check too long path
    UnixDomainServer udsServer("/tmp/sock.uds");
    m_server = new TemplateServer<UnixDomainServer>(std::move(udsServer),
                                                    this,
                                                    &processConnectionUnixDomain,
                                                    "UnixDomain_Test");

    // init server
    TEST_EQUAL(m_server->initServer(error), true);
    TEST_EQUAL(m_server->getType(), 1);
    TEST_EQUAL(m_server->startThread(), true);

    usleep(100000);

    // init client
    UnixDomainSocket udsSocket("/tmp/sock.uds");
    m_socketClientSide = new TemplateSocket<UnixDomainSocket>(std::move(udsSocket),
                                                              "UnixDomain_Test_client");
    TEST_EQUAL(m_socketClientSide->initConnection(error), true);
    TEST_EQUAL(m_socketClientSide->initConnection(error), true);
    TEST_EQUAL(m_socketClientSide->getType(), 1);

    usleep(100000);
}

/**
 * checkLittleDataTransfer
 */
void
UnixDomain_Test::checkLittleDataTransfer()
{
    usleep(100000);
    ErrorContainer error;

    std::string sendMessage("poipoipoi");
    TEST_EQUAL(m_socketClientSide->sendMessage(sendMessage, error), true);
    usleep(100000);
    TEST_EQUAL(m_buffer->usedBufferSize, 9);

    if(m_buffer->usedBufferSize == 9)
    {
        DataBuffer* buffer = m_buffer;
        uint64_t bufferSize = buffer->usedBufferSize;
        char recvMessage[bufferSize];
        memcpy(recvMessage, buffer->data, bufferSize);
        TEST_EQUAL(bufferSize, 9);
        TEST_EQUAL(recvMessage[2], sendMessage.at(2));
        reset_DataBuffer(*m_buffer, 1000);
    }
}

/**
 * checkBigDataTransfer
 */
void
UnixDomain_Test::checkBigDataTransfer()
{
    ErrorContainer error;

    std::string sendMessage = "poi";
    TEST_EQUAL(m_socketClientSide->sendMessage(sendMessage, error), true);
    for(uint32_t i = 0; i < 99999; i++) {
        m_socketClientSide->sendMessage(sendMessage, error);
    }

    usleep(10000);
    uint64_t totalIncom = m_buffer->usedBufferSize;
    DataBuffer* dataBuffer = m_buffer;
    TEST_EQUAL(totalIncom, 300000);
    TEST_EQUAL(dataBuffer->usedBufferSize, 300000);

    uint32_t numberOfPois = 0;
    for(uint32_t i = 0; i < 300000; i=i+3)
    {
        uint8_t* dataBufferData = static_cast<uint8_t*>(dataBuffer->data);
        if(dataBufferData[i] == 'p'
                && dataBufferData[i+1] == 'o'
                && dataBufferData[i+2] == 'i')
        {
            numberOfPois++;
        }
    }

    TEST_EQUAL(numberOfPois, 100000);
}

/**
 * cleanupTestCase
 */
void
UnixDomain_Test::cleanupTestCase()
{
    TEST_EQUAL(m_socketServerSide->closeSocket(), true);
    TEST_EQUAL(m_server->closeServer(), true);
    TEST_EQUAL(m_socketServerSide->scheduleThreadForDeletion(), true);
    TEST_EQUAL(m_server->scheduleThreadForDeletion(), true);

    delete m_buffer;
}

}
