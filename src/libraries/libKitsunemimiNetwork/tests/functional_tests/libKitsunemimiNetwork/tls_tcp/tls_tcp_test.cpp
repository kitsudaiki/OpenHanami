/**
 *  @file    tls_tcp_socket_tls_tcp_server_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "tls_tcp_test.h"
#include <libKitsunemimiCommon/buffer/ring_buffer.h>

#include <libKitsunemimiNetwork/template_socket.h>
#include <libKitsunemimiNetwork/template_server.h>

#include <cert_init.h>

namespace Kitsunemimi
{

/**
 * processMessageTlsTcp-callback
 */
uint64_t processMessageTlsTcp(void* target,
                              Kitsunemimi::RingBuffer* recvBuffer,
                              AbstractSocket*)
{
    TlsTcp_Test* targetTest = static_cast<TlsTcp_Test*>(target);
    const uint8_t* dataPointer = getDataPointer_RingBuffer(*recvBuffer, recvBuffer->usedSize);
    if(dataPointer == nullptr) {
        return 0;
    }

    addData_DataBuffer(*targetTest->m_buffer, dataPointer, recvBuffer->usedSize);
    return recvBuffer->usedSize;
}

/**
 * processConnectionTlsTcp-callback
 */
void processConnectionTlsTcp(void* target,
                             AbstractSocket* socket)
{
    TlsTcp_Test* targetTest = static_cast<TlsTcp_Test*>(target);
    targetTest->m_socketServerSide = socket;
    socket->setMessageCallback(target, &processMessageTlsTcp);
    socket->startThread();
}


TlsTcp_Test::TlsTcp_Test() :
    Kitsunemimi::CompareTestHelper("TlsTcp_Test")
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
TlsTcp_Test::initTestCase()
{
    writeTestCerts();

    m_buffer = new DataBuffer(1000);
}

/**
 * checkConnectionInit
 */
void
TlsTcp_Test::checkConnectionInit()
{
    ErrorContainer error;

    // init server
    TcpServer tcpServer(12345);

    TlsTcpServer tlsTcpServer(std::move(tcpServer),
                              "/tmp/cert.pem",
                              "/tmp/key.pem");
    m_server = new TemplateServer<TlsTcpServer>(std::move(tlsTcpServer),
                                                this,
                                                &processConnectionTlsTcp,
                                                "TlsTcp_Test");

    TEST_EQUAL(m_server->initServer(error), true);
    TEST_EQUAL(m_server->getType(), 3);
    TEST_EQUAL(m_server->startThread(), true);

    // init client
    TcpSocket tcpSocket("127.0.0.1", 12345);
    TlsTcpSocket tlsTcpSocket(std::move(tcpSocket),
                              "/tmp/cert.pem",
                              "/tmp/key.pem");
    m_socketClientSide = new TemplateSocket<TlsTcpSocket>(std::move(tlsTcpSocket),
                                                          "Tcp_Test_client");
    TEST_EQUAL(m_socketClientSide->initConnection(error), true);
    TEST_EQUAL(m_socketClientSide->getType(), 3);

    usleep(100000);
}

/**
 * checkLittleDataTransfer
 */
void
TlsTcp_Test::checkLittleDataTransfer()
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
TlsTcp_Test::checkBigDataTransfer()
{
    ErrorContainer error;

    std::string sendMessage = "poi";
    TEST_EQUAL(m_socketClientSide->sendMessage(sendMessage, error), true);

    for(uint32_t i = 0; i < 99999; i++) {
        m_socketClientSide->sendMessage(sendMessage, error);
    }

    usleep(1000000);

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
TlsTcp_Test::cleanupTestCase()
{
    TEST_EQUAL(m_socketServerSide->closeSocket(), true);
    TEST_EQUAL(m_server->closeServer(), true);
    TEST_EQUAL(m_socketServerSide->scheduleThreadForDeletion(), true);
    TEST_EQUAL(m_server->scheduleThreadForDeletion(), true);

    delete m_buffer;
}

}
