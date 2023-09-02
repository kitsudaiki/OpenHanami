/**
 *  @file    tls_tcp_socket_tls_tcp_server_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef TLSTCPSOCKET_TLSTCPSERVER_TEST_H
#define TLSTCPSOCKET_TLSTCPSERVER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{
struct DataBuffer;

class AbstractSocket;

class TcpServer;
class TcpSocket;

class TlsTcpServer;
class TlsTcpSocket;

template<class>
class TemplateSocket;

template<class>
class TemplateServer;

class TlsTcp_Test
        : public Hanami::CompareTestHelper
{
public:
    TlsTcp_Test();

    DataBuffer* m_buffer = nullptr;
    AbstractSocket* m_socketServerSide = nullptr;

private:
    void initTestCase();
    void checkConnectionInit();
    void checkLittleDataTransfer();
    void checkBigDataTransfer();
    void cleanupTestCase();

    TemplateServer<TlsTcpServer>* m_server = nullptr;
    TemplateSocket<TlsTcpSocket>* m_socketClientSide = nullptr;
};

}

#endif // TLSTCPSOCKET_TLSTCPSERVER_TEST_H
