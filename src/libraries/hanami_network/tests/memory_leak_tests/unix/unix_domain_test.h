/**
 *  @file    unix_domain_socket_unix_domain_server_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef UNIX_DOMAIN_TEST_H
#define UNIX_DOMAIN_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{
struct DataBuffer;

class AbstractSocket;

class UnixDomainServer;
class UnixDomainSocket;

template<class>
class TemplateSocket;

template<class>
class TemplateServer;

class UnixDomain_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    UnixDomain_Test();

    DataBuffer* m_buffer = nullptr;
    AbstractSocket* m_socketServerSide = nullptr;

private:
    void initTestCase();
    void checkConnectionInit();
    void checkLittleDataTransfer();
    void checkBigDataTransfer();
    void cleanupTestCase();

    TemplateServer<UnixDomainServer>* m_server = nullptr;
    TemplateSocket<UnixDomainSocket>* m_socketClientSide = nullptr;
};

}

#endif // UNIX_DOMAIN_TEST_H
