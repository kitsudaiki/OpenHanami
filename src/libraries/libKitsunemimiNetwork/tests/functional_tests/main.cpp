/**
 *  @file    main.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <libKitsunemimiNetwork/tcp/tcp_test.h>
#include <libKitsunemimiNetwork/unix/unix_domain_test.h>
#include <libKitsunemimiNetwork/tls_tcp/tls_tcp_test.h>

#include <libKitsunemimiCommon/logger.h>

int main()
{
    Kitsunemimi::initConsoleLogger(true);

    Kitsunemimi::UnixDomain_Test();
    Kitsunemimi::Tcp_Test();
    Kitsunemimi::TlsTcp_Test();
}
