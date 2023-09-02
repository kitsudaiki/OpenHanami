/**
 * @file    main.cpp
 *
 * @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

#include <hanami_common/logger.h>

#include <session_test.h>
#include <tcp/tcp_test.h>
#include <unix/unix_domain_test.h>
#include <tls_tcp/tls_tcp_test.h>

int main()
{
    Kitsunemimi::initConsoleLogger(true);

    Kitsunemimi::UnixDomain_Test();
    Kitsunemimi::Tcp_Test();
    Kitsunemimi::TlsTcp_Test();
    Kitsunemimi::Sakura::Session_Test();
}
