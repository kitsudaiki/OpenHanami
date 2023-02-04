/**
 * @file        azuki_input.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
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

#include <libAzukiHeart/azuki_input.h>
#include <get_thread_mapping.h>
#include <bind_thread_to_core.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

namespace Azuki
{

/**
 * @brief HanamiMessaging::initAzukiBlossoms
 * @return
 */
bool
initAzukiBlossoms()
{
    // init predefined blossoms
    Kitsunemimi::Hanami::HanamiMessaging * interface =
            Kitsunemimi::Hanami::HanamiMessaging::getInstance();
    const std::string group = "-";

    if(interface->addBlossom(group, "get_thread_mapping", new GetThreadMapping()) == false) {
        return false;
    }

    // add new endpoints
    if(interface->addEndpoint("v1/get_thread_mapping",
                              Kitsunemimi::Hanami::GET_TYPE,
                              Kitsunemimi::Hanami::BLOSSOM_TYPE,
                              "-",
                              "get_thread_mapping") == false)
    {
        return false;
    }

    if(interface->addBlossom(group, "bind_thread_to_core", new BindThreadToCore()) == false) {
        return false;
    }

    // add new endpoints
    if(interface->addEndpoint("v1/bind_thread_to_core",
                              Kitsunemimi::Hanami::POST_TYPE,
                              Kitsunemimi::Hanami::BLOSSOM_TYPE,
                              "-",
                              "bind_thread_to_core") == false)
    {
        return false;
    }

    return true;
}

}
