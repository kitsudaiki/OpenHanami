/**
 * @file        main.cpp
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

#include <iostream>

#include <libKitsunemimiSakuraHardware/host.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/logger.h>

void
handleErrorCallback(const std::string &errorMessage)
{
}


int main()
{
    Kitsunemimi::initConsoleLogger(true);
    Kitsunemimi::setErrorLogCallback(&handleErrorCallback);

    Kitsunemimi::Sakura::Host* host = Kitsunemimi::Sakura::Host::getInstance();
    Kitsunemimi::ErrorContainer error;
    host->initHost(error);

    std::cout<<"wait for 10 seconds"<<std::endl;
    sleep(10);
    Kitsunemimi::JsonItem json;
    std::string errorMessage = "";
    const bool success = json.parse(host->toJsonString(), error);
    if(success == false)
    {
        std::cout<<"error: "<<errorMessage<<std::endl;
        return 1;
    }

    std::cout<<json.toString(true)<<std::endl;

    return 0;
}
