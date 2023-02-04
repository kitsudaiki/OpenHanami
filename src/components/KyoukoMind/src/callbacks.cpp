/**
 * @file        callbacks.cpp
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

#include <callbacks.h>
#include <libKitsunemimiSakuraNetwork/session.h>

#include <io/protobuf_messages.h>

class Cluster;

/**
 * @brief process stream-message
 *
 * @param target target-cluster of the session-endpoint
 * @param data incoming data
 * @param dataSize number of incoming data
 */
void
streamDataCallback(void* target,
                   Kitsunemimi::Sakura::Session*,
                   const void* data,
                   const uint64_t dataSize)
{
    Cluster* cluster = static_cast<Cluster*>(target);
    recvClusterInputMessage(cluster, data, dataSize);

    /**std::cout<<"#################################################"<<std::endl;
    std::cout<<"number of values: "<<msg.numberOfValues<<std::endl;
    std::cout<<"val0: "<<msg.values[0]<<std::endl;
    std::cout<<"val1: "<<msg.values[1]<<std::endl;
    std::cout<<"#################################################"<<std::endl;*/
}
