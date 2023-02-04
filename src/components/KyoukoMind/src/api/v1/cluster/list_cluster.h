/**
 * @file        list_cluster.h
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

#ifndef KYOUKOMIND_SHOWCLUSTERS_H
#define KYOUKOMIND_SHOWCLUSTERS_H

#include <libKitsunemimiHanamiNetwork/blossom.h>

class ListCluster
        : public Kitsunemimi::Hanami::Blossom
{
public:
    ListCluster();

protected:
    bool runTask(Kitsunemimi::Hanami::BlossomIO &blossomIO,
                 const Kitsunemimi::DataMap &context,
                 Kitsunemimi::Hanami::BlossomStatus &status,
                 Kitsunemimi::ErrorContainer &error);
};

#endif // KYOUKOMIND_SHOWCLUSTERS_H
