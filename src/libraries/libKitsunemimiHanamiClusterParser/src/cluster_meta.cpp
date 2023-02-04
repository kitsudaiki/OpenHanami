/**
 * @file       cluster_meta.cpp
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
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

#include <libKitsunemimiHanamiClusterParser/cluster_meta.h>
#include <cluster_parsing/cluster_parser_interface.h>

namespace Kitsunemimi
{
namespace Hanami
{

/**
 * @brief convert a cluster-formated string into a cluster-object-tree
 *
 * @param input cluster-formated string, which should be parsed
 * @param error reference for error-message output
 *
 * @return true, if successful, else false
 */
bool
parseCluster(ClusterMeta* result,
             const std::string &input,
             ErrorContainer &error)
{
    ClusterParserInterface* parser = ClusterParserInterface::getInstance();

    if(input.size() == 0)
    {
        // TODO: error-message
        return false;
    }

    return parser->parse(result, input, error);
}

}  // namespace Hanami
}  // namespace Kitsunemimi
