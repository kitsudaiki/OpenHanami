/**
 * @file        create_cluster_template.h
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

#ifndef HANAMI_CREATECLUSTER_H
#define HANAMI_CREATECLUSTER_H

#include <api/endpoint_processing/blossom.h>

#include <libKitsunemimiHanamiCommon/structs.h>
#include <libKitsunemimiHanamiClusterParser/cluster_meta.h>
#include <libKitsunemimiHanamiSegmentParser/segment_meta.h>

class Cluster;

using Kitsunemimi::Hanami::SegmentMeta;
using Kitsunemimi::Hanami::SegmentMetaPtr;
using Kitsunemimi::Hanami::BrickMeta;

class CreateCluster
        : public Blossom
{
public:
    CreateCluster();

protected:
    bool runTask(BlossomIO &blossomIO,
                 const Kitsunemimi::DataMap &context,
                 Kitsunemimi::Hanami::BlossomStatus &status,
                 Kitsunemimi::ErrorContainer &error);

private:
    bool initCluster(Cluster* cluster,
                     const std::string &clusterUuid,
                     Kitsunemimi::Hanami::ClusterMeta &clusterDefinition,
                     const Kitsunemimi::Hanami::UserContext &userContext,
                     Kitsunemimi::Hanami::BlossomStatus &status,
                     Kitsunemimi::ErrorContainer &error);

    bool getSegmentTemplate(Kitsunemimi::Hanami::SegmentMeta* segmentMeta,
                            const std::string &name,
                            const Kitsunemimi::Hanami::UserContext &userContext,
                            Kitsunemimi::ErrorContainer &error);

    bool checkConnections(Kitsunemimi::Hanami::ClusterMeta &clusterTemplate,
                          std::map<std::string, SegmentMeta> &segmentTemplates,
                          Kitsunemimi::Hanami::BlossomStatus &status,
                          Kitsunemimi::ErrorContainer &error);
};

#endif // HANAMI_CREATECLUSTER_H
