/**
 * @file        create_cluster_template.cpp
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

#include "create_cluster.h"

#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster.h>

#include <libKitsunemimiHanamiCommon/uuid.h>
#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/structs.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiJson/json_item.h>

#include <kyouko_root.h>

using namespace Kitsunemimi::Hanami;

CreateCluster::CreateCluster()
    : Blossom("Create new cluster.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("name",
                       SAKURA_STRING_TYPE,
                       true,
                       "Name for the new cluster.");
    assert(addFieldBorder("name", 4, 256));
    assert(addFieldRegex("name", NAME_REGEX));

    registerInputField("template",
                       SAKURA_STRING_TYPE,
                       false,
                       "Cluster-template as base64-string.");

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the new created cluster.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the new created cluster.");
    registerOutputField("owner_id",
                        SAKURA_STRING_TYPE,
                        "ID of the user, who created the new cluster.");
    registerOutputField("project_id",
                        SAKURA_STRING_TYPE,
                        "ID of the project, where the new cluster belongs to.");
    registerOutputField("visibility",
                        SAKURA_STRING_TYPE,
                        "Visibility of the new created cluster (private, shared, public).");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CreateCluster::runTask(BlossomIO &blossomIO,
                       const Kitsunemimi::DataMap &context,
                       BlossomStatus &status,
                       Kitsunemimi::ErrorContainer &error)
{
    const std::string clusterName = blossomIO.input.get("name").getString();
    const std::string base64Template = blossomIO.input.get("template").getString();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    // check if user already exist within the table
    Kitsunemimi::JsonItem getResult;
    if(KyoukoRoot::clustersTable->getClusterByName(getResult, clusterName, userContext, error))
    {
        status.errorMessage = "Cluster with name '" + clusterName + "' already exist.";
        error.addMeesage(status.errorMessage);
        status.statusCode = Kitsunemimi::Hanami::CONFLICT_RTYPE;
        return false;
    }
    error._errorMessages.clear();
    error._possibleSolution.clear();

    Kitsunemimi::Hanami::ClusterMeta parsedCluster;
    if(base64Template != "")
    {
        // decode base64 formated template to check if valid base64-string
        Kitsunemimi::DataBuffer convertedTemplate;
        if(Kitsunemimi::decodeBase64(convertedTemplate, base64Template) == false)
        {
            status.errorMessage = "Uploaded template is not a valid base64-String.";
            status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
            error.addMeesage(status.errorMessage);
            return false;
        }

        // parse segment-template to validate syntax
        const std::string convertedTemplateStr(static_cast<const char*>(convertedTemplate.data),
                                               convertedTemplate.usedBufferSize);
        if(Kitsunemimi::Hanami::parseCluster(&parsedCluster, convertedTemplateStr, error) == false)
        {
            status.errorMessage = "Uploaded template is not a valid cluster-template: \n";
            status.errorMessage += error.toString();
            status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
            error.addMeesage(status.errorMessage);
            return false;
        }
    }

    // convert values
    Kitsunemimi::JsonItem clusterData;
    clusterData.insert("name", clusterName);
    clusterData.insert("project_id", userContext.projectId);
    clusterData.insert("owner_id", userContext.userId);
    clusterData.insert("visibility", "private");

    // add new user to table
    if(KyoukoRoot::clustersTable->addCluster(clusterData, userContext, error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Failed to add cluster to database");
        return false;
    }

    // get new created user from database
    if(KyoukoRoot::clustersTable->getClusterByName(blossomIO.output,
                                                   clusterName,
                                                   userContext,
                                                   error) == false)
    {
        error.addMeesage("Failed to get cluster from database by name '" + clusterName + "'");
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    const std::string uuid = blossomIO.output.get("uuid").getString();

    // create new cluster
    Cluster* newCluster = new Cluster();
    if(base64Template != "")
    {
        if(initCluster(newCluster, uuid, parsedCluster, userContext, status, error) == false)
        {
            delete newCluster;
            error.addMeesage("Failed to initialize cluster");
            KyoukoRoot::clustersTable->deleteCluster(uuid, userContext, error);
            return false;
        }
    }

    KyoukoRoot::m_clusterHandler->addCluster(uuid, newCluster);

    // remove irrelevant fields
    blossomIO.output.remove("owner_id");
    blossomIO.output.remove("project_id");
    blossomIO.output.remove("visibility");

    return true;
}

/**
 * @brief CreateCluster::initCluster
 *
 * @param cluster pointer to the cluster, which should be initialized
 * @param clusterUuid uuid of the cluster
 * @param clusterDefinition definition, which describe the new cluster
 * @param userContext context-object with date for the access to the database-tables
 * @param status reference for status-output
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CreateCluster::initCluster(Cluster* cluster,
                           const std::string &clusterUuid,
                           Kitsunemimi::Hanami::ClusterMeta &clusterDefinition,
                           const Kitsunemimi::Hanami::UserContext &userContext,
                           Kitsunemimi::Hanami::BlossomStatus &status,
                           Kitsunemimi::ErrorContainer &error)
{
    // collect all segment-templates, which are required by the cluster-template
    std::map<std::string, Kitsunemimi::Hanami::SegmentMeta> segmentTemplates;
    for(const Kitsunemimi::Hanami::SegmentMetaPtr& segmentCon : clusterDefinition.segments)
    {
        // skip input- and output-segments, because they are generated anyway
        if(segmentCon.type == "input"
                || segmentCon.type == "output")
        {
            continue;
        }

        // get the content of the segment-template
        Kitsunemimi::Hanami::SegmentMeta segmentMeta;
        if(getSegmentTemplate(&segmentMeta, segmentCon.type, userContext, error) == false)
        {
            status.errorMessage = "Failed to get segment-template with name '"
                                  + segmentCon.type
                                  + "'";
            status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
            error.addMeesage(status.errorMessage);
            return false;
        }

        // add segment-template to a map, which is generated later when creating the segments
        // based on these templates
        segmentTemplates.emplace(segmentCon.type, segmentMeta);
    }

    // check if all connections within the cluster-definition are valid
    if(checkConnections(clusterDefinition, segmentTemplates, status, error) == false)
    {
        error.addMeesage("Validation of the connections within the cluster-definition failed");
        return false;
    }

    // generate and initialize the cluster based on the cluster- and segment-templates
    if(cluster->init(clusterDefinition, segmentTemplates, clusterUuid) == false)
    {
        error.addMeesage("Failed to initialize cluster based on a template");
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}

/**
 * @brief Request a segment-template from the database and convert and parse the content
 *
 * @param segmentMeta pointer of the output of the result
 * @param name name of the segment-template, which should be loaded from the database
 * @param userContext user-context for filtering database-access
 * @param error reference for internal error-output
 *
 * @return true, if successful, else false
 */
bool
CreateCluster::getSegmentTemplate(Kitsunemimi::Hanami::SegmentMeta* segmentMeta,
                                  const std::string &name,
                                  const Kitsunemimi::Hanami::UserContext &userContext,
                                  Kitsunemimi::ErrorContainer &error)
{
    // get segment-template from database
    Kitsunemimi::JsonItem templateData;
    if(KyoukoRoot::templateTable->getTemplateByName(templateData,
                                                    name,
                                                    userContext,
                                                    error,
                                                    true) == false)
    {
        return false;
    }

    // decode template
    std::string decodedTemplate = "";
    if(Kitsunemimi::decodeBase64(decodedTemplate,
                                 templateData.get("data").getString()) == false)
    {
        // TODO: better error-messages with uuid
        error.addMeesage("base64-decoding of the template failes");
        return false;
    }

    // parse segment-template
    if(Kitsunemimi::Hanami::parseSegment(segmentMeta, decodedTemplate, error) == false)
    {
        error.addMeesage("Failed to parse decoded segment-template");
        return false;
    }

    return true;
}

/**
 * @brief Check all connections of the cluster-definition and initialize input- and output-segments
 *
 * @param clusterTemplate template with the cluster-definition
 * @param segmentTemplates list with all required segment-templates for the cluster
 * @param status reference for status-output in case of an error
 * @param error reference for internal error-output
 *
 * @return true, if successful, else false
 */
bool
CreateCluster::checkConnections(Kitsunemimi::Hanami::ClusterMeta &clusterTemplate,
                                std::map<std::string, SegmentMeta> &segmentTemplates,
                                Kitsunemimi::Hanami::BlossomStatus &status,
                                Kitsunemimi::ErrorContainer &error)
{
    for(Kitsunemimi::Hanami::SegmentMetaPtr &sourceSegmentPtr : clusterTemplate.segments)
    {
        for(Kitsunemimi::Hanami::ClusterConnection &conn : sourceSegmentPtr.outputs)
        {
            // skip output-segments, because they have not outgoing connections
            if(sourceSegmentPtr.type == "output") {
                continue;
            }

            // get segment-meta-data of the target-segment
            SegmentMetaPtr* targetSegmentPtr = clusterTemplate.getSegmentMetaPtr(conn.targetSegment);
            if(targetSegmentPtr == nullptr)
            {
                status.errorMessage = "Segment-template with name '"
                                      + conn.targetSegment
                                      + "' not found.";
                status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
                error.addMeesage(status.errorMessage);
                return false;
            }

            // check that input is not directly connected to output
            if(sourceSegmentPtr.type == "input"
                    && targetSegmentPtr->type == "output")
            {
                status.errorMessage = "Input- and Output-segments are not allowed to be directly "
                                      "connected with each other.";
                status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
                error.addMeesage(status.errorMessage);
                return false;
            }

            if(targetSegmentPtr->type != "output")
            {
                // get segment-meta-data of the target-segment
                std::map<std::string, SegmentMeta>::iterator targetSegmentIt;
                targetSegmentIt = segmentTemplates.find(targetSegmentPtr->type);
                if(targetSegmentIt == segmentTemplates.end())
                {
                    status.errorMessage = "Segment-template with name '"
                                          + targetSegmentPtr->type
                                          + "' not found.";
                    status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
                    error.addMeesage(status.errorMessage);
                    return false;
                }

                // get target-brick of the target-segment
                BrickMeta* brickMeta = targetSegmentIt->second.getBrick(conn.targetBrick);
                if(brickMeta == nullptr)
                {
                    status.errorMessage = "Segment-template with name '"
                                          + targetSegmentPtr->type
                                          + "' has no brick with name '"
                                          + conn.targetBrick
                                          + "'";
                    status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
                    error.addMeesage(status.errorMessage);
                    return false;
                }

                // if source of the connection is an input-segment, then create a new segment-meta-
                // object for this input with the number of inputs depending on the target-brick
                if(sourceSegmentPtr.type == "input")
                {
                    BrickMeta inputBrick;
                    inputBrick.numberOfNeurons = brickMeta->numberOfNeurons;
                    SegmentMeta inputSegment;
                    inputSegment.bricks.push_back(inputBrick);
                    // TODO: check if name already exist
                    segmentTemplates.emplace(sourceSegmentPtr.name, inputSegment);
                }
            }
            else
            {
                // get segment-meta-data of the source-segment
                std::map<std::string, SegmentMeta>::iterator sourceSegmentIt;
                sourceSegmentIt = segmentTemplates.find(sourceSegmentPtr.type);
                if(sourceSegmentIt == segmentTemplates.end())
                {
                    status.errorMessage = "Segment-template with name '"
                                          + sourceSegmentPtr.type
                                          + "' not found.";
                    status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
                    error.addMeesage(status.errorMessage);
                    return false;
                }

                // get source-brick of the source-segment
                BrickMeta* brickMeta = sourceSegmentIt->second.getBrick(conn.sourceBrick);
                if(brickMeta == nullptr)
                {
                    status.errorMessage = "Segment-template with name '"
                                          + sourceSegmentPtr.type
                                          + "' has no brick with name '"
                                          + conn.sourceBrick
                                          + "'";
                    status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
                    error.addMeesage(status.errorMessage);
                    return false;
                }

                // if target of the connection is an output-segment, then create a new segment-meta-
                // object for this output with the number of outputs depending on the source-brick
                Kitsunemimi::Hanami::BrickMeta outputBrick;
                outputBrick.numberOfNeurons = brickMeta->numberOfNeurons;
                SegmentMeta inputSegment;
                inputSegment.bricks.push_back(outputBrick);
                // TODO: check if name already exist
                segmentTemplates.emplace(targetSegmentPtr->name, inputSegment);
            }

        }
    }

    return true;
}
