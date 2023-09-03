/**
 * @file        hanami_root.cpp
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

#include <hanami_root.h>

#include <core/processing/processing_unit_handler.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster.h>
#include <core/cluster/cluster_init.h>

#include <hanami_common/logger.h>
#include <hanami_common/files/text_file.h>
#include <hanami_config/config_handler.h>

#include <api/endpoint_processing/http_server.h>
#include <api/endpoint_processing/http_websocket_thread.h>
#include <api/endpoint_processing/blossom.h>
#include <api/endpoint_processing/items/item_methods.h>

#include <hanami_hardware/power_measuring.h>
#include <hanami_hardware/speed_measuring.h>
#include <hanami_hardware/temperature_measuring.h>

#include <hanami_hardware/host.h>
#include <hanami_database/sql_database.h>

#include <database/cluster_table.h>
#include <database/users_table.h>
#include <database/projects_table.h>
#include <database/data_set_table.h>
#include <database/checkpoint_table.h>
#include <database/request_result_table.h>
#include <database/error_log_table.h>
#include <database/audit_log_table.h>
#include <core/temp_file_handler.h>
#include <core/thread_binder.h>

// init static variables
uint32_t* HanamiRoot::m_randomValues = nullptr;
Hanami::GpuInterface* HanamiRoot::gpuInterface = nullptr;
HanamiRoot* HanamiRoot::root = nullptr;
HttpServer* HanamiRoot::httpServer = nullptr;
CryptoPP::SecByteBlock HanamiRoot::tokenKey{};

// static flag to switch to experimental gpu-support (see issue #44 and #76)
bool HanamiRoot::useCuda = false;

/**
 * @brief constructor
 */
HanamiRoot::HanamiRoot()
{
    root = this;
}

/**
 * @brief destructor
 */
HanamiRoot::~HanamiRoot() {}

/**
 * @brief init root-object
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiRoot::init(Hanami::ErrorContainer &error)
{
    /*if(useOpencl)
    {
        Hanami::GpuHandler oclHandler;
        assert(oclHandler.initDevice(error));
        assert(oclHandler.m_interfaces.size() == 1);
        gpuInterface = oclHandler.m_interfaces.at(0);
    }*/

    // init predefinde random-values
    m_randomValues = new uint32_t[NUMBER_OF_RAND_VALUES];
    srand(time(NULL));
    for(uint32_t i = 0; i < NUMBER_OF_RAND_VALUES; i++) {
        m_randomValues[i] = static_cast<uint32_t>(rand());
    }

    if(initDatabase(error) == false)
    {
        error.addMeesage("Failed to initialize database");
        return false;
    }

    clearCluster(error);

    if(initJwt(error) == false)
    {
        error.addMeesage("Failed to initialize jwt");
        return false;
    }

    if(initPolicies(error) == false)
    {
        error.addMeesage("Failed to initialize policies");
        return false;
    }

    if(initHttpServer() == false)
    {
        error.addMeesage("initializing http-server failed");
        LOG_ERROR(error);
        return false;
    }

    Hanami::Host* host = Hanami::Host::getInstance();
    if(host->initHost(error) == false)
    {
        error.addMeesage("Failed to initialize host-information.");
        LOG_ERROR(error);
        return false;
    }

    bool success = false;
    useCuda = GET_BOOL_CONFIG("DEFAULT", "use_cuda", success);
    assert(success);

    // create thread-binder
    if(ThreadBinder::getInstance()->init(error) == false)
    {
        error.addMeesage("failed to init thread-binder");
        return false;
    }
    ThreadBinder::getInstance()->startThread();

    // start monitoring
    PowerMeasuring::getInstance()->startThread();
    SpeedMeasuring::getInstance()->startThread();
    TemperatureMeasuring::getInstance()->startThread();

    return true;
}

/**
 * @brief create processing-threads
 *
 * @return true, if successful, else false
 */
bool
HanamiRoot::initThreads()
{
    ProcessingUnitHandler* processingUnitHandler = ProcessingUnitHandler::getInstance();
    if(processingUnitHandler->initProcessingUnits(1) == false) {
        return false;
    }

    return true;
}

/**
 * @brief init database
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiRoot::initDatabase(Hanami::ErrorContainer &error)
{
    bool success = false;

    // read database-path from config
    Hanami::SqlDatabase* database = Hanami::SqlDatabase::getInstance();
    const std::string databasePath = GET_STRING_CONFIG("DEFAULT", "database", success);
    LOG_DEBUG("database-path: '" + databasePath + "'");
    if(success == false)
    {
        error.addMeesage("No database-path defined in config.");
        LOG_ERROR(error);
        return false;
    }

    // initalize database
    if(database->initDatabase(databasePath, error) == false)
    {
        error.addMeesage("Failed to initialize sql-database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize cluster-table
    ClusterTable* clustersTable = ClusterTable::getInstance();
    if(clustersTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize cluster-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize projects-table
    ProjectsTable* projectsTable = ProjectsTable::getInstance();
    if(projectsTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize project-table in database.");
        return false;
    }

    // initialize users-table
    UsersTable* usersTable = UsersTable::getInstance();
    if(usersTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize user-table in database.");
        return false;
    }
    if(usersTable->initNewAdminUser(error) == false)
    {
        error.addMeesage("Failed to initialize new admin-user even this is necessary.");
        return false;
    }

    // initialize dataset-table
    DataSetTable* dataSetTable = DataSetTable::getInstance();
    if(dataSetTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize dataset-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize request-result-table
    RequestResultTable* requestResultTable = RequestResultTable::getInstance();
    if(requestResultTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize request-result-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize checkpoint-table
    CheckpointTable* clusterCheckpointTable = CheckpointTable::getInstance();
    if(clusterCheckpointTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize checkpoint-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize error-log-table
    ErrorLogTable* errorLogTable = ErrorLogTable::getInstance();
    if(errorLogTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize error-log-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize audit-log-table
    AuditLogTable* auditLogTable = AuditLogTable::getInstance();
    if(auditLogTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize audit-log-table in database.");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief initialze http server
 *
 * @return true, if successful, else false
 */
bool
HanamiRoot::initHttpServer()
{
    bool success = false;

    // check if http is enabled
    if(GET_BOOL_CONFIG("http", "enable", success) == false) {
        return true;
    }

    // get stuff from config
    const uint16_t port =            GET_INT_CONFIG(    "http", "port",              success);
    const std::string ip =           GET_STRING_CONFIG( "http", "ip",                success);
    const std::string cert =         GET_STRING_CONFIG( "http", "certificate",       success);
    const std::string key =          GET_STRING_CONFIG( "http", "key",               success);
    const uint32_t numberOfThreads = GET_INT_CONFIG(    "http", "number_of_threads", success);

    // create server
    httpServer = new HttpServer(ip, port, cert, key);
    httpServer->startThread();

    // start threads
    for(uint32_t i = 0; i < numberOfThreads; i++)
    {
        const std::string name = "HttpWebsocketThread";
        HttpWebsocketThread* httpWebsocketThread = new HttpWebsocketThread(name);
        httpWebsocketThread->startThread();
        m_threads.push_back(httpWebsocketThread);
    }

    return true;
}

/**
 * @brief init policies
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiRoot::initPolicies(Hanami::ErrorContainer &error)
{
    bool success = false;

    // read policy-file-path from config
    const std::string policyFilePath = GET_STRING_CONFIG("auth", "policies", success);
    if(success == false)
    {
        error.addMeesage("No policy-file defined in config.");
        return false;
    }

    // read policy-file
    std::string policyFileContent;
    if(Hanami::readFile(policyFileContent, policyFilePath, error) == false)
    {
        error.addMeesage("Failed to read policy-file");
        return false;
    }

    // parse policy-file
    Hanami::Policy* policies = Hanami::Policy::getInstance();
    if(policies->parse(policyFileContent, error) == false)
    {
        error.addMeesage("Failed to parser policy-file");
        return false;
    }

    return true;
}

/**
 * @brief init jwt-class to validate incoming requested
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiRoot::initJwt(Hanami::ErrorContainer &error)
{
    bool success = false;

    // read jwt-token-key from config
    const std::string tokenKeyPath = GET_STRING_CONFIG("auth", "token_key_path", success);
    if(success == false)
    {
        error.addMeesage("No token_key_path defined in config.");
        return false;
    }

    std::string tokenKeyString;
    if(Hanami::readFile(tokenKeyString, tokenKeyPath, error) == false)
    {
        error.addMeesage("Failed to read token-file '" + tokenKeyPath + "'");
        return false;
    }

    // init jwt for token create and sign
    tokenKey = CryptoPP::SecByteBlock((unsigned char*)tokenKeyString.c_str(), tokenKeyString.size());

    return true;
}

/**
 * @brief delete all clusters, because after a restart these are broken
 */
void
HanamiRoot::clearCluster(Hanami::ErrorContainer &error)
{
    ClusterTable::getInstance()->deleteAllCluster(error);
    // TODO: if a checkpoint exist for a broken cluster, then the cluster should be
    //       restored with the last available checkpoint
}

/**
 * @brief check if a specific blossom was registered
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 *
 * @return true, if blossom with the given group- and item-name exist, else false
 */
bool
HanamiRoot::doesBlossomExist(const std::string &groupName,
                             const std::string &itemName)
{
    auto groupIt = m_registeredBlossoms.find(groupName);
    if(groupIt != m_registeredBlossoms.end())
    {
        if(groupIt->second.find(itemName) != groupIt->second.end()) {
            return true;
        }
    }

    return false;
}

/**
 * @brief SakuraLangInterface::addBlossom
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 * @param newBlossom pointer to the new blossom
 *
 * @return true, if blossom was registered or false, if the group- and item-name are already
 *         registered
 */
bool
HanamiRoot::addBlossom(const std::string &groupName,
                       const std::string &itemName,
                       Blossom* newBlossom)
{
    // check if already used
    if(doesBlossomExist(groupName, itemName) == true) {
        return false;
    }

    // create internal group-map, if not already exist
    auto groupIt = m_registeredBlossoms.find(groupName);
    if(groupIt == m_registeredBlossoms.end())
    {
        std::map<std::string, Blossom*> newMap;
        m_registeredBlossoms.try_emplace(groupName, newMap);
    }

    // add item to group
    groupIt = m_registeredBlossoms.find(groupName);
    groupIt->second.try_emplace(itemName, newBlossom);

    return true;
}

/**
 * @brief request a registered blossom
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 *
 * @return pointer to the blossom or
 *         nullptr, if blossom the given group- and item-name was not found
 */
Blossom*
HanamiRoot::getBlossom(const std::string &groupName,
                       const std::string &itemName)
{
    // search for group
    auto groupIt = m_registeredBlossoms.find(groupName);
    if(groupIt != m_registeredBlossoms.end())
    {
        // search for item within group
        auto itemIt = groupIt->second.find(itemName);
        if(itemIt != groupIt->second.end()) {
            return itemIt->second;
        }
    }

    return nullptr;
}

/**
 * @brief trigger existing blossom
 *
 * @param result map with resulting items
 * @param blossomName id of the blossom to trigger
 * @param blossomGroupName id of the group of the blossom to trigger
 * @param initialValues input-values for the tree
 * @param status reference for status-output
 * @param error reference for error-output
 *
 * @return true, if successfule, else false
 */
bool
HanamiRoot::triggerBlossom(Hanami::DataMap &result,
                           const std::string &blossomName,
                           const std::string &blossomGroupName,
                           const Hanami::DataMap &context,
                           const Hanami::DataMap &initialValues,
                           BlossomStatus &status,
                           Hanami::ErrorContainer &error)
{
    LOG_DEBUG("trigger blossom");

    // get initial blossom-item
    Blossom* blossom = getBlossom(blossomGroupName, blossomName);
    if(blossom == nullptr)
    {
        error.addMeesage("No blosom found for the id " + blossomName);
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        LOG_ERROR(error);
        return false;
    }

    bool success = false;

    do
    {
        // inialize a new blossom-leaf for processing
        BlossomIO blossomIO;
        blossomIO.blossomName = blossomName;
        blossomIO.blossomPath = blossomName;
        blossomIO.blossomGroupType = blossomGroupName;
        blossomIO.input = &initialValues;
        blossomIO.parentValues = blossomIO.input.getItemContent()->toMap();
        blossomIO.nameHirarchie.push_back("BLOSSOM: " + blossomName);

        // check input to be complete
        std::string errorMessage;
        if(blossom->validateFieldsCompleteness(initialValues,
                                               *blossom->getInputValidationMap(),
                                               FieldDef::INPUT_TYPE,
                                               errorMessage) == false)
        {
            error.addMeesage(errorMessage);
            error.addMeesage("check of completeness of input-fields failed");
            error.addMeesage("Check of blossom '"
                             + blossomName
                             + " in group '"
                             + blossomGroupName
                             + "' failed.");
            status.statusCode = BAD_REQUEST_RTYPE;
            status.errorMessage = errorMessage;
            break;
        }

        // process blossom
        if(blossom->growBlossom(blossomIO, &context, status, error) == false)
        {
            error.addMeesage("trigger blossom failed.");
            break;
        }

        // check output to be complete
        Hanami::DataMap* output = blossomIO.output.getItemContent()->toMap();
        if(blossom->validateFieldsCompleteness(*output,
                                               *blossom->getOutputValidationMap(),
                                               FieldDef::OUTPUT_TYPE,
                                               errorMessage) == false)
        {
            error.addMeesage(errorMessage);
            error.addMeesage("check of completeness of output-fields failed");
            error.addMeesage("Check of blossom '"
                             + blossomName
                             + " in group '"
                             + blossomGroupName
                             + "' failed.");
            status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
            break;
        }

        // TODO: override only with the output-values to avoid unnecessary conflicts
        result.clear();
        overrideItems(result, *output, ALL);

        success = true;
    }
    while(false);

    if(success == false) {
        LOG_ERROR(error);
    }

    checkStatusCode(blossom, blossomName, blossomGroupName, status, error);

    return success;
}

/**
 * @brief check if the given status-code is allowed for the endpoint
 *
 * @param blossom pointer to related blossom
 * @param blossomName name of blossom for error-message
 * @param blossomGroupName group of the blossom for error-message
 * @param status status to check
 * @param error reference for error-output
 */
void
HanamiRoot::checkStatusCode(Blossom* blossom,
                            const std::string &blossomName,
                            const std::string &blossomGroupName,
                            BlossomStatus &status,
                            Hanami::ErrorContainer &error)
{
    if(status.statusCode)
    {
        bool found = false;
        for(const uint32_t allowed : blossom->errorCodes)
        {
            if(allowed == status.statusCode) {
                found = true;
            }
        }

        // if given status-code is unexprected, then override it and clear the message
        // to avoid leaking unwanted information
        if(found == false)
        {
            error.addMeesage("Status-code '"
                             + std::to_string(status.statusCode)
                             + "' is not allowed as output for blossom '"
                             + blossomName
                             + "' in group '"
                             + blossomGroupName
                             + "'");
            status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
            status.errorMessage = "";
        }
    }

}

/**
 * @brief map the endpoint to the real target
 *
 * @param result reference to the result to identify the target
 * @param id request-id
 * @param type requested http-request-type
 *
 * @return false, if mapping failes, else true
 */
bool
HanamiRoot::mapEndpoint(EndpointEntry &result,
                        const std::string &id,
                        const HttpRequestType type)
{
    const auto id_it = endpointRules.find(id);
    if(id_it != endpointRules.end())
    {
        auto type_it = id_it->second.find(type);
        if(type_it != id_it->second.end())
        {
            result.type = type_it->second.type;
            result.group = type_it->second.group;
            result.name = type_it->second.name;

            return true;
        }
    }

    return false;
}

/**
 * @brief add new custom-endpoint without the parser
 *
 * @param id identifier for the new entry
 * @param httpType http-type (get, post, put, delete)
 * @param sakuraType sakura-type (tree or blossom)
 * @param group blossom-group
 * @param name tree- or blossom-id
 *
 * @return false, if id together with http-type is already registered, else true
 */
bool
HanamiRoot::addEndpoint(const std::string &id,
                        const HttpRequestType &httpType,
                        const SakuraObjectType &sakuraType,
                        const std::string &group,
                        const std::string &name)
{
    EndpointEntry newEntry;
    newEntry.type = sakuraType;
    newEntry.group = group;
    newEntry.name = name;

    // search for id
    auto id_it = endpointRules.find(id);
    if(id_it != endpointRules.end())
    {
        // search for http-type
        if(id_it->second.find(httpType) != id_it->second.end()) {
            return false;
        }

        // add new
        id_it->second.emplace(httpType, newEntry);
    }
    else
    {
        // add new
        std::map<HttpRequestType, EndpointEntry> typeEntry;
        typeEntry.emplace(httpType, newEntry);
        endpointRules.emplace(id, typeEntry);
    }

    return true;
}
