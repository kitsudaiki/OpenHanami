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

#include <api/endpoint_processing/http_server.h>
#include <api/endpoint_processing/http_websocket_thread.h>
#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster_init.h>
#include <core/processing/processing_unit_handler.h>
#include <core/temp_file_handler.h>
#include <core/thread_binder.h>
#include <database/audit_log_table.h>
#include <database/checkpoint_table.h>
#include <database/cluster_table.h>
#include <database/data_set_table.h>
#include <database/error_log_table.h>
#include <database/projects_table.h>
#include <database/request_result_table.h>
#include <database/tempfile_table.h>
#include <database/users_table.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/logger.h>
#include <hanami_common/methods/file_methods.h>
#include <hanami_config/config_handler.h>
#include <hanami_database/sql_database.h>
#include <hanami_hardware/host.h>
#include <hanami_hardware/power_measuring.h>
#include <hanami_hardware/speed_measuring.h>
#include <hanami_hardware/temperature_measuring.h>
#include <hanami_root.h>

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
HanamiRoot::HanamiRoot() { root = this; }

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
HanamiRoot::init(Hanami::ErrorContainer& error)
{
    /*if(useOpencl)
    {
        Hanami::GpuHandler oclHandler;
        assert(oclHandler.initDevice(error));
        assert(oclHandler.m_interfaces.size() == 1);
        gpuInterface = oclHandler.m_interfaces.at(0);
    }*/

    if (initDataDirectory(error) == false) {
        error.addMeesage("Failed to initialize directories");
        return false;
    }

    // init predefinde random-values
    m_randomValues = new uint32_t[NUMBER_OF_RAND_VALUES];
    srand(time(NULL));
    for (uint32_t i = 0; i < NUMBER_OF_RAND_VALUES; i++) {
        m_randomValues[i] = static_cast<uint32_t>(rand());
    }

    if (initDatabase(error) == false) {
        error.addMeesage("Failed to initialize database");
        return false;
    }

    clearCluster(error);

    if (initJwt(error) == false) {
        error.addMeesage("Failed to initialize jwt");
        return false;
    }

    if (initPolicies(error) == false) {
        error.addMeesage("Failed to initialize policies");
        return false;
    }

    if (initHttpServer() == false) {
        error.addMeesage("initializing http-server failed");
        LOG_ERROR(error);
        return false;
    }

    Hanami::Host* host = Hanami::Host::getInstance();
    if (host->initHost(error) == false) {
        error.addMeesage("Failed to initialize host-information.");
        LOG_ERROR(error);
        // can fail because it runs within a virtual machine for example, but this should
        // not prevent the process from starting
    }

    bool success = false;
    useCuda = GET_BOOL_CONFIG("DEFAULT", "use_cuda", success);
    assert(success);

    // create thread-binder
    if (ThreadBinder::getInstance()->init(error) == false) {
        error.addMeesage("failed to init thread-binder");
        LOG_ERROR(error);
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
    if (processingUnitHandler->initProcessingUnits(1) == false) {
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
HanamiRoot::initDatabase(Hanami::ErrorContainer& error)
{
    bool success = false;

    // read database-path from config
    Hanami::SqlDatabase* database = Hanami::SqlDatabase::getInstance();
    const std::string databasePath = GET_STRING_CONFIG("DEFAULT", "database", success);
    LOG_DEBUG("database-path: '" + databasePath + "'");
    if (success == false) {
        error.addMeesage("No database-path defined in config.");
        LOG_ERROR(error);
        return false;
    }

    // initalize database
    if (database->initDatabase(databasePath, error) == false) {
        error.addMeesage("Failed to initialize sql-database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize cluster-table
    ClusterTable* clustersTable = ClusterTable::getInstance();
    if (clustersTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize cluster-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize projects-table
    ProjectsTable* projectsTable = ProjectsTable::getInstance();
    if (projectsTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize project-table in database.");
        return false;
    }

    // initialize users-table
    UsersTable* usersTable = UsersTable::getInstance();
    if (usersTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize user-table in database.");
        return false;
    }
    if (usersTable->initNewAdminUser(error) == false) {
        error.addMeesage("Failed to initialize new admin-user even this is necessary.");
        return false;
    }

    // initialize dataset-table
    DataSetTable* dataSetTable = DataSetTable::getInstance();
    if (dataSetTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize dataset-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize request-result-table
    RequestResultTable* requestResultTable = RequestResultTable::getInstance();
    if (requestResultTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize request-result-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize checkpoint-table
    CheckpointTable* clusterCheckpointTable = CheckpointTable::getInstance();
    if (clusterCheckpointTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize checkpoint-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize tempfile-table
    TempfileTable* tempfileTable = TempfileTable::getInstance();
    if (tempfileTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize tempfile-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize error-log-table
    ErrorLogTable* errorLogTable = ErrorLogTable::getInstance();
    if (errorLogTable->initTable(error) == false) {
        error.addMeesage("Failed to initialize error-log-table in database.");
        LOG_ERROR(error);
        return false;
    }

    // initialize audit-log-table
    AuditLogTable* auditLogTable = AuditLogTable::getInstance();
    if (auditLogTable->initTable(error) == false) {
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
    if (GET_BOOL_CONFIG("http", "enable", success) == false) {
        return true;
    }

    // get stuff from config
    const uint16_t port = GET_INT_CONFIG("http", "port", success);
    const std::string ip = GET_STRING_CONFIG("http", "ip", success);
    const uint32_t numberOfThreads = GET_INT_CONFIG("http", "number_of_threads", success);

    // create server
    httpServer = new HttpServer(ip, port);
    httpServer->startThread();

    // start threads
    for (uint32_t i = 0; i < numberOfThreads; i++) {
        const std::string name = "HttpWebsocketThread";
        HttpWebsocketThread* httpWebsocketThread = new HttpWebsocketThread(name);
        httpWebsocketThread->startThread();
        m_threads.push_back(httpWebsocketThread);
    }

    return true;
}

/**
 * @brief HanamiRoot::initDataDirectory
 * @return
 */
bool
HanamiRoot::initDataDirectory(Hanami::ErrorContainer& error)
{
    bool success = false;

    const std::string datasetsPath = GET_STRING_CONFIG("storage", "data_set_location", success);
    if (success == false) {
        error.addMeesage("No data_set_location defined in config.");
        return false;
    }

    if (createDirectory(datasetsPath, error) == false) {
        error.addMeesage("Failed to create directory for data-sets.");
        return false;
    }

    const std::string checkpointsPath
        = GET_STRING_CONFIG("storage", "checkpoint_location", success);
    if (success == false) {
        error.addMeesage("No checkpoint_location defined in config.");
        return false;
    }

    if (createDirectory(checkpointsPath, error) == false) {
        error.addMeesage("Failed to create directory for checkpoints.");
        return false;
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
HanamiRoot::initPolicies(Hanami::ErrorContainer& error)
{
    bool success = false;

    // read policy-file-path from config
    const std::string policyFilePath = GET_STRING_CONFIG("auth", "policies", success);
    if (success == false) {
        error.addMeesage("No policy-file defined in config.");
        return false;
    }

    // read policy-file
    std::string policyFileContent;
    if (Hanami::readFile(policyFileContent, policyFilePath, error) == false) {
        error.addMeesage("Failed to read policy-file");
        return false;
    }

    // parse policy-file
    Hanami::Policy* policies = Hanami::Policy::getInstance();
    if (policies->parse(policyFileContent, error) == false) {
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
HanamiRoot::initJwt(Hanami::ErrorContainer& error)
{
    bool success = false;

    // read jwt-token-key from config
    const std::string tokenKeyPath = GET_STRING_CONFIG("auth", "token_key_path", success);
    if (success == false) {
        error.addMeesage("No token_key_path defined in config.");
        return false;
    }

    std::string tokenKeyString;
    if (Hanami::readFile(tokenKeyString, tokenKeyPath, error) == false) {
        error.addMeesage("Failed to read token-file '" + tokenKeyPath + "'");
        return false;
    }

    // init jwt for token create and sign
    tokenKey
        = CryptoPP::SecByteBlock((unsigned char*)tokenKeyString.c_str(), tokenKeyString.size());

    return true;
}

/**
 * @brief delete all clusters, because after a restart these are broken
 */
void
HanamiRoot::clearCluster(Hanami::ErrorContainer& error)
{
    ClusterTable::getInstance()->deleteAllCluster(error);
    // TODO: if a checkpoint exist for a broken cluster, then the cluster should be
    //       restored with the last available checkpoint
}
