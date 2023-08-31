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

#include <thread>

#include <common.h>

#include <hanami_root.h>
#include <args.h>
#include <config.h>
#include <api/websocket/cluster_io.h>
#include <api/http/v1/blossom_initializing.h>

#include <documentation/generate_rest_api_docu.h>

#include <libKitsunemimiArgs/arg_parser.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiConfig/config_handler.h>

#include <database/cluster_table.h>
#include <database/users_table.h>
#include <database/projects_table.h>
#include <database/data_set_table.h>
#include <database/checkpoint_table.h>
#include <database/request_result_table.h>
#include <database/error_log_table.h>
#include <database/audit_log_table.h>

int
main(int argc, char *argv[])
{
    Kitsunemimi::ErrorContainer error;
    HanamiRoot rootObj;
    initBlossoms();

    Kitsunemimi::initConsoleLogger(true);

    // create and init argument-parser
    Kitsunemimi::ArgParser argParser;
    registerArguments(&argParser, error);
    registerConfigs(error);

    // parse cli-input
    if(argParser.parse(argc, argv, error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }

    // generate api-, config- and database-docu, if requested
    if(argParser.wasSet("generate_docu"))
    {
        namespace fs = std::filesystem;

        Kitsunemimi::ErrorContainer error;

        //-------------------------------------------------------------------------

        std::string openApiDocu = "";
        createOpenApiDocumentation(openApiDocu);
        fs::path complete = fs::current_path() / fs::path{"open_api_docu.json"};
        if(writeFile(complete.generic_string(), openApiDocu, error, true) == false)
        {
            LOG_ERROR(error);
            return 1;
        }
        std::cout<<"Written OpenAPI-docu to "<<complete<<std::endl;

        //-------------------------------------------------------------------------

        std::string configDocu = "# Configs of Hanami\n\n";
        Kitsunemimi::ConfigHandler::getInstance()->createDocumentation(configDocu);
        complete = fs::current_path() / fs::path{"config.md"};
        if(writeFile(complete.generic_string(), configDocu, error, true) == false)
        {
            LOG_ERROR(error);
            return 1;
        }
        std::cout<<"Written Config-docu to "<<complete<<std::endl;

        //-------------------------------------------------------------------------

        std::string dbDocu = "# Database-Tables\n\n";
        ClusterTable::getInstance()->createDocumentation(dbDocu);
        ProjectsTable::getInstance()->createDocumentation(dbDocu);
        UsersTable::getInstance()->createDocumentation(dbDocu);
        DataSetTable::getInstance()->createDocumentation(dbDocu);
        RequestResultTable::getInstance()->createDocumentation(dbDocu);
        CheckpointTable::getInstance()->createDocumentation(dbDocu);
        ErrorLogTable::getInstance()->createDocumentation(dbDocu);
        AuditLogTable::getInstance()->createDocumentation(dbDocu);
        complete = fs::current_path() / fs::path{"db.md"};
        if(writeFile(complete.generic_string(), dbDocu, error, true) == false)
        {
            LOG_ERROR(error);
            return 1;
        }
        std::cout<<"Written Database-docu to "<<complete<<std::endl;

        //-------------------------------------------------------------------------

        return 0;
    }

    // init and check config-file
    std::string configPath = argParser.getStringValue("config");
    if(configPath == "") {
        configPath = "/etc/hanami/hanami.conf";
    }
    if(INIT_CONFIG(configPath, error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }

    // get config-parameter for logger
    bool success = false;
    const bool enableDebug = GET_BOOL_CONFIG("DEFAULT", "debug", success);
    if(success == false) {
        return 1;
    }

    const std::string logPath = GET_STRING_CONFIG("DEFAULT", "log_path", success);
    if(success == false) {
        return 1;
    }

    // init logger
    Kitsunemimi::initConsoleLogger(enableDebug);
    Kitsunemimi::initFileLogger(logPath, "hanami", enableDebug);

    // init root-object
    if(rootObj.init(error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }
    rootObj.initThreads();

    // sleep forever
    std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock>::max());

    return 0;
}
