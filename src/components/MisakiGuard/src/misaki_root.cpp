/**
 * @file        misaki_root.cpp
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

#include "misaki_root.h"

#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiSakuraDatabase/sql_database.h>
#include <libKitsunemimiCommon/files/text_file.h>

#include <api/blossom_initializing.h>

Kitsunemimi::Jwt* MisakiRoot::jwt = nullptr;
UsersTable* MisakiRoot::usersTable = nullptr;
ProjectsTable* MisakiRoot::projectsTable = nullptr;
Kitsunemimi::Sakura::SqlDatabase* MisakiRoot::database = nullptr;
Kitsunemimi::Hanami::Policy* MisakiRoot::policies = nullptr;

/**
 * @brief constructor
 */
MisakiRoot::MisakiRoot() {}

/**
 * @brief init root-object
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
MisakiRoot::init(Kitsunemimi::ErrorContainer &error)
{
    if(initDatabase(error) == false)
    {
        error.addMeesage("Failed to initialize database");
        return false;
    }

    initBlossoms();

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
MisakiRoot::initDatabase(Kitsunemimi::ErrorContainer &error)
{
    bool success = false;

    // read database-path from config
    database = new Kitsunemimi::Sakura::SqlDatabase();
    const std::string databasePath = GET_STRING_CONFIG("DEFAULT", "database", success);
    if(success == false)
    {
        error.addMeesage("No database-path defined in config.");
        return false;
    }

    // initalize database
    if(database->initDatabase(databasePath, error) == false)
    {
        error.addMeesage("Failed to initialize sql-database.");
        return false;
    }

    // initialize projects-table
    projectsTable = new ProjectsTable(database);
    if(projectsTable->initTable(error) == false)
    {
        error.addMeesage("Failed to initialize project-table in database.");
        return false;
    }

    // initialize users-table
    usersTable = new UsersTable(database);
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
MisakiRoot::initPolicies(Kitsunemimi::ErrorContainer &error)
{
    bool success = false;

    // read policy-file-path from config
    const std::string policyFilePath = GET_STRING_CONFIG("misaki", "policies", success);
    if(success == false)
    {
        error.addMeesage("No policy-file defined in config.");
        return false;
    }

    // read policy-file
    std::string policyFileContent;
    if(Kitsunemimi::readFile(policyFileContent, policyFilePath, error) == false)
    {
        error.addMeesage("Failed to read policy-file");
        return false;
    }

    // parse policy-file
    policies = new Kitsunemimi::Hanami::Policy();
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
MisakiRoot::initJwt(Kitsunemimi::ErrorContainer &error)
{
    bool success = false;

    // read jwt-token-key from config
    const std::string tokenKeyPath = GET_STRING_CONFIG("misaki", "token_key_path", success);
    if(success == false)
    {
        error.addMeesage("No token_key_path defined in config.");
        return false;
    }

    std::string tokenKeyString;
    if(Kitsunemimi::readFile(tokenKeyString, tokenKeyPath, error) == false)
    {
        error.addMeesage("Failed to read token-file '" + tokenKeyPath + "'");
        return false;
    }

    // init jwt for token create and sign
    CryptoPP::SecByteBlock tokenKey((unsigned char*)tokenKeyString.c_str(), tokenKeyString.size());
    jwt = new Kitsunemimi::Jwt(tokenKey);

    return true;
}
