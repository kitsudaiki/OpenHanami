/**
 * @file        users_database.cpp
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

#include <database/users_table.h>

#include <hanami_common/items/table_item.h>
#include <hanami_common/methods/string_methods.h>
#include <hanami_json/json_item.h>
#include <hanami_crypto/hashes.h>

#include <hanami_database/sql_database.h>

UsersTable* UsersTable::instance = nullptr;

/**
 * @brief constructor
 */
UsersTable::UsersTable()
    : HanamiSqlAdminTable(Hanami::SqlDatabase::getInstance())
{
    m_tableName = "users";

    DbHeaderEntry projects;
    projects.name = "projects";
    m_tableHeader.push_back(projects);

    DbHeaderEntry isAdmin;
    isAdmin.name = "is_admin";
    isAdmin.type = BOOL_TYPE;
    m_tableHeader.push_back(isAdmin);

    DbHeaderEntry pwHash;
    pwHash.name = "pw_hash";
    pwHash.maxLength = 64;
    pwHash.hide = true;
    m_tableHeader.push_back(pwHash);

    DbHeaderEntry saltVal;
    saltVal.name = "salt";
    saltVal.maxLength = 64;
    saltVal.hide = true;
    m_tableHeader.push_back(saltVal);
}

/**
 * @brief destructor
 */
UsersTable::~UsersTable() {}

/**
 * @brief get content of an environment-variable
 *
 * @param content reference for output
 * @param key name of the environment-variable
 *
 * @return false, if varibale is not set, else true
 */
bool
UsersTable::getEnvVar(std::string &content,
                      const std::string &key) const
{
    const char* val = getenv(key.c_str());
    if(val == NULL) {
        return false;
    }

    content = std::string(val);
    return true;
}

/**
 * @brief get list of all users, who have admin-status
 *
 * @param error reference for error-output
 *
 * @return true, if seccuessful, else false
 */
bool
UsersTable::getAllAdminUser(Hanami::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("is_admin", "true");

    // get admin-user from db
    Hanami::JsonItem users;
    if(getFromDb(users, conditions, error, false) == false)
    {
        error.addMeesage("Failed to get admin-users from database");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief try to initialize a new admin-user in database
 *
 * @param error reference for error-output
 *
 * @return true, if seccuessful, else false
 */
bool
UsersTable::initNewAdminUser(Hanami::ErrorContainer &error)
{
    std::string userId = "";
    std::string userName = "";
    std::string pw = "";

    // check if there is already an admin-user in the databasae
    if(getAllAdminUser(error)) {
        return true;
    }
    LOG_DEBUG("Found no admin-users in database, so try to create a new one");

    // get env with new admin-user id
    if(getEnvVar(userId, "HANAMI_ADMIN_USER_ID") == false)
    {
        error.addMeesage("environment variable 'HANAMI_ADMIN_USER_ID' was not set, "
                         "but is required to initialize a new admin-user");
        LOG_ERROR(error);
        return false;
    }

    // get env with new admin-user name
    if(getEnvVar(userName, "HANAMI_ADMIN_USER_NAME") == false)
    {
        error.addMeesage("environment variable 'HANAMI_ADMIN_USER_NAME' was not set, "
                         "but is required to initialize a new admin-user");
        LOG_ERROR(error);
        return false;
    }

    // get env with new admin-user password
    if(getEnvVar(pw, "HANAMI_ADMIN_PASSWORD") == false)
    {
        error.addMeesage("environment variable 'HANAMI_ADMIN_PASSWORD' was not set, "
                         "but is required to initialize a new admin-user");
        LOG_ERROR(error);
        return false;
    }

    // generate hash from password
    std::string pwHash;
    const std::string salt = "e307bee0-9286-49bd-9273-6f644c12da1d";
    const std::string saltedPw = pw + salt;
    Hanami::generate_SHA_256(pwHash, saltedPw);

    Hanami::JsonItem userData;
    userData.insert("id", userId);
    userData.insert("name", userName);
    userData.insert("projects", "[]");
    userData.insert("is_admin", true);
    userData.insert("creator_id", "MISAKI");
    userData.insert("pw_hash", pwHash);
    userData.insert("salt", salt);

    // add new admin-user to db
    if(addUser(userData, error) == false)
    {
        error.addMeesage("Failed to add new initial admin-user to database");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief add a new user to the database
 *
 * @param userData json-item with all information of the user to add to database
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
UsersTable::addUser(Hanami::JsonItem &userData,
                    Hanami::ErrorContainer &error)
{
    if(insertToDb(userData, error) == false)
    {
        error.addMeesage("Failed to add user to database");
        return false;
    }

    return true;
}

/**
 * @brief get a user from the database
 *
 * @param result reference for the result-output in case that a user with this name was found
 * @param userId id of the requested user
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
UsersTable::getUser(Hanami::JsonItem &result,
                    const std::string &userId,
                    Hanami::ErrorContainer &error,
                    const bool showHiddenValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("id", userId);

    // get user from db
    if(getFromDb(result, conditions, error, showHiddenValues) == false)
    {
        error.addMeesage("Failed to get user with id '"
                         + userId
                         + "' from database");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get all users from the database table
 *
 * @param result reference for the result-output
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
UsersTable::getAllUser(Hanami::TableItem &result,
                       Hanami::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    if(getFromDb(result, conditions, error, false) == false)
    {
        error.addMeesage("Failed to get all users from database");
        return false;
    }

    return true;
}

/**
 * @brief delete a user from the table
 *
 * @param userId id of the user to delete
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
UsersTable::deleteUser(const std::string &userId,
                       Hanami::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("id", userId);

    if(deleteFromDb(conditions, error) == false)
    {
        error.addMeesage("Failed to delete user with id '"
                         + userId
                         + "' from database");
        return false;
    }

    return true;
}

/**
 * @brief update the projects-frild of a specific user
 *
 * @param userId id of the user, who has to be updated
 * @param newProjects new projects-entry for the database
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
UsersTable::updateProjectsOfUser(const std::string &userId,
                                 const std::string &newProjects,
                                 Hanami::ErrorContainer &error)
{
    Hanami::JsonItem newValues;
    newValues.insert("projects", Hanami::JsonItem(newProjects));

    std::vector<RequestCondition> conditions;
    conditions.emplace_back("id", userId);

    if(updateInDb(conditions, newValues, error) == false)
    {
        error.addMeesage("Failed to update projects for user with id '"
                         + userId
                         + "' from database");
        return false;
    }

    return true;
}
