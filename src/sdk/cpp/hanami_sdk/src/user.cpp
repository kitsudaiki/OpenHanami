/**
 * @file        user.cpp
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

#include <hanami_sdk/user.h>
#include <common/http_client.h>

namespace Hanami
{

/**
 * @brief create a new user in misaki
 *
 * @param result reference for response-message
 * @param id id of the new user
 * @param userName name of the new user
 * @param password password of the new user
 * @param isAdmin true to make new user to an admin
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
createUser(std::string &result,
           const std::string &userId,
           const std::string &userName,
           const std::string &password,
           const bool isAdmin,
           Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user";
    const std::string vars = "";
    const std::string jsonBody = "{\"id\":\""
                                 + userId
                                 + "\",\"name\":\""
                                 + userName
                                 + "\",\"password\":\""
                                 + password
                                 + "\",\"is_admin\":"
                                 + (isAdmin ? "true" : "false") +
                                 + "}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to create user with name '" + userName + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get information of a user from misaki
 *
 * @param result reference for response-message
 * @param userId id of the requested user
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getUser(std::string &result,
        const std::string &userId,
        Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user";
    const std::string vars = "id=" + userId;

    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get user with id '" + userId + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all visible users on misaki
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listUser(std::string &result,
         Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user/all";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list users");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete a user from misaki
 *
 * @param result reference for response-message
 * @param userId id of the user, which should be deleted
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteUser(std::string &result,
           const std::string &userId,
           Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user";
    const std::string vars = "id=" + userId;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete user with id '" + userId + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief assign an already existing project to a user.
 *
 * @param result reference for response-message
 * @param userId id of the user, who should be assigned to another projects
 * @param projectId id of the project, which should be assigned to the user
 * @param role role of the user, while signed-in in the project
 * @param isProjectAdmin true, if user should be admin within the new added project
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
addProjectToUser(std::string &result,
                 const std::string &userId,
                 const std::string &projectId,
                 const std::string &role,
                 const bool isProjectAdmin,
                 Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user/project";
    const std::string vars = "";
    const std::string jsonBody = "{id:\"" + userId;
                                 + "\",project_id:\"" + projectId
                                 + "\",role:\"" + role
                                 + "\",is_project_admin:" + (isProjectAdmin ? "true" : "false")
                                 + "}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to add project with id '"
                         + projectId
                         + "' to user with id '"
                         + userId
                         + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief unassign project from a user
 *
 * @param result reference for response-message
 * @param userId id of the user, who should be unassigned
 * @param projectId id of the project, which should be removed from the user
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
removeProjectFromUser(std::string &result,
                      const std::string &userId,
                      const std::string &projectId,
                      Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user/project";
    const std::string vars = "id=" + userId + "&project_id=" + projectId;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to remove project with id '"
                         + projectId
                         + "' from user with id '"
                         + userId
                         + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all projects where the current user is assigned to
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listProjectsOfUser(std::string &result,
                   Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user/project";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list project of user");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief switch to another project by requesting a new token for the selected project. The project
 *        must be in the list of assigned projects of the user.
 *
 * @param result reference for response-message
 * @param projectId id of the project, where to switch to
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
switchProject(std::string &result,
              const std::string &projectId,
              Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/user/project";
    const std::string vars = "";
    const std::string jsonBody = "{\"project_id\":\"" + projectId + "\"}";

    // send request
    if(request->sendPutRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to swtich to project with id '" + projectId + "'");
        LOG_ERROR(error);
        return false;
    }

    // try to parse response
    json item = json::parse(result, nullptr, false);
    if(item.is_discarded())
    {
        error.addMeesage("Failed to parse token-response");
        LOG_ERROR(error);
        return false;
    }

    // get token from parsed item
    const std::string newToken = item["token"];
    if(newToken == "")
    {
        error.addMeesage("Can not find token in token-response");
        LOG_ERROR(error);
        return false;
    }

    request->updateToken(newToken);

    return true;
}

} // namespace Hanami
