// Apache License Version 2.0

// Copyright 2020 Tobias Anker

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function createUser_request(outputFunc, userId, name, password, isAdmin, token)
{
    const path = "/control/misaki/v1/user";
    let reqContent = "{id:\"" + userId;
    reqContent += "\",name:\"" + name;
    reqContent += "\",password:\"" + password;
    reqContent += "\",is_admin:" + isAdmin + "}";
    createObject_request(outputFunc, path, reqContent, token);
}

function addProjectToUser_request(outputFunc, userId, projectId, role, isProjectAdmin, token)
{
    const path = "/control/misaki/v1/user/project";
    let reqContent = "{id:\"" + userId;
    reqContent += "\",project_id:\"" + projectId;
    reqContent += "\",role:\"" + role;
    reqContent += "\",is_project_admin:" + isProjectAdmin + "}";
    createObject_request(outputFunc, path, reqContent, token);
}

function removeProjectFromUser_request(outputFunc, userId, projectId, token)
{
    let path = "/control/misaki/v1/user/project";
    path += "?project_id=" + projectId;
    path += "&id=" + userId;
    deleteObject_request(outputFunc, path, token);
}

function listUsers_request(outputFunc, token)
{
    listObjects_request(outputFunc, "/control/misaki/v1/user/all", token);
}

function deleteUser_request(postProcessFunc, userId, token)
{
    const request = "/control/misaki/v1/user?id=" + userId;
    deleteObject_request(postProcessFunc, request, token);
}

