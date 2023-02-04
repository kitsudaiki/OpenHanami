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
 
function createProject_request(outputFunc, projectId, name, token)
{
    const path = "/control/misaki/v1/project";
    const reqContent = "{\"id\":\"" + projectId + "\",\"name\":\"" + name + "\"}";
    createObject_request(outputFunc, path, reqContent, token);
}

function listProjects_request(outputFunc, token)
{
    listObjects_request(outputFunc, "/control/misaki/v1/project/all", token);
}

function deleteProject_request(postProcessFunc, projectId, token)
{
    const request = "/control/misaki/v1/project?id=" + projectId;
    deleteObject_request(postProcessFunc, request, token);
}

