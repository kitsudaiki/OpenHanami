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

function createTask_request(outputFunc, name, type, clusterUuid, datasetUuid, token)
{
    let path = "/control/kyouko/v1/task";

    // create request-content
    var reqContent = "{\"name\":\"" + name
                     + "\",\"type\":\"" + type
                     + "\",\"cluster_uuid\":\"" + clusterUuid
                     + "\",\"data_set_uuid\":\"" + datasetUuid + "\"}";

    createObject_request(outputFunc, path, reqContent, token);
}

function listTasks_request(outputFunc, clusterUuid, token)
{
    const path = "/control/kyouko/v1/task/all?cluster_uuid=" + clusterUuid
    listObjects_request(outputFunc, path, token);
}

function deleteTask_request(outputFunc, taskUuid, clusterUuid, token)
{
    const request = "/control/kyouko/v1/task?cluster_uuid=" + clusterUuid + "&uuid=" + taskUuid;
    deleteObject_request(outputFunc, request, token);
}
