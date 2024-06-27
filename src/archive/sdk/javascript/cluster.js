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
 
function createCluster_request(outputFunc, name, templateStr, token)
{
    const path = "/control/v1/cluster";
    let reqContent = "{\"name\":\"" + name;
    reqContent += "\",\"template\":\"" + btoa(templateStr) + "\"}";
    createObject_request(outputFunc, path, reqContent, token);
}

function saveCluster_request(outputFunc, name, clusterUuid, token)
{
    const path = "/control/v1/cluster/save";
    let reqContent = "{\"name\":\"" + name;
    reqContent += "\",\"cluster_uuid\":\"" + clusterUuid + "\"}";
    createObject_request(outputFunc, path, reqContent, token);
}

function restoreCluster_request(outputFunc, snapshotUuid, clusterUuid, token)
{
    const path = "/control/v1/cluster/load";
    let reqContent = "{\"snapshot_uuid\":\"" + snapshotUuid;
    reqContent += "\",\"cluster_uuid\":\"" + clusterUuid + "\"}";
    createObject_request(outputFunc, path, reqContent, token);
}

function listClusters_request(outputFunc, token)
{
    listObjects_request(outputFunc, "/control/v1/cluster/all", token);
}

function deleteCluster_request(postProcessFunc, clusterUuid, token)
{
    const request = "/control/v1/cluster?uuid=" + clusterUuid;
    deleteObject_request(postProcessFunc, request, token);
}

