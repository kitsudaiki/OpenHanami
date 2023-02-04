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

function getRequestResultData_request(outputFunc, resultUuid, token)
{
    const request = "/control/shiori/v1/request_result?uuid=" + resultUuid;
    getObject_request(outputFunc, request, token);
}

function listResults_request(outputFunc, token)
{
    listObjects_request(outputFunc, "/control/shiori/v1/request_result/all", token);
}

function deleteResult_request(postProcessFunc, resultUuid, token)
{
    const request = "/control/shiori/v1/request_result?uuid=" + resultUuid;
    deleteObject_request(postProcessFunc, request, token);
}

