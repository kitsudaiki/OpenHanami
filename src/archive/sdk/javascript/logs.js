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

function listAuditLogs_request(outputFunc, userId, page, token)
{
    let request = "/control/v1/audit_log?";
    if(userId !== "") {
        request += "user_id=" + userId + "&";
    }
    request += "page=" + page;
    getObject_request(outputFunc, request, token);
}
 
function listErrorLogs_request(outputFunc, userId, page, token)
{
    const request = "/control/v1/error_log?user_id=" + userId + "&page=" + page;
    getObject_request(outputFunc, request, token);
}
