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

function makeHttpRequest(outputFunc, path, type, payload, token)
{
    // create request
    let requestConnection = new XMLHttpRequest();
    requestConnection.open(type, path, true);
    requestConnection.setRequestHeader("X-Auth-Token", token);

    // create requeset
    requestConnection.onload = function(e) 
    {
        if(requestConnection.status != 200) 
        {
            console.log("HTTP-request failed: " + requestConnection.status);
        }

        outputFunc(requestConnection.status, requestConnection.responseText);
    };

    // callback for fail
    requestConnection.onerror = function(e) 
    {
        console.log("Failed to request request-results from shiori.");
    };

    requestConnection.send(payload);
}

function createObject_request(outputFunc, path, payload, token)
{
    makeHttpRequest(outputFunc, path, "POST", payload, token);
}

function updateObject_request(outputFunc, path, payload, token)
{
    makeHttpRequest(outputFunc, path, "PUT", payload, token);
}

function getObject_request(outputFunc, path, token)
{
    makeHttpRequest(outputFunc, path, "GET", null, token);
}

function listObjects_request(outputFunc, path, token)
{
    makeHttpRequest(outputFunc, path, "GET", null, token);
}

function deleteObject_request(postProcessFunc, path, token)
{
    makeHttpRequest(postProcessFunc, path, "DELETE", null, token);
}
