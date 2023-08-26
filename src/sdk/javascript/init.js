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
 
function login_request(loginFunc, user, pw)
{
    const path = "/control/v1/token";
    const reqContent = "{\"id\":\"" + user + "\",\"password\":\"" + pw + "\"}";

    let loginConnection = new XMLHttpRequest();
    loginConnection.open("POST", path, true);

    // create requeset
    loginConnection.onload = function(e) 
    {
        if(loginConnection.status != 200) 
        {
            // TODO: error-popup
            showErrorInModal("login", loginConnection.responseText);
            return;
        }

        loginFunc(JSON.parse(loginConnection.responseText));
    };

    // callback for fail
    loginConnection.onerror = function(e) 
    {
        console.log("Failed to request request-results from shiori.");
    };

    loginConnection.send(reqContent);
}

function checkTokenRequest(errorFunc, token)
{
    // create request
    const request = "/control/v1/auth?token=" + token;
    //console.log("TOKENNNNNNNNNNNNNNNNNNNNNNNNNNN: " + token);
    let authConnection = new XMLHttpRequest();
    authConnection.open("GET", request, true);
    authConnection.setRequestHeader("X-Auth-Token", token);

    // callback for success
    authConnection.onload = function(e) 
    {
        if(authConnection.status != 200) 
        {
            console.log("token-check failed");
            errorFunc();
        }
    };

    // callback for fail
    authConnection.onerror = function(e) 
    {
        errorFunc();
    };

    authConnection.send(null);
}
