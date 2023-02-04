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

/**
 * Remove all cookies of the side
 */
function deleteAllCookies() 
{
    const cookies = document.cookie.split(";");
    for(var i = 0; i < cookies.length; i++) {
        document.cookie = cookies[i] + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
    }
}

var loginFunction = function(responseJson)
{
    console.log("login successful");

    // TODO: check if json-parsing is successful
    document.cookie = "Auth_JWT_Token=" + responseJson.token + "; SameSite=Strict; Secure";
    document.cookie = "User_Name=" + responseJson.name + "; SameSite=Strict;";
    document.cookie = "Is_Admin=" + responseJson.is_admin + "; SameSite=Strict;";

    updateSidebar();

    document.getElementById("login_id_field").value = "";
    document.getElementById("login_pw_field").value = "";
    document.getElementById("header_user_name").innerHTML = responseJson.name;

    // load cluster-overview as first site
    $("#content_div").load("/subsites/kyouko/cluster.html"); 

    let modal = document.getElementById("login_modal");
    modal.style.display = "none";
}

/**
 * Trigger login-modal
 */
var loginModalFunction = function() 
{
    deleteAllCookies();
    let modal = document.getElementById("login_modal");
    let loginButton = document.getElementById("modal_login_button");

    // handle login-button
    loginButton.onclick = function() 
    {
        const userId = document.getElementById("login_id_field").value;
        const pw = document.getElementById("login_pw_field").value;
        login_request(loginFunction, userId, pw);
    }

    modal.style.display = "block";
}

/**
 * Check if token is still valid. If expired or invalid, return to login
 */
function getAndCheckToken() 
{
    const authToken = getCookieValue("Auth_JWT_Token");
    if(authToken == "") {
        loginModalFunction();
    } else {
        checkTokenRequest(loginModalFunction, authToken);
    }
    return authToken;
}

/**
 * Delete cookies and return to login
 */
function logout()
{
    deleteAllCookies() 
    loginModalFunction();
}
