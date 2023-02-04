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

//================================================
// Alerting
//================================================

/**
 * Hide error-message block from the modal. Is necessary for initializing 
 * to avoid an empty error-message block inside of the modal.
 *
 * @param {target} base-name of the modal, where the error-message belongs to
 */
function clearAlertBox(target)
{
    var modal = document.getElementById(target + "_alert_box");

    if(modal.style.display === "block") 
    {
        const modalSize = document.getElementById(target + "_modal_content").clientHeight;
        const alertHeight = document.getElementById(target + "_alert_box").clientHeight;
        document.getElementById(target + "_alert_text_label").innerHTML = "";
        document.getElementById(target + "_modal_content").style.height = (modalSize - 20 - alertHeight) + "px";
    }

    modal.style.display = "none";
    document.getElementById(target + "_alert_text_label").innerHTML = "";
}

/**
 * Show error-message within a specific modal
 *
 * @param {target} base-name of the modal, where the error-message belongs to
 * @param {message} message, which should be printed
 */
function showErrorInModal(target, message)
{
    var modal = document.getElementById(target + "_alert_box");

    // in an old error-message is already shown, then close this first
    if(modal.style.display === "block") {
        clearAlertBox(target);
    }

    modal.style.display = "block";

    // calculate and update the size of the modal, to have enough space to insert the error-message
    const modalSize = document.getElementById(target + "_modal_content").clientHeight;
    const alertHeight = document.getElementById(target + "_alert_box").clientHeight;
    document.getElementById(target + "_alert_text_label").innerHTML = message;
    document.getElementById(target + "_modal_content").style.height = (modalSize + 20 + alertHeight) + "px";
}

//================================================
// cookies
//================================================

/**
 * Get value of a specific cookie
 *
 * @param {name} name of the cookie
 */
function getCookieValue(name) 
{
    const cn = name + "=";
    const idx = document.cookie.indexOf(cn)

    if(idx != -1) 
    {
        var end = document.cookie.indexOf(";", idx + 1);
        if (end == -1) end = document.cookie.length;
        return document.cookie.substring(idx + cn.length, end);
    } 
    else 
    {
        return "";
    }
}


//================================================
// Dropdown-menus
//================================================

/**
 * Fill dropdown-menu with values, which are requestd from the backend
 *
 * @param {dropdownDiv} ID of the dev, which should be filled with the requested set of values
 * @param {dropdownListRequest} request-path to get list of values from the backend to fill the dropdown-list
 */
function fillDropdownList(dropdownDiv, dropdownListRequest)
{     
    // get and check token
    const token = getAndCheckToken();
    if(token == "") {
        return;
    }

    // create request
    let listRequestConnection = new XMLHttpRequest();
    listRequestConnection.open("GET", dropdownListRequest, true);
    listRequestConnection.setRequestHeader("X-Auth-Token", token);

    // callback for success
    listRequestConnection.onload = function(e) 
    {
        if(listRequestConnection.status != 200) 
        {
            showErrorInModal("create", listRequestConnection.responseText);
            return;
        }

        // remove the old dropdown-list to create a new one
        const dropdownNode = document.getElementById(dropdownDiv);
        dropdownNode.innerHTML = '';

        // prepare container for the dropdown-menu
        var select = document.createElement("select");
        select.name = dropdownDiv + "_select";
        select.id = dropdownDiv + "_select";

        let idPos = 0;
        let namePos = 0;

        const content = JSON.parse(listRequestConnection.responseText);

        // search for the column, which has the title "name" and get its position
        const headerContent = content.header;
        for(let i = 0; i < headerContent.length; i++) 
        {
            if(headerContent[i] === "name") 
            {
                namePos = i;
                break;
            } 
        }

        // fill menu with name and id of all entries
        const bodyContent = content.body;
        for(let row = 0; row < bodyContent.length; row++) 
        {
            const id = bodyContent[row][idPos];
            const name = bodyContent[row][namePos];

            var option = document.createElement("option");
            option.value = id;
            option.text = name + "   ( " + id + " )";
            select.appendChild(option);
        }

        document.getElementById(dropdownDiv).appendChild(select);
    };

    // callback for fail
    listRequestConnection.onerror = function(e) 
    {
        console.log("Failed to load list of segment-templates.");
    };

    listRequestConnection.send(null);
}

/**
 * Fill dropdown-menu with a static list of values
 *
 * @param {dropdownDiv} ID of the dev, which should be filled with the static set of values
 * @param {values} List of values to fill into the dropdown-list
 */
function fillStaticDropdownList(dropdownDiv, values)
{     
    // remove the old dropdown-list to create a new one
    const dropdownNode = document.getElementById(dropdownDiv);
    dropdownNode.innerHTML = '';

    // prepare container for the dropdown-menu
    var select = document.createElement("select");
    select.name = dropdownDiv + "_select";
    select.id = dropdownDiv + "_select";
 
    // fill static values into dropdown-menu
    for(const val of values)
    {
        var option = document.createElement("option");
        option.value = val;
        option.text = val;
        select.appendChild(option);
    }
 
    document.getElementById(dropdownDiv).appendChild(select);
}

/**
 * Fill dropdown-menu with all projects, which are assigned to the actual user
 *
 * @param {dropdownDiv} ID of the dev, which should be filled with projects of the user
 */
function fillUserProjectDropdownList(dropdownDiv, userId="")
{
    // get and check token
    const authToken = getCookieValue("Auth_JWT_Token");
    if(authToken == "") {
        return;
    }

    // create request
    var listRequestConnection = new XMLHttpRequest();
    let path = "/control/misaki/v1/user/project";
    if(userId !== "") {
        path += "?user_id=" + userId
    }
    listRequestConnection.open("GET", path, true);
    listRequestConnection.setRequestHeader("X-Auth-Token", authToken);

    // callback for success
    listRequestConnection.onload = function(e) 
    {
        if(listRequestConnection.status != 200) 
        {
            showErrorInModal("create", listRequestConnection.responseText);
            return;
        }

        // remove the old dropdown-list to create a new one
        const dropdownNode = document.getElementById(dropdownDiv);
        dropdownNode.innerHTML = '';

        // prepare container for the dropdown-menu
        var select = document.createElement("select");
        select.name = dropdownDiv + "_select";
        select.id = dropdownDiv + "_select";

        const content = JSON.parse(listRequestConnection.responseText);

        // fill menu with name and id of all entries
        const projectList = content.projects;
        for(var row = 0; row < projectList.length; row++) 
        {
            const id = projectList[row].project_id;
            const role = projectList[row].role
            const isProjectAdmin = projectList[row].is_project_admin

            var option = document.createElement("option");
            option.value = id;
            option.text = id + " | " + role ;

            if(isProjectAdmin) {
                option.text += "  (Project-Admin)"
            }

            select.appendChild(option);
        }

        document.getElementById(dropdownDiv).appendChild(select);
    };

    // callback for fail
    listRequestConnection.onerror = function(e) 
    {
        console.log("Failed to load projects of user.");
    };

    listRequestConnection.send(null);
}

/**
 * Get selected value of a specific dropdown-menu
 *
 * @param {dropdownDiv} ID of the dev, which contains the requested dropdown-menu
 */
function getDropdownValue(dropdownDiv)
{
    var e = document.getElementById(dropdownDiv + "_select");
    return e.options[e.selectedIndex].value;
}

