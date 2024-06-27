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

function openDeleteModal(deleteFunction, identifier)
{
    clearAlertBox("delete");

    let modal = document.getElementById("delete_modal");
    let acceptButton = document.getElementById("delete_modal_accept_button");
    let cancelButton = document.getElementById("delete_modal_cancel_button");

    document.getElementById('delete_label_text').innerText = identifier;

    // handle accept-button
    acceptButton.onclick = function() 
    {
        deleteFunction(identifier);
    }

    // handle cancel-button
    cancelButton.onclick = function() {
        modal.style.display = "none";
    } 

    // handle click outside of the window
    window.onclick = function(event) 
    {
        if(event.target == modal) {
            modal.style.display = "none";
        }
    } 

    modal.style.display = "block";
}

function openGenericModal(outputFunc, modalName, clearFunction, closeModal=true) 
{
    clearFunction();

    let modal = document.getElementById(modalName);
    let acceptButton = document.getElementById(modalName + "_accept_button");
    let cancelButton = document.getElementById(modalName + "_cancel_button");

    // handle accept-button
    acceptButton.onclick = function() 
    {
        outputFunc();
    }

    // handle cancel-button
    cancelButton.onclick = function() 
    {
        clearFunction();
        modal.style.display = "none";
    } 

    // handle click outside of the window
    window.onclick = function(event) 
    {
        if(event.target == modal)
        {
            clearFunction();
            modal.style.display = "none";
        }
    } 

    modal.style.display = "block";
}
