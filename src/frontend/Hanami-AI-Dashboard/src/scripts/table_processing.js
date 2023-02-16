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
 * convert json-object of a list-request into a table
 *
 * @param {content} json-content with the data for the table
 * @param {headerMapping} map to filter and replace names in header
 * @param {selector} id of the div, where the table should be printed
 * @param {additionalButtons} additional buttons for each row
 */
function constructTable(content, headerMapping, selector, additionalButtons) 
{
    // clear old table-content
    $(selector).empty();

    const colIds = Headers(content.header, headerMapping, selector); 
    Body(content.body, selector, colIds, additionalButtons); 
}
   
/**
 * convert json-object of a list-request into a table-header
 *
 * @param {headerContent} json-content with the header for the table
 * @param {headerMapping} map to filter and replace names in header
 * @param {selector} id of the div, where the table should be printed
 */
function Headers(headerContent, headerMapping, selector) 
{
    let colIds = [];
    let header = $('<tr/>');             
    for(let i = 0; i < headerContent.length; i++) 
    {
        if(headerMapping.has(headerContent[i])) 
        {
            colIds.push(i);
            header.append($('<th/>').html(headerMapping.get(headerContent[i])));
        }
    }
    $(selector).append(header);

    return colIds;
}     

/**
 * convert json-object of a list-request into a table-content
 *
 * @param {bodyContent} json-content with the content for the table
 * @param {selector} id of the div, where the table should be printed
 * @param {colIds} list of column-ids to filter the content
 * @param {additionalButtons} additional buttons for each row
 */
function Body(bodyContent, selector, colIds, additionalButtons) 
{
    for(let row = 0; row < bodyContent.length; row++) 
    {
        let body = $('<tr/>');  

        // add textual values to the row
        let rowContent = bodyContent[row];   
        for(let i = 0; i < colIds.length; i++) 
        {
            // if cell is a json-object, then transform it into a better readable version by replacing special characters
            let cell = JSON.stringify(rowContent[colIds[i]]);
            cell = cell.replaceAll("},", "},<br/>");
            cell = cell.replaceAll(",", "<br/>");
            cell = cell.replaceAll("[", "");
            cell = cell.replaceAll("\"", "");
            cell = cell.replaceAll("]", "");
            cell = cell.replaceAll("{", "");
            cell = cell.replaceAll("}", "");
            cell = cell.replaceAll("\\n", "<br/>");
            body.append($('<td/>').html(cell));
        }

        if(additionalButtons.length > 0)
        {
            // add additional buttons to each row of the table
            let buttons = "";
            let dropdown = "<div class=\"dropdown\"><button class=\"dropbtn\">Settings</button><div class=\"dropdown-content\" style=\"right:0;\">"
            for(let i = 0; i < additionalButtons.length; i++)
            {
                buttons += '<button class="table_dropdown_button" value="' + rowContent[0] + '" ';
                buttons += additionalButtons[i];
            }
            dropdown += buttons;
            dropdown += "</div></div>"

            // additional buttons should be aligned at the right side
            body.append($('<td/ style="text-align: right;">').html($(dropdown)));
        }


        $(selector).append(body);
    }
} 
