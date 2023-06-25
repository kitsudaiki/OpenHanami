/**
 * @file        generate_rest_api_docu.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

#include "generate_rest_api_docu.h"

#include <hanami_root.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiCommon/methods/file_methods.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiCommon/files/binary_file.h>
#include <libKitsunemimiCommon/process_execution.h>
#include <libKitsunemimiJson/json_item.h>

/**
 * @brief constructor
 */
GenerateRestApiDocu::GenerateRestApiDocu()
    : Blossom("Generate a documentation for the REST-API of all available components.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("type",
                       SAKURA_STRING_TYPE,
                       false,
                       "Output-type of the document (pdf, rst, md).");
    assert(addFieldDefault("type", new Kitsunemimi::DataValue("pdf")));
    assert(addFieldRegex("type", "^(pdf|rst|md)$"));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("documentation",
                        SAKURA_STRING_TYPE,
                        "REST-API-documentation as base64 converted string.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

bool
appendDocu(std::string &completeDocumentation,
           const std::string &componentDocu,
           Kitsunemimi::ErrorContainer &error)
{
    std::string rstDocu;
    if(Kitsunemimi::decodeBase64(rstDocu, componentDocu) == false)
    {
        error.addMeesage("Unable to convert documentation-payload from base64 back to rst");
        return false;
    }

    // attach new text to the final document
    completeDocumentation.append("\n");
    completeDocumentation.append(rstDocu);

    return true;
}

/**
 * @brief request endpoint-documentation from misaki itself
 *
 * @param completeDocumentation reference for the final document to attach new content
 */
bool
makeInternalRequest(std::string &completeDocumentation,
                    const std::string &type)
{
    Kitsunemimi::DataMap result;
    Kitsunemimi::ErrorContainer error;
    BlossomStatus status;
    Kitsunemimi::DataMap values;
    values.insert("type", new Kitsunemimi::DataValue(type));

    const bool ret = HanamiRoot::root->triggerBlossom(result,
                                                      "get_api_documentation",
                                                      "-",
                                                      Kitsunemimi::DataMap(),
                                                      values,
                                                      status,
                                                      error);
    if(ret == false) {
        return false;
    }

    return appendDocu(completeDocumentation, result.getStringByKey("documentation"), error);
}

/**
 * @brief runTask
 */
bool
GenerateRestApiDocu::runTask(BlossomIO &blossomIO,
                             const Kitsunemimi::DataMap &context,
                             BlossomStatus &status,
                             Kitsunemimi::ErrorContainer &error)
{
    const std::string role = context.getStringByKey("role");
    const std::string type = blossomIO.input.get("type").getString();
    const std::string token = context.getStringByKey("token");

    // create request for remote-calls
    RequestMessage request;
    request.id = "v1/documentation/api";
    request.httpType = Kitsunemimi::Hanami::GET_TYPE;
    request.inputValues = "{\"token\":\"" + token + "\",\"type\":\"" + type + "\"}";

    // create header of the final document
    std::string completeDocumentation = "";

    if(type == "pdf"
            || type == "rst")
    {
        completeDocumentation.append("*****************\n");
        completeDocumentation.append("API documentation\n");
        completeDocumentation.append("*****************\n\n");
    }
    else if(type == "md")
    {
        completeDocumentation.append("# API documentation\n");
    }

    makeInternalRequest(completeDocumentation, type);

    std::string output;

    if(type == "pdf")
    {
        if(convertRstToPdf(output, completeDocumentation, error) == false)
        {
            status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
            error.addMeesage("Failed to convert documentation from 'rst' to 'pdf'");
            return false;
        }
    }
    else
    {
        Kitsunemimi::encodeBase64(output,
                                  completeDocumentation.c_str(),
                                  completeDocumentation.size());
    }

    blossomIO.output.insert("documentation", output);

    return true;
}

/**
 * @brief convert rst-document into a pdf-document
 *
 * @param pdfOutput reference to return the resulting pdf-document as base64 converted string
 * @param rstInput string with rst-formated document
 * @param error reference for error-output
 *
 * @return true, if conversion was successful, else false
 */
bool
GenerateRestApiDocu::convertRstToPdf(std::string &pdfOutput,
                                     const std::string &rstInput,
                                     Kitsunemimi::ErrorContainer &error)
{
    bool result = false;
    const std::string uuid = generateUuid().toString();
    const std::string tempDir = "/tmp/" + uuid;

    do
    {
        // create unique temporary directory
        if(Kitsunemimi::createDirectory(tempDir, error) == false)
        {
            error.addMeesage("Failed to create temporary rst-directory to path '" + tempDir + "'");
            break;
        }

        // define file-paths
        const std::string rstPath = "/tmp/" + uuid + "/rest_api_docu.rst";
        const std::string pdfPath = "/tmp/" + uuid + "/output.pdf";

        // write complete rst-content to the source-file
        if(Kitsunemimi::writeFile(rstPath, rstInput, error) == false)
        {
            error.addMeesage("Failed to write temporary rst-file to path '" + rstPath + "'");
            break;
        }

        // run rst2pdf to convert the rst-document into a pdf-document
        std::vector<std::string> args;
        args.reserve(2);
        args.emplace_back(rstPath);
        args.emplace_back(pdfPath);
        Kitsunemimi::ProcessResult ret = Kitsunemimi::runSyncProcess("rst2pdf", args);
        if(ret.success == false)
        {
            error.addMeesage("Failed execute 'rst2pdf' to convert rst-file '"
                             + rstPath
                             + "' to pdf-file '"
                             + pdfPath
                             + "'");
            error.addSolution("Check if tool 'rst2pdf' is installed.");
            error.addSolution("Check if tool 'rst2pdf' is executable "
                              "and if not fix this with 'chmod +x /PATH/TO/BINARY'.");
            error.addSolution("Check if enough memory and storage is available.");
            break;
        }

        // read pdf-document into a byte-buffer
        Kitsunemimi::DataBuffer pdfContent;
        Kitsunemimi::BinaryFile pdfFile(pdfPath);

        if(pdfFile.readCompleteFile(pdfContent, error) == false)
        {
            error.addMeesage("Failed to read pdf-file on path '" + pdfPath + "'");
            break;
        }
        pdfFile.closeFile(error);

        // create output for the client
        Kitsunemimi::encodeBase64(pdfOutput, pdfContent.data, pdfContent.usedBufferSize);

        result = true;
        break;
    }
    while(true);

    // HINT(kitsudaiki): ignore result here, because it is only for cleanup
    Kitsunemimi::deleteFileOrDir(tempDir, error);

    return result;
}
