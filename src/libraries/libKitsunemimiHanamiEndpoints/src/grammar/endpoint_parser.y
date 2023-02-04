/**
 * @file        endpoint_parser.y
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2021 Tobias Anker
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

%skeleton "lalr1.cc"

%defines
%require "3.0.4"

%define parser_class_name {EndpointParser}

%define api.prefix {endpoint}
%define api.namespace {Kitsunemimi::Hanami}
%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires
{
#include <string>
#include <map>
#include <iostream>
#include <libKitsunemimiHanamiCommon/enums.h>

namespace Kitsunemimi
{
namespace Hanami
{

class EndpointParserInterface;

}  // namespace Hanami
}  // namespace Kitsunemimi
}

// The parsing context.
%param { Kitsunemimi::Hanami::EndpointParserInterface& driver }

%locations

%code
{
#include <endpoint_parsing/endpoint_parser_interface.h>
#include <libKitsunemimiHanamiEndpoints/endpoint.h>

# undef YY_DECL
# define YY_DECL \
    Kitsunemimi::Hanami::EndpointParser::symbol_type endpointlex (Kitsunemimi::Hanami::EndpointParserInterface& driver)
YY_DECL;

std::map<Kitsunemimi::Hanami::HttpRequestType, Kitsunemimi::Hanami::EndpointEntry> tempEndpoint;
}

// Token
%define api.token.prefix {Endpoint_}
%token
    END  0  "end of file"
    MINUS  "-"
    ARROW  "->"
    ASSIGN  ":"
    GET  "GET"
    PUT  "PUT"
    POST  "POST"
    DELETE  "DELETE"
    BLOSSOM  "blossom"
    TREE  "tree"
;

%token <std::string> PATH "path"

%type  <SakuraObjectType> object_type
%type  <Kitsunemimi::Hanami::HttpRequestType> request_type

%%
%start startpoint;


startpoint:
    endpoint_content
    {
    }
|
    %empty
    {
    }

endpoint_content:
    endpoint_content "path" endpoint_object_content
    {
        driver.m_result->insert(std::make_pair($2, tempEndpoint));
    }
|
    "path" endpoint_object_content
    {
        driver.m_result->clear();
        driver.m_result->insert(std::make_pair($1, tempEndpoint));
    }

endpoint_object_content:
    endpoint_object_content "-" request_type "->" object_type ":" "path" ":" "path"
    {
        EndpointEntry entry;
        entry.type = $5;
        entry.group = $7;
        entry.name = $9;
        tempEndpoint.insert(std::make_pair($3, entry));
    }
|
    endpoint_object_content "-" request_type "->" object_type ":" "path"
    {
        EndpointEntry entry;
        entry.type = $5;
        entry.group = "-";
        entry.name = $7;
        tempEndpoint.insert(std::make_pair($3, entry));
    }
|
    "-" request_type "->" object_type ":" "path" ":" "path"
    {
        tempEndpoint.clear();
        EndpointEntry entry;
        entry.type = $4;
        entry.group = $6;
        entry.name = $8;
        tempEndpoint.insert(std::make_pair($2, entry));
    }
|
    "-" request_type "->" object_type ":" "path"
    {
        tempEndpoint.clear();
        EndpointEntry entry;
        entry.type = $4;
        entry.group = "-";
        entry.name = $6;
        tempEndpoint.insert(std::make_pair($2, entry));
    }

object_type:
    "blossom"
    {
        $$ = SakuraObjectType::BLOSSOM_TYPE;
    }
|
    "tree"
    {
        $$ = SakuraObjectType::TREE_TYPE;
    }

request_type:
    "GET"
    {
        $$ = Kitsunemimi::Hanami::HttpRequestType::GET_TYPE;
    }
|
    "POST"
    {
        $$ = Kitsunemimi::Hanami::HttpRequestType::POST_TYPE;
    }
|
    "PUT"
    {
        $$ = Kitsunemimi::Hanami::HttpRequestType::PUT_TYPE;
    }
|
    "DELETE"
    {
        $$ = Kitsunemimi::Hanami::HttpRequestType::DELETE_TYPE;
    }

%%

void Kitsunemimi::Hanami::EndpointParser::error(const Kitsunemimi::Hanami::location& location,
                                                const std::string& message)
{
    driver.error(location, message);
}
