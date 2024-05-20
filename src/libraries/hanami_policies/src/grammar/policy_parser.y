/**
 * @file        policy_parser.y
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

%skeleton "lalr1.cc"

%defines
%require "3.8.2"

%define api.parser.class {PolicyParser}

%define api.prefix {policy}
%define api.namespace {Hanami}
%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires
{
#include <string>
#include <map>
#include <vector>
#include <iostream>

#include <hanami_policies/policy.h>

namespace Hanami
{

class PolicyParserInterface;

}
}

// The parsing context.
%param { Hanami::PolicyParserInterface& driver }

%locations

%code
{
#include <policy_parsing/policy_parser_interface.h>

# undef YY_DECL
# define YY_DECL \
    Hanami::PolicyParser::symbol_type policylex (Hanami::PolicyParserInterface& driver)
YY_DECL;

Hanami::PolicyEntry tempPolicyEntry;
std::vector<std::string> tempRules;
}

// Token
%define api.token.prefix {Policy_}
%token
    END  0  "end of file"
    BRAKET_OPEN  "["
    BRAKET_CLOSE "]"
    MINUS  "-"
    COMMA  ","
    ASSIGN  ":"
    GET  "GET"
    PUT  "PUT"
    POST  "POST"
    DELETE  "DELETE"
;

%token <std::string> IDENTIFIER "identifier"
%token <std::string> PATH "path"

%type  <HttpRequestType> request_type
%type  <std::string> endpoint;

%%
%start component_policy_content;

component_policy_content:
    component_policy_content endpoint policy_entry
    {
        driver.m_result->insert(std::make_pair($2, tempPolicyEntry));
    }
|
    endpoint policy_entry
    {
        driver.m_result->insert(std::make_pair($1, tempPolicyEntry));
    }

policy_entry:
    policy_entry "-" request_type ":" rule_list
    {
        if($3 == HttpRequestType::GET_TYPE) {
            tempPolicyEntry.getRules = tempRules;
        }
        if($3 == HttpRequestType::POST_TYPE) {
            tempPolicyEntry.postRules = tempRules;
        }
        if($3 == HttpRequestType::PUT_TYPE) {
            tempPolicyEntry.putRules = tempRules;
        }
        if($3 == HttpRequestType::DELETE_TYPE) {
            tempPolicyEntry.deleteRules = tempRules;
        }
    }
|
    "-" request_type ":" rule_list
    {
        tempPolicyEntry.getRules.clear();
        tempPolicyEntry.postRules.clear();
        tempPolicyEntry.putRules.clear();
        tempPolicyEntry.deleteRules.clear();

        if($2 == HttpRequestType::GET_TYPE) {
            tempPolicyEntry.getRules = tempRules;
        }
        if($2 == HttpRequestType::POST_TYPE) {
            tempPolicyEntry.postRules = tempRules;
        }
        if($2 == HttpRequestType::PUT_TYPE) {
            tempPolicyEntry.putRules = tempRules;
        }
        if($2 == HttpRequestType::DELETE_TYPE) {
            tempPolicyEntry.deleteRules = tempRules;
        }
    }

rule_list:
    rule_list "," "identifier"
    {
        tempRules.push_back($3);
    }
|
    "identifier"
    {
        tempRules.clear();
        tempRules.push_back($1);
    }


endpoint:
    "path"
    {
        $$ = $1;
    }
|
    "identifier"
    {
        $$ = $1;
    }

request_type:
    "GET"
    {
        $$ = HttpRequestType::GET_TYPE;
    }
|
    "POST"
    {
        $$ = HttpRequestType::POST_TYPE;
    }
|
    "PUT"
    {
        $$ = HttpRequestType::PUT_TYPE;
    }
|
    "DELETE"
    {
        $$ = HttpRequestType::DELETE_TYPE;
    }

%%

void Hanami::PolicyParser::error(const Hanami::location& location,
                                              const std::string& message)
{
    driver.error(location, message);
}
