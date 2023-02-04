/**
 * @file       cluster_parser.y
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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
%require "3.0.4"

%define parser_class_name {ClusterParser}

%define api.prefix {cluster}
%define api.namespace {Kitsunemimi::Hanami}
%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires
{
#include <string>
#include <iostream>
#include <vector>
#include <libKitsunemimiCommon/items/data_items.h>
#include <libKitsunemimiHanamiClusterParser/cluster_meta.h>

using Kitsunemimi::DataItem;
using Kitsunemimi::DataArray;
using Kitsunemimi::DataValue;
using Kitsunemimi::DataMap;


namespace Kitsunemimi
{
namespace Hanami
{

class ClusterParserInterface;

}  // namespace Hanami
}  // namespace Kitsunemimi
}

// The parsing context.
%param { Kitsunemimi::Hanami::ClusterParserInterface& driver }

%locations

%code
{
#include <cluster_parsing/cluster_parser_interface.h>
# undef YY_DECL
# define YY_DECL \
    Kitsunemimi::Hanami::ClusterParser::symbol_type clusterlex (Kitsunemimi::Hanami::ClusterParserInterface& driver)
YY_DECL;
}

// Token
%define api.token.prefix {Cluster_}
%token
    END  0  "end of file"
    COMMA  ","
    ASSIGN  ":"
    ARROW  "->"
    LINEBREAK "linebreak"
    VERSION_1 "version1"
    OUT "out"
    NAME "name"
    SEGMENTS "segments"
;


%token <std::string> IDENTIFIER "identifier"
%token <std::string> STRING "string"
%token <std::string> STRING_PLN "string_pln"
%token <long> NUMBER "number"
%token <double> FLOAT "float"

%type  <std::string> string_ident
%type  <std::vector<Kitsunemimi::Hanami::SegmentMetaPtr>> segments
%type  <Kitsunemimi::Hanami::SegmentMetaPtr> segment
%type  <std::vector<Kitsunemimi::Hanami::ClusterConnection>> outputs
%type  <Kitsunemimi::Hanami::ClusterConnection> output

%%
%start startpoint;

// example
//
// version: 1
// segments:
//     input
//         name: input
//         out: -> central : test_input
//
//     example_segment
//         name: central
//         out:  test_output -> output
//
//     output
//         name: output
//         combine: 2

startpoint:
    "version1" linebreaks "segments" ":" linebreaks segments
    {
        driver.output->version = 1;
        driver.output->segments = $6;
    }

segments:
    segments segment
    {
        $$ = $1;
        $$.push_back($2);
    }
|
    segment
    {
        std::vector<SegmentMetaPtr> outputs;
        outputs.push_back($1);
        $$ = outputs;
    }

segment:
    string_ident linebreaks "name" ":" string_ident linebreaks_eno outputs
    {
        if($1 == "output")
        {
            driver.error(yyla.location, "Output-segments are not allowed to have any outputs.");
            return 1;
        }

        SegmentMetaPtr segment;
        segment.type = $1;
        segment.name = $5;
        segment.outputs = $7;
        $$ = segment;
    }
|
    string_ident linebreaks "name" ":" string_ident linebreaks_eno
    {
        if($1 == "input")
        {
            driver.error(yyla.location, "Input-segments must have at least one output.");
            return 1;
        }

        SegmentMetaPtr segment;
        segment.type = $1;
        segment.name = $5;
        $$ = segment;
    }

outputs:
    outputs output
    {
        $$ = $1;
        $$.push_back($2);
    }
|
    output
    {
        std::vector<ClusterConnection> outputs;
        outputs.push_back($1);
        $$ = outputs;
    }

output:
    "out" ":" "->" string_ident ":" string_ident linebreaks_eno
    {
        ClusterConnection connection;
        connection.sourceBrick = "x";
        connection.targetSegment = $4;
        connection.targetBrick = $6;
        $$ = connection;
    }
|
    "out" ":" string_ident "->" string_ident linebreaks_eno
    {
        ClusterConnection connection;
        connection.sourceBrick = $3;
        connection.targetSegment = $5;
        connection.targetBrick = "x";
        $$ = connection;
    }
|
    "out" ":" string_ident "->" string_ident ":" string_ident linebreaks_eno
    {
        ClusterConnection connection;
        connection.sourceBrick = $3;
        connection.targetSegment = $5;
        connection.targetBrick = $7;
        $$ = connection;
    }

string_ident:
    "identifier"
    {
        $$ = $1;
    }
|
    "string"
    {
        $$ = $1;
    }

linebreaks:
    linebreaks "linebreak"
    {}
|
    "linebreak"
    {}

linebreaks_eno:
    linebreaks "linebreak"
    {}
|
    "linebreak"
    {}
|
    "end of file"
    {}

%%

void Kitsunemimi::Hanami::ClusterParser::error(const Kitsunemimi::Hanami::location& location,
                                               const std::string& message)
{
    driver.error(location, message);
}
