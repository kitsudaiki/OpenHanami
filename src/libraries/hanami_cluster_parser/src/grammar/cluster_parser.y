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
%require "3.8.2"

%define api.parser.class {ClusterParser}

%define api.prefix {cluster}
%define api.namespace {Hanami}
%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires
{
#include <string>
#include <iostream>
#include <map>
#include <vector>

#include <hanami_cluster_parser/cluster_meta.h>
#include <hanami_common/structs.h>

namespace Hanami
{

class ClusterParserInterface;

}
}

// The parsing context.
%param { Hanami::ClusterParserInterface& driver }

%locations

%code
{
#include <cluster_parsing/cluster_parser_interface.h>
# undef YY_DECL
# define YY_DECL \
    Hanami::ClusterParser::symbol_type clusterlex (Hanami::ClusterParserInterface& driver)
YY_DECL;
}

// Token
%define api.token.prefix {Cluster_}
%token
    END  0  "end of file"
    COMMA  ","
    ASSIGN  ":"
    LINEBREAK "linebreak"
    VERSION_1 "version1"
    SETTINGS "settings"
    BRICKS "bricks"
    BOOL_TRUE  "true"
    BOOL_FALSE "false"
;

%token <std::string> IDENTIFIER "identifier"
%token <std::string> STRING "string"
%token <long> NUMBER "number"
%token <double> FLOAT "float"

%type  <Hanami::Position> position
%type  <Hanami::BrickMeta> brick_settings

%%
%start startpoint;

// example
//
// version: 1
// settings:
//
// bricks:
//     1,1,1
//         input: test_input
//         number_of_neurons: 20
//
//     2,1,1
//         number_of_neurons": 10
//
//     3,1,1
//         output: test_output
//         number_of_neurons: 5
//         number_of_outputs: 2

startpoint:
    "version1" linebreaks "settings" ":" linebreaks settings "bricks" ":" linebreaks bricks
    {
        driver.output->version = 1;
    }
|
    "version1" linebreaks "bricks" ":" linebreaks bricks
    {
        driver.output->version = 1;
    }

settings:
    settings "identifier" ":" "number" linebreaks
    {
        if($2 == "neuron_cooldown")
        {
            driver.output->neuronCooldown = $4;
        }
        else if($2 == "max_connection_distance")
        {
            driver.output->maxConnectionDistance = $4;
        }
        else if($2 == "refractory_time")
        {
            if($4 < 1) {
                driver.error(yyla.location, "refractory_time must be >= 1");
                return 1;
            }
            driver.output->refractoryTime = $4;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $2 + "'");
            return 1;
        }
    }
|
    settings "identifier" ":" "float" linebreaks
    {
        if($2 == "neuron_cooldown")
        {
            driver.output->neuronCooldown = $4;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $2 + "'");
            return 1;
        }
    }
|
    settings "identifier" ":" "true" linebreaks
    {
        if($2 == "enable_reduction")
        {
            driver.output->enableReduction = true;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $2 + "'");
            return 1;
        }
    }
|
    settings "identifier" ":" "false" linebreaks
    {
        if($2 == "enable_reduction")
        {
            driver.output->enableReduction = false;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $2 + "'");
            return 1;
        }
    }
|
    "identifier" ":" "number" linebreaks
    {
        if($1 == "neuron_cooldown")
        {
            driver.output->neuronCooldown = $3;
        }
        else if($1 == "max_connection_distance")
        {
            driver.output->maxConnectionDistance = $3;
        }
        else if($1 == "refractory_time")
        {
            if($3 < 1) {
                driver.error(yyla.location, "refractory_time must be >= 1");
                return 1;
            }
            driver.output->refractoryTime = $3;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $1 + "'");
            return 1;
        }
    }
|
    "identifier" ":" "float" linebreaks
    {
        if($1 == "neuron_cooldown")
        {
            driver.output->neuronCooldown = $3;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $1 + "'");
            return 1;
        }
    }
|
    "identifier" ":" "true" linebreaks
    {
        if($1 == "enable_reduction")
        {
            driver.output->enableReduction = true;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $1 + "'");
            return 1;
        }
    }
|
    "identifier" ":" "false" linebreaks
    {
        if($1 == "enable_reduction")
        {
            driver.output->enableReduction = false;
        }
        else
        {
            driver.error(yyla.location, "unkown settings-field '" + $1 + "'");
            return 1;
        }
    }

bricks:
    bricks position linebreaks brick_settings
    {
        $4.position = $2;
        driver.output->bricks.push_back($4);
    }
|
    position linebreaks brick_settings
    {
        $3.position = $1;
        driver.output->bricks.push_back($3);
    }

brick_settings:
    brick_settings "identifier" ":" "identifier" linebreaks_eno
    {
        if($2 == "input")
        {
            $1.type = INPUT_BRICK_TYPE;
            $1.name = $4;
        }
        else if($2 == "output")
        {
            $1.type = OUTPUT_BRICK_TYPE;
            $1.name = $4;
        }
        else
        {
            driver.error(yyla.location, "unkown brick-field '" + $2 + "'");
            return 1;
        }

        $$ = $1;
    }
|
    brick_settings "identifier" ":" "number" linebreaks_eno
    {
        if($2 == "number_of_neurons")
        {
            $1.numberOfNeurons = $4;
        }
        else if($2 == "number_of_outputs")
        {
            $1.numberOfOutputs = $4;
        }
        else
        {
            driver.error(yyla.location, "unkown brick-field '" + $2 + "'");
            return 1;
        }

        $$ = $1;
    }
|
    "identifier" ":" "identifier" linebreaks
    {
        Hanami::BrickMeta brickMeta;

        if($1 == "input")
        {
            brickMeta.type = INPUT_BRICK_TYPE;
            brickMeta.name = $3;
        }
        else if($1 == "output")
        {
            brickMeta.type = OUTPUT_BRICK_TYPE;
            brickMeta.name = $3;
        }
        else
        {
            driver.error(yyla.location, "unkown brick-field '" + $1 + "'");
            return 1;
        }

        $$ = brickMeta;
    }
|
    "identifier" ":" "number" linebreaks
    {
        Hanami::BrickMeta brickMeta;

        if($1 == "number_of_neurons")
        {
            brickMeta.numberOfNeurons = $3;
        }
        else
        {
            driver.error(yyla.location, "unkown brick-field '" + $1 + "'");
            return 1;
        }

        $$ = brickMeta;
    }

position:
    "number" "," "number" "," "number"
    {
        Hanami::Position pos;
        pos.x = $1;
        pos.y = $3;
        pos.z = $5;
        $$ = pos;
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

void Hanami::ClusterParser::error(const Hanami::location& location,
                                               const std::string& message)
{
    driver.error(location, message);
}
