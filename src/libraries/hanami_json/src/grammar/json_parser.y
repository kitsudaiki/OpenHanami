/**
 *  @file    json_parser.y
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

%skeleton "lalr1.cc"

%defines
%require "3.8.2"

%define api.parser.class {JsonParser}

%define api.prefix {json}
%define api.namespace {Kitsunemimi}
%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires
{
#include <string>
#include <iostream>
#include <cmath>
#include <hanami_common/items/data_items.h>

using Kitsunemimi::DataItem;
using Kitsunemimi::DataArray;
using Kitsunemimi::DataValue;
using Kitsunemimi::DataMap;


namespace Kitsunemimi
{

class JsonParserInterface;

}
}

// The parsing context.
%param { Kitsunemimi::JsonParserInterface& driver }

%locations

%code
{
#include <json_parsing/json_parser_interface.h>
# undef YY_DECL
# define YY_DECL \
    Kitsunemimi::JsonParser::symbol_type jsonlex (Kitsunemimi::JsonParserInterface& driver)
YY_DECL;
}

// Token
%define api.token.prefix {Json_}
%token
    END  0  "end of file"
    EXPRESTART  "{"
    EXPREEND  "}"
    BRACKOPEN  "["
    BRACKCLOSE  "]"
    COMMA  ","
    ASSIGN  ":"
    EXPONENT "e+"
    BOOL_TRUE  "true"
    BOOL_FALSE "false"
    NULLVAL "null"
;


%token <std::string> IDENTIFIER "identifier"
%token <std::string> STRING "string"
%token <long> NUMBER "number"
%token <double> FLOAT "float"

%type  <DataItem*> json_abstract
%type  <DataValue*> json_value
%type  <DataArray*> json_array
%type  <DataArray*> json_array_content
%type  <DataMap*> json_object
%type  <DataMap*> json_object_content

%%
%start startpoint;


startpoint:
    json_abstract
    {
        if(driver.dryRun == false) {
            driver.setOutput($1);
        }
    }

json_abstract:
    json_object
    {
        if(driver.dryRun == false) {
            $$ = (DataItem*)$1;
        } else {
            $$ = nullptr;
        }
    }
|
    json_array
    {
        if(driver.dryRun == false) {
            $$ = (DataItem*)$1;
        } else {
            $$ = nullptr;
        }
    }
|
    json_value
    {
        if(driver.dryRun == false) {
            $$ = (DataItem*)$1;
        } else {
            $$ = nullptr;
        }
    }

json_object:
    "{" json_object_content "}"
    {
        $$ = $2;
    }
|
   "{" "}"
    {
        if(driver.dryRun == false) {
            $$ = new DataMap();
        } else {
            $$ = nullptr;
        }
    }

json_object_content:
    json_object_content "," "identifier" ":" json_abstract
    {
        if(driver.dryRun == false) {
            $1->insert($3, $5);
        }
        $$ = $1;
    }
|
    "identifier" ":" json_abstract
    {
        if(driver.dryRun == false) {
            $$ = new DataMap();
            $$->insert($1, $3);
        } else {
            $$ = nullptr;
        }
    }
|
    json_object_content "," "string" ":" json_abstract
    {
        if(driver.dryRun == false) {
            $1->insert(driver.removeQuotes($3), $5);
        }
        $$ = $1;
    }
|
    "string" ":" json_abstract
    {
        if(driver.dryRun == false) {
            $$ = new DataMap();
            $$->insert(driver.removeQuotes($1), $3);
        } else {
            $$ = nullptr;
        }
    }

json_array:
    "[" json_array_content "]"
    {
        $$ = $2;
    }
|
    "[" "]"
    {
        if(driver.dryRun == false) {
            $$ = new DataArray();
        } else {
            $$ = nullptr;
        }
    }

json_array_content:
    json_array_content "," json_abstract
    {
        if(driver.dryRun == false) {
            $1->append($3);
        }
        $$ = $1;
    }
|
    json_abstract
    {
        if(driver.dryRun == false) {
            $$ = new DataArray();
            $$->append($1);
        } else {
            $$ = nullptr;
        }
    }

json_value:
    "identifier"
    {
        if(driver.dryRun == false) {
            $$ = new DataValue($1);
        } else {
            $$ = nullptr;
        }
    }
|
    "number"
    {
        if(driver.dryRun == false) {
            $$ = new DataValue($1);
        } else {
            $$ = nullptr;
        }
    }
|
    "float" "e+" "number"
    {
        if(driver.dryRun == false)
        {
            float value = $1;
            value *= std::pow(10, $3);
            $$ = new DataValue(value);
        }
        else {
            $$ = nullptr;
        }
    }
|
    "float"
    {
        if(driver.dryRun == false) {
            $$ = new DataValue($1);
        } else {
            $$ = nullptr;
        }
    }
|
    "string"
    {
        if(driver.dryRun == false) {
            $$ = new DataValue(driver.removeQuotes($1));
        } else {
            $$ = nullptr;
        }
    }
|
    "true"
    {
        if(driver.dryRun == false) {
            $$ = new DataValue(true);
        } else {
            $$ = nullptr;
        }
    }
|
    "false"
    {
        if(driver.dryRun == false) {
            $$ = new DataValue(false);
        } else {
            $$ = nullptr;
        }
    }
|
    "null"
    {
        $$ = nullptr;
    }

%%

void Kitsunemimi::JsonParser::error(const Kitsunemimi::location& location,
                                          const std::string& message)
{
    driver.error(location, message);
}
