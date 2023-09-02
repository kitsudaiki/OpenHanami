/**
 *  @file    ini_parser.y
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

%skeleton "lalr1.cc"

%defines

//requires 3.2 to avoid the creation of the stack.hh
%require "3.8.2"
%define api.parser.class {IniParser}

%define api.prefix {ini}
%define api.namespace {Kitsunemimi}
%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires
{
#include <string>
#include <map>
#include <utility>
#include <iostream>
#include <vector>
#include <hanami_common/items/data_items.h>

using Kitsunemimi::DataItem;
using Kitsunemimi::DataArray;
using Kitsunemimi::DataValue;
using Kitsunemimi::DataMap;

namespace Kitsunemimi
{
class IniParserInterface;
}
}

// The parsing context.
%param { Kitsunemimi::IniParserInterface& driver }

%locations

%code
{
#include <ini_parsing/ini_parser_interface.h>
# undef YY_DECL
# define YY_DECL \
    Kitsunemimi::IniParser::symbol_type inilex (Kitsunemimi::IniParserInterface& driver)
YY_DECL;
}

// Token
%define api.token.prefix {Ini_}
%token
    END  0  "end of file"
    LINEBREAK "lbreak"
    BOOL_TRUE  "true"
    BOOL_FALSE "false"
    EQUAL "="
    BRACKOPEN "["
    BRACKCLOSE "]"
    COMMA ","
;

%token <std::string> IDENTIFIER "identifier"
%token <std::string> STRING "string"
%token <std::string> STRING_PLN "string_pln"
%token <long> NUMBER "number"
%token <double> FLOAT "float"

%type <DataMap*> grouplist
%type <std::string> groupheader
%type <DataMap*> itemlist
%type <DataItem*> itemValue
%type <DataItem*> identifierlist

%%
%start startpoint;

startpoint:
    grouplist
    {
        driver.setOutput($1);
    }

grouplist:
    grouplist groupheader itemlist
    {
        $1->insert($2, $3);
        $$ = $1;
    }
|
    grouplist groupheader
    {
        $1->insert($2, new DataMap());
        $$ = $1;
    }
|
    groupheader itemlist
    {
        DataMap* newMap = new DataMap();
        newMap->insert($1, $2);
        $$ = newMap;
    }
|
    groupheader
    {
        DataMap* newMap = new DataMap();
        newMap->insert($1, new DataMap());
        $$ = newMap;
    }

groupheader:
    "[" "identifier" "]" linebreaks
    {
        $$ = $2;
    }

itemlist:
    itemlist "identifier" "=" itemValue linebreaks
    {
        $1->insert($2, $4);
        $$ = $1;
    }
|
    "identifier" "=" itemValue linebreaks
    {
        DataMap* newMap = new DataMap();
        newMap->insert($1, $3);
        $$ = newMap;
    }

itemValue:
    "identifier" "=" "identifier"
    {
        std::string temp = $1 + "=" + $3;
        $$ = new DataValue(temp);
    }
|
    identifierlist
    {
        $$ = $1;
    }
|
    "identifier"
    {
        $$ = new DataValue($1);
    }
|
    "string_pln"
    {
        $$ = new DataValue($1);
    }
|
    "string"
    {
        $$ = new DataValue(driver.removeQuotes($1));
    }
 |
    "number"
    {
        $$ = new DataValue($1);
    }
|
    "float"
    {
        $$ = new DataValue($1);
    }
|
    "true"
    {
        $$ = new DataValue(true);
    }
|
    "false"
    {
        $$ = new DataValue(false);
    }
|
    %empty
    {
        $$ = new DataValue("");
    }


identifierlist:
    identifierlist "," "string"
    {
        DataArray* array = dynamic_cast<DataArray*>($1);
        array->append(new DataValue(driver.removeQuotes($3)));
        $$ = $1;
    }
|
    identifierlist "," "string_pln"
    {
        DataArray* array = dynamic_cast<DataArray*>($1);
        array->append(new DataValue($3));
        $$ = $1;
    }
|
    identifierlist "," "identifier"
    {
        DataArray* array = dynamic_cast<DataArray*>($1);
        array->append(new DataValue($3));
        $$ = $1;
    }
|
    "string" "," "string"
    {
        DataArray* tempItem = new DataArray();
        tempItem->append(new DataValue(driver.removeQuotes($1)));
        tempItem->append(new DataValue(driver.removeQuotes($3)));
        $$ = tempItem;
    }
|
    "string_pln" "," "string_pln"
    {
        DataArray* tempItem = new DataArray();
        tempItem->append(new DataValue($1));
        tempItem->append(new DataValue($3));
        $$ = tempItem;
    }
|
    "identifier" "," "identifier"
    {
        DataArray* tempItem = new DataArray();
        tempItem->append(new DataValue($1));
        tempItem->append(new DataValue($3));
        $$ = tempItem;
    }


linebreaks:
   linebreaks "lbreak"
|
   "lbreak"


%%

void Kitsunemimi::IniParser::error(const Kitsunemimi::location& location,
                                   const std::string& message)
{
    driver.error(location, message);
}
