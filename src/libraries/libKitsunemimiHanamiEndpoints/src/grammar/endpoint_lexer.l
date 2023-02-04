%{ /* -*- C++ -*- */
/**
 * @file        endpoint_lexer.l
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

# include <cerrno>
# include <climits>
# include <cstdlib>
# include <string>
# include <endpoint_parsing/endpoint_parser_interface.h>
# include <endpoint_parser.h>

# undef yywrap
# define yywrap() 1


// The location of the current token.
# ifdef YY_DECL
# undef YY_DECL
# endif
# define YY_DECL \
    Kitsunemimi::Hanami::EndpointParser::symbol_type endpointlex (Kitsunemimi::Hanami::EndpointParserInterface& driver)
YY_DECL;
Kitsunemimi::Hanami::location endpointloc;

%}


%option noyywrap nounput batch debug yylineno prefix="endpoint"
path  [a-zA-Z_0-9\/\-_]*
long  -?([0-9]+)
blank [ \t]

%{
    # define YY_USER_ACTION  endpointloc.columns (yyleng);
%}

%%

%{
    endpointloc.step();
%}

{blank}+  endpointloc.step();
[\n]      {endpointloc.lines(1); endpointloc.step(); }
"-"       return Kitsunemimi::Hanami::EndpointParser::make_MINUS (endpointloc);
":"       return Kitsunemimi::Hanami::EndpointParser::make_ASSIGN (endpointloc);
"->"      return Kitsunemimi::Hanami::EndpointParser::make_ARROW (endpointloc);
"GET"     return Kitsunemimi::Hanami::EndpointParser::make_GET (endpointloc);
"PUT"     return Kitsunemimi::Hanami::EndpointParser::make_PUT (endpointloc);
"POST"    return Kitsunemimi::Hanami::EndpointParser::make_POST (endpointloc);
"DELETE"  return Kitsunemimi::Hanami::EndpointParser::make_DELETE (endpointloc);
"blossom" return Kitsunemimi::Hanami::EndpointParser::make_BLOSSOM (endpointloc);
"tree"    return Kitsunemimi::Hanami::EndpointParser::make_TREE (endpointloc);

{path}    return Kitsunemimi::Hanami::EndpointParser::make_PATH(yytext, endpointloc);

.          driver.error(endpointloc, "invalid character");
<<EOF>>    return Kitsunemimi::Hanami::EndpointParser::make_END(endpointloc);

%%


void Kitsunemimi::Hanami::EndpointParserInterface::scan_begin(const std::string &inputString)
{
    Kitsunemimi::Hanami::location newEndpointloc;
    endpointloc = newEndpointloc;
    yy_flex_debug = m_traceParsing;
    yy_scan_string(inputString.c_str());
}

void Kitsunemimi::Hanami::EndpointParserInterface::scan_end()
{
    yy_delete_buffer(YY_CURRENT_BUFFER);
}


