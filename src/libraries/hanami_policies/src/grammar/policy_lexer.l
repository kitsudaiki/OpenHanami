%{ /* -*- C++ -*- */
/**
 * @file        policy_lexer.l
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

# include <cerrno>
# include <climits>
# include <cstdlib>
# include <string>
# include <policy_parsing/policy_parser_interface.h>
# include <policy_parser.h>

# undef yywrap
# define yywrap() 1


// The location of the current token.
# ifdef YY_DECL
# undef YY_DECL
# endif
# define YY_DECL \
    Hanami::PolicyParser::symbol_type policylex (Hanami::PolicyParserInterface& driver)
YY_DECL;
Hanami::location policyloc;

%}


%option noyywrap nounput batch debug yylineno prefix="policy"

path  [a-zA-Z_0-9\/\-_]*
id    [a-zA-Z_0-9]*
long  -?([0-9]+)
blank [ \t]

%{
    # define YY_USER_ACTION  policyloc.columns (yyleng);
%}

%%

%{
    policyloc.step();
%}

{blank}+  policyloc.step();
[\n]      {policyloc.lines(1); policyloc.step(); }
"["       return Hanami::PolicyParser::make_BRAKET_OPEN (policyloc);
"]"       return Hanami::PolicyParser::make_BRAKET_CLOSE (policyloc);
"-"       return Hanami::PolicyParser::make_MINUS (policyloc);
","       return Hanami::PolicyParser::make_COMMA (policyloc);
":"       return Hanami::PolicyParser::make_ASSIGN (policyloc);
"GET"     return Hanami::PolicyParser::make_GET (policyloc);
"PUT"     return Hanami::PolicyParser::make_PUT (policyloc);
"POST"    return Hanami::PolicyParser::make_POST (policyloc);
"DELETE"  return Hanami::PolicyParser::make_DELETE (policyloc);

{id}      return Hanami::PolicyParser::make_IDENTIFIER(yytext, policyloc);
{path}    return Hanami::PolicyParser::make_PATH(yytext, policyloc);

.          driver.error(policyloc, "invalid character");
<<EOF>>    return Hanami::PolicyParser::make_END(policyloc);

%%


void Hanami::PolicyParserInterface::scan_begin(const std::string &inputString)
{
    Hanami::location newPolicyloc;
    policyloc = newPolicyloc;
    yy_flex_debug = m_traceParsing;
    yy_scan_string(inputString.c_str());
}

void Hanami::PolicyParserInterface::scan_end()
{
    yy_delete_buffer(YY_CURRENT_BUFFER);
}


