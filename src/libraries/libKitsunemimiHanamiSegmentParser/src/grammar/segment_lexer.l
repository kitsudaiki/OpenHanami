%{ /* -*- C++ -*- */
/**
 * @file       segment_lexer.l
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

# include <cerrno>
# include <climits>
# include <cstdlib>
# include <string>
# include <segment_parsing/segment_parser_interface.h>
# include <segment_parser.h>

# undef yywrap
# define yywrap() 1


// The location of the current token.
# ifdef YY_DECL
# undef YY_DECL
# endif
# define YY_DECL \
    Kitsunemimi::Hanami::SegmentParser::symbol_type segmentlex (Kitsunemimi::Hanami::SegmentParserInterface& driver)
YY_DECL;
Kitsunemimi::Hanami::location segmentloc;

%}


%option noyywrap nounput batch debug yylineno prefix="segment"

id    [a-zA-Z_0-9]*
long  -?([0-9]+)
double -?([0-9]+)\.([0-9]+)
blank [ \t]

%{
    # define YY_USER_ACTION  segmentloc.columns (yyleng);
%}

%%

%{
    segmentloc.step();
%}

{blank}+  segmentloc.step();
[\n]      {segmentloc.lines(1); return Kitsunemimi::Hanami::SegmentParser::make_LINEBREAK (segmentloc); }
"version:1"    return Kitsunemimi::Hanami::SegmentParser::make_VERSION_1 (segmentloc);
"version: 1"   return Kitsunemimi::Hanami::SegmentParser::make_VERSION_1 (segmentloc);
"segment_type" return Kitsunemimi::Hanami::SegmentParser::make_SEGMENT_TYPE (segmentloc);
"settings"     return Kitsunemimi::Hanami::SegmentParser::make_SETTINGS (segmentloc);
"bricks"       return Kitsunemimi::Hanami::SegmentParser::make_BRICKS (segmentloc);
","            return Kitsunemimi::Hanami::SegmentParser::make_COMMA (segmentloc);
":"            return Kitsunemimi::Hanami::SegmentParser::make_ASSIGN (segmentloc);

\"(\\.|[^"\\])*\" { return Kitsunemimi::Hanami::SegmentParser::make_STRING(yytext, segmentloc); }

{long}      {
    errno = 0;
    long length = strtol(yytext, NULL, 10);
    if (!(LONG_MIN <= length
        && length <= LONG_MAX
        && errno != ERANGE))
    {
        driver.error(segmentloc, "integer is out of range");
    }
    return Kitsunemimi::Hanami::SegmentParser::make_NUMBER (length, segmentloc);
}

{double}	{
    double value = strtod( yytext , NULL );
    return Kitsunemimi::Hanami::SegmentParser::make_FLOAT(value, segmentloc);
}

{id}       return Kitsunemimi::Hanami::SegmentParser::make_IDENTIFIER(yytext, segmentloc);

[a-zA-Z_0-9|\-|.]* {
    return Kitsunemimi::Hanami::SegmentParser::make_STRING_PLN(yytext, segmentloc);
}

.          driver.error(segmentloc, "invalid character");
<<EOF>>    return Kitsunemimi::Hanami::SegmentParser::make_END(segmentloc);

%%


void Kitsunemimi::Hanami::SegmentParserInterface::scan_begin(const std::string &inputString)
{
    Kitsunemimi::Hanami::location newSegmentloc;
    segmentloc = newSegmentloc;
    yy_flex_debug = m_traceParsing;
    yy_scan_string(inputString.c_str());
}

void Kitsunemimi::Hanami::SegmentParserInterface::scan_end()
{
    yy_delete_buffer(YY_CURRENT_BUFFER);
}


