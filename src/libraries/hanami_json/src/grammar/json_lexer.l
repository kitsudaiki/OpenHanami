%{ /* -*- C++ -*- */
/**
 *  @file    json_lexer.l
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
*/

# include <cerrno>
# include <climits>
# include <cstdlib>
# include <string>
# include <json_parsing/json_parser_interface.h>
# include <json_parser.h>

# undef yywrap
# define yywrap() 1


// The location of the current token.
# ifdef YY_DECL
# undef YY_DECL
# endif
# define YY_DECL \
    Kitsunemimi::JsonParser::symbol_type jsonlex (Kitsunemimi::JsonParserInterface& driver)
YY_DECL;
Kitsunemimi::location jsonloc;

%}


%option noyywrap nounput batch debug yylineno prefix="json"

id    [a-zA-Z_0-9]*
long  -?([0-9]+)
double -?([0-9]+)\.([0-9]+)
blank [ \t]

%{
    # define YY_USER_ACTION  jsonloc.columns (yyleng);
%}

%%

%{
    jsonloc.step();
%}

{blank}+  jsonloc.step();
[\n]      {jsonloc.lines(1); jsonloc.step(); }
"{"       return Kitsunemimi::JsonParser::make_EXPRESTART (jsonloc);
"}"       return Kitsunemimi::JsonParser::make_EXPREEND (jsonloc);
"["       return Kitsunemimi::JsonParser::make_BRACKOPEN (jsonloc);
"]"       return Kitsunemimi::JsonParser::make_BRACKCLOSE (jsonloc);
","       return Kitsunemimi::JsonParser::make_COMMA (jsonloc);
":"       return Kitsunemimi::JsonParser::make_ASSIGN (jsonloc);
"e+"      return Kitsunemimi::JsonParser::make_EXPONENT (jsonloc);

"true" return Kitsunemimi::JsonParser::make_BOOL_TRUE (jsonloc);
"false" return Kitsunemimi::JsonParser::make_BOOL_FALSE (jsonloc);
"null" return Kitsunemimi::JsonParser::make_NULLVAL (jsonloc);

\"(\\.|[^"\\])*\" { return Kitsunemimi::JsonParser::make_STRING(yytext, jsonloc); }

{long}      {
    errno = 0;
    long length = strtol(yytext, NULL, 10);
    if (!(LONG_MIN <= length
        && length <= LONG_MAX
        && errno != ERANGE))
    {
        driver.error(jsonloc, "integer is out of range");
    }
    return Kitsunemimi::JsonParser::make_NUMBER (length, jsonloc);
}

{double}	{
    double value = strtod( yytext , NULL );
    return Kitsunemimi::JsonParser::make_FLOAT(value, jsonloc);
}

{id}       return Kitsunemimi::JsonParser::make_IDENTIFIER(yytext, jsonloc);

.          driver.error(jsonloc, "invalid character");
<<EOF>>    return Kitsunemimi::JsonParser::make_END(jsonloc);

%%


void Kitsunemimi::JsonParserInterface::scan_begin(const std::string &inputString)
{
    Kitsunemimi::location newJsonloc;
    jsonloc = newJsonloc;
    yy_flex_debug = m_traceParsing;
    yy_scan_string(inputString.c_str());
}

void Kitsunemimi::JsonParserInterface::scan_end()
{
    yy_delete_buffer(YY_CURRENT_BUFFER);
}


