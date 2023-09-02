%{ /* -*- C++ -*-*/
/**
 *  @file    ini_lexer.l
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
*/

# include <cerrno>
# include <climits>
# include <cstdlib>
# include <string>
# include <ini_parsing/ini_parser_interface.h>
# include <ini_parser.h>

# undef yywrap
# define yywrap() 1

// The location of the current token.
# ifdef YY_DECL
# undef YY_DECL
# endif
# define YY_DECL \
    Kitsunemimi::IniParser::symbol_type inilex (Kitsunemimi::IniParserInterface& driver)
YY_DECL;

Kitsunemimi::location iniloc;
%}


%option noyywrap nounput batch debug yylineno prefix="ini"
id    [%\-+\/@.:a-zA-Z_0-9]*
long   -?([0-9]+)
double -?([0-9]+)\.([0-9]+)
blank [ \t]

%{
    # define YY_USER_ACTION  iniloc.columns (yyleng);
%}

%%

%{
    iniloc.step();
%}

#.*$        iniloc.step();
{blank}+    iniloc.step();
[\n]        {iniloc.lines(1); return Kitsunemimi::IniParser::make_LINEBREAK (iniloc); }
"true"      return Kitsunemimi::IniParser::make_BOOL_TRUE (iniloc);
"false"     return Kitsunemimi::IniParser::make_BOOL_FALSE (iniloc);
"True"      return Kitsunemimi::IniParser::make_BOOL_TRUE (iniloc);
"False"     return Kitsunemimi::IniParser::make_BOOL_FALSE (iniloc);
"="         return Kitsunemimi::IniParser::make_EQUAL (iniloc);
"["         return Kitsunemimi::IniParser::make_BRACKOPEN (iniloc);
"]"         return Kitsunemimi::IniParser::make_BRACKCLOSE (iniloc);
","         return Kitsunemimi::IniParser::make_COMMA (iniloc);
\"(\\.|[^"\\])*\" {
            return Kitsunemimi::IniParser::make_STRING(yytext, iniloc);
}


{long}      {
    errno = 0;
    long length = strtol(yytext, NULL, 10);
    if (!(LONG_MIN <= length
        && length <= LONG_MAX
        && errno != ERANGE))
    {
        driver.error(iniloc, "integer is out of range");
    }
    return Kitsunemimi::IniParser::make_NUMBER (length, iniloc);
}

{double}	{
    double value = strtod( yytext , NULL );
    return Kitsunemimi::IniParser::make_FLOAT(value, iniloc);
}

{id} return Kitsunemimi::IniParser::make_IDENTIFIER(yytext, iniloc);


[a-zA-Z_0-9|\-|.]* {
    return Kitsunemimi::IniParser::make_STRING_PLN(yytext, iniloc);
}

.          driver.error(iniloc, "invalid character");
<<EOF>>    return Kitsunemimi::IniParser::make_END(iniloc);

%%


void Kitsunemimi::IniParserInterface::scan_begin(const std::string &inputString)
{
    Kitsunemimi::location newLoc;
    iniloc = newLoc;
    yy_flex_debug = m_traceParsing;
    yy_scan_string(inputString.c_str());
}

void Kitsunemimi::IniParserInterface::scan_end()
{
    yy_delete_buffer(YY_CURRENT_BUFFER);
}
