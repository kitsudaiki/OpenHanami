/**
 *  @file       jwt.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  Apache License Version 2.0
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

#include <libKitsunemimiJwt/jwt.h>

#include <libKitsunemimiCrypto/signing.h>
#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiCommon/methods/string_methods.h>

namespace Kitsunemimi
{

/**
 * @brief constructor
 *
 * @param signingKey key for signing and validation
 */
Jwt::Jwt(const CryptoPP::SecByteBlock &signingKey)
{
    m_signingKey = signingKey;
}

/**
 * @brief create a new HS256-Token
 *
 * @param result reference for the resulting token
 * @param payload payload which has to be signed
 * @param validSeconds timespan in second in which the token is valid.
 *                     If value is 0, the token doesn't expire
 * @param error reference for error-output
 *
 * @return true, if successfull, else false
 */
bool
Jwt::create_HS256_Token(std::string &result,
                        JsonItem &payload,
                        const u_int32_t validSeconds,
                        ErrorContainer &error)
{
    LOG_DEBUG("Create new HS256-JWT-Token");

    // convert header
    const std::string header = "{\"alg\":\"HS256\",\"typ\":\"JWT\"}";
    std::string headerBase64;
    encodeBase64(headerBase64, header.c_str(), header.size());
    base64ToBase64Url(headerBase64);
    result = headerBase64;

    // add timestamps
    addTimesToPayload(payload, validSeconds);

    // convert payload
    std::string payloadBase64;
    const std::string payloadString = payload.toString();
    encodeBase64(payloadBase64, payloadString.c_str(), payloadString.size());
    base64ToBase64Url(payloadBase64);
    result += "." + payloadBase64;

    // create signature
    std::string secretHmac;
    if(create_HMAC_SHA256(secretHmac, result, m_signingKey, error) == false)
    {
        error.addMeesage("Failed to create HMAC");
        return false;
    }
    base64ToBase64Url(secretHmac);
    result += "." + secretHmac;

    return true;
}

/**
 * @brief get payload of token without validation
 *
 * @param parsedResult reference for returning the payload of the token, if valid
 * @param token token to parse
 * @param error reference for error-output
 *
 * @return true, if token is valid, else false
 */
bool
getJwtTokenPayload(JsonItem &parsedResult,
                   const std::string &token,
                   ErrorContainer &error)
{
    // split token
    std::vector<std::string> tokenParts;
    splitStringByDelimiter(tokenParts, token, '.');
    if(tokenParts.size() != 3)
    {
        error.addMeesage("Token is broken");
        LOG_ERROR(error);
        return false;
    }

    // convert and parse payload
    std::string payloadString = tokenParts.at(1);
    base64UrlToBase64(payloadString);
    decodeBase64(payloadString, payloadString);
    if(parsedResult.parse(payloadString, error) == false)
    {
        error.addMeesage("Token-payload is broken");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief validate a jwt-Token
 *
 * @param resultPayload reference for returning the payload of the token, if valid
 * @param token token to validate
 * @param publicError error-string for output to user
 * @param error reference for error-output
 *
 * @return true, if token is valid, else false
 */
bool
Jwt::validateToken(JsonItem &resultPayload,
                   const std::string &token,
                   std::string &publicError,
                   ErrorContainer &error)
{
    LOG_DEBUG("Validate JWT-Token");

    // precheck token
    if(token.size() == 0)
    {
        error.addMeesage("Token is empty");
        LOG_ERROR(error);
        return false;
    }

    // filter relevant part from the token
    std::vector<std::string> tokenParts;
    splitStringByDelimiter(tokenParts, token, '.');
    if(tokenParts.size() != 3)
    {
        publicError = "Token is broken";
        error.addMeesage(publicError);
        LOG_ERROR(error);
        return false;
    }
    const std::string relevantPart = tokenParts.at(0) + "." + tokenParts.at(1);

    // convert header the get information
    JsonItem header;
    std::string headerString = tokenParts.at(0);
    base64UrlToBase64(headerString);
    decodeBase64(headerString, headerString);
    if(header.parse(headerString, error) == false)
    {
        publicError = "Token is broken";
        error.addMeesage("Token-header is broken");
        LOG_ERROR(error);
        return false;
    }

    // check if header is complete
    if(header.contains("alg") == false
            || header.contains("typ") == false)
    {
        publicError = "Token is broken";
        error.addMeesage("Token-header is not a valid JWT-header");
        LOG_ERROR(error);
        return false;
    }

    // get values from header
    const std::string alg = header["alg"].getString();
    const std::string typ = header["typ"].getString();

    // check type
    if(typ != "JWT")
    {
        publicError = "Token is not a JWT-token";
        error.addMeesage(publicError);
        LOG_ERROR(error);
        return false;
    }

    // try to validate the jwt-token
    if(validateSignature(alg, relevantPart, tokenParts.at(2), error) ==  false)
    {
        publicError = "Token is invalid";
        error.addMeesage("Validation of JWT-token failed.");
        LOG_ERROR(error);
        return false;
    }

    // convert payload for output
    std::string payloadString = tokenParts.at(1);
    base64UrlToBase64(payloadString);
    decodeBase64(payloadString, payloadString);
    if(resultPayload.parse(payloadString, error) == false)
    {
        publicError = "Token is broken";
        error.addMeesage("Jwt-payload is broken");
        LOG_ERROR(error);
        return false;
    }

    // check time-stamps within the payload
    if(checkTimesInPayload(resultPayload, error) ==  false)
    {
        publicError = "Token is expired";
        error.addMeesage("Time-check of JWT-token failed.");
        LOG_ERROR(error);
        return false;
    }

    LOG_DEBUG("Validation of JWT-Token was successfull.");

    return true;
}

/**
 * @brief try to validate the JWT-token based on the used algorithm
 *
 * @param alg type of the jwt-algorithm for the validation
 * @param payload reference for returning the payload of the token, if valid
 * @param token token to validate
 * @param error reference for error-output
 *
 * @return true, if token can be validated and is valid, else false
 */
bool
Jwt::validateSignature(const std::string &alg,
                       const std::string &relevantPart,
                       const std::string &signature,
                       ErrorContainer &error)
{
    if(alg == "HS256") {
        return validate_HS256_Signature(relevantPart, signature, error);
    }

    error.addMeesage("Jwt-token can not be validated, because the algorithm \"" + alg + "\"\n"
                     "is not supported by this library or doesn't even exist.");
    LOG_ERROR(error);
    return false;
}

/**
 * @brief validate a HS256-Token
 *
 * @param payload reference for returning the payload of the token, if valid
 * @param token token to validate
 * @param error reference for error-output
 *
 * @return true, if token is valid, else false
 */
bool
Jwt::validate_HS256_Signature(const std::string &relevantPart,
                              const std::string &signature,
                              ErrorContainer &error)
{
    // create hmac again
    std::string compare;
    if(create_HMAC_SHA256(compare, relevantPart, m_signingKey, error) == false)
    {
        error.addMeesage("Failed to create HMAC");
        return false;
    }
    base64ToBase64Url(compare);

    // compare new create hmac-value with the one from the token
    if(compare.size() == signature.size()
            && CRYPTO_memcmp(compare.c_str(), signature.c_str(), compare.size()) == 0)
    {
        return true;
    }

    error.addMeesage("Check HS256-signature of the JWT-Token failed");
    LOG_ERROR(error);

    return false;
}

/**
 * @brief add timestamps to jwt-token-payload
 *
 * @param payload token-payload to write times into it
 * @param validSeconds timespan in second in which the token is valid.
 *                     If value is 0, the token doesn't expire
 */
void
Jwt::addTimesToPayload(JsonItem &payload,
                       const u_int32_t validSeconds)
{
    // get times
    const long nowSec = getTimeSinceEpoch();
    const long exp = nowSec + validSeconds;

    // add times to payload
    // IMPORTANT: use of force-flag to ensure that are no predefined values are allowed
    if(validSeconds != 0) {
        payload.insert("exp", exp, true);
    }
    payload.insert("nbf", nowSec, true);
    payload.insert("iat", nowSec, true);
}

/**
 * @brief check timestamps within the token-payload
 *
 * @param payload payload of the jwt-token
 * @param error reference for error-output
 *
 * @return true, if times are valid, else false
 */
bool
Jwt::checkTimesInPayload(const JsonItem &payload,
                         ErrorContainer &error)
{
    const long nowSec = getTimeSinceEpoch();

    // check if token is already expired
    if(payload.contains("exp"))
    {
        const long exp = payload.get("exp").getLong();
        if(exp < nowSec)
        {
            error.addMeesage("Jwt-token is expired");
            LOG_ERROR(error);
            return false;
        }
    }

    // check if token is already allowed
    if(payload.contains("nbf"))
    {
        const long nbf = payload.get("nbf").getLong();
        if(nbf > nowSec)
        {
            error.addMeesage("Jwt-token is at the current time still not allowed");
            LOG_ERROR(error);
            return false;
        }
    }

    return true;
}

/**
 * @brief get time since epoch in seconds
 */
long
Jwt::getTimeSinceEpoch()
{
    const std::chrono::high_resolution_clock::time_point now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
}

}  // namespace Kitsunemimi
