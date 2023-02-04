/**
 *  @file       jwt_test.cpp
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

#include "jwt_test.h"

#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{

JWT_Test::JWT_Test()
    : Kitsunemimi::CompareTestHelper("JWT_Test")
{
    Kitsunemimi::initConsoleLogger(true);
    create_validate_HS256_Token_test();
}

/**
 * @brief create_validate_HS256_Token_test
 */
void
JWT_Test::create_validate_HS256_Token_test()
{
    // create test-secte
    const std::string testSecret = "your-256-bit-secret";
    CryptoPP::SecByteBlock key((unsigned char*)testSecret.c_str(), testSecret.size());

    // init test-class
    Jwt jwt(key);

    // prepare test-payload
    const std::string testPayload = "{"
                                    "    \"sub\":\"1234567890\","
                                    "    \"name\":\"Test-User\","
                                    "    \"iat\":1516239022"
                                    "}";
    JsonItem payloadJson;
    ErrorContainer error;
    std::string publicError = "";
    assert(payloadJson.parse(testPayload, error));
    error._errorMessages.clear();

    // test token-creation
    std::string token;
    TEST_EQUAL(jwt.create_HS256_Token(token, payloadJson, 1000, error), true);
    LOG_DEBUG("token: " + token);

    // test token-validation with valid token
    JsonItem resultPayloadJson;
    TEST_EQUAL(jwt.validateToken(resultPayloadJson, token, publicError, error), true);
    TEST_EQUAL(resultPayloadJson.get("name").getString(), "Test-User");
    error._errorMessages.clear();

    // test getter for token-payload without validation
    JsonItem resultPayloadJson2;
    TEST_EQUAL(getJwtTokenPayload(resultPayloadJson2, token, error), true);
    TEST_EQUAL(resultPayloadJson2.get("name").getString(), "Test-User");
    error._errorMessages.clear();

    // test token-validation with broken token
    token[token.size() - 5] = 'x';
    TEST_EQUAL(jwt.validateToken(resultPayloadJson, token, publicError, error), false);
    error._errorMessages.clear();
}

}
