/**
 *  @file       signing_test.cpp
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

#include "signing_test.h"

#include <libKitsunemimiCrypto/signing.h>
#include <libKitsunemimiCrypto/common.h>

namespace Kitsunemimi
{

Signing_Test::Signing_Test()
    : Kitsunemimi::CompareTestHelper("Signing_Test")
{
    create_verify_HMAC_Sha256();
}

/**
 * @brief create_verify_HMAC_Sha256
 */
void
Signing_Test::create_verify_HMAC_Sha256()
{
    Kitsunemimi::ErrorContainer error;
    std::string testData = "this is a test-string";
    CryptoPP::SecByteBlock key((unsigned char*)"asdf", 4);
    std::string resultingHmac;

    TEST_EQUAL(create_HMAC_SHA256(resultingHmac, testData, key, error), true);
    TEST_EQUAL(create_HMAC_SHA256(resultingHmac, "", key, error), false);
    CryptoPP::SecByteBlock emptyey((unsigned char*)"", 0);
    TEST_EQUAL(create_HMAC_SHA256(resultingHmac, testData, emptyey, error), false);

    TEST_EQUAL(resultingHmac, "58yA7QZ+I1opAOhoaWLwj4wnxUKz5xaYafjE+Vcb6c4=");

    // positive validation test
    TEST_EQUAL(verify_HMAC_SHA256(testData, resultingHmac, key), true);

    // negative validation test
    testData[2] = 'a';
    TEST_EQUAL(verify_HMAC_SHA256(testData, resultingHmac, key), false);
}

} // namespace Kitsunemimi
