/**
 *  @file       signing.h
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

#ifndef SIGNING_H
#define SIGNING_H

#include <string>
#include <iostream>
#include <openssl/hmac.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/ec.h>
#include <cryptopp/aes.h>

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{

//--------------------------------------------------------------------------------------------------
// HMAC
//--------------------------------------------------------------------------------------------------
bool create_HMAC_SHA256(std::string &result,
                        const std::string &input,
                        const CryptoPP::SecByteBlock &key,
                        Kitsunemimi::ErrorContainer &error);

bool verify_HMAC_SHA256(const std::string &input,
                        const std::string &hmac,
                        const CryptoPP::SecByteBlock &key);

} // namespace Kitsunemimi

#endif // SIGNING_H
