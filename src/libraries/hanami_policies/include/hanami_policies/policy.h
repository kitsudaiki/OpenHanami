/**
 * @file        policy.h
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

#ifndef POLICY_H
#define POLICY_H

#include <hanami_common/enums.h>
#include <hanami_common/logger.h>

#include <map>
#include <string>
#include <vector>

namespace Hanami
{
class Policy_Test;

struct PolicyEntry {
    std::vector<std::string> getRules;
    std::vector<std::string> postRules;
    std::vector<std::string> putRules;
    std::vector<std::string> deleteRules;
};

class Policy
{
   public:
    static Policy* getInstance()
    {
        if (instance == nullptr) {
            instance = new Policy();
        }
        return instance;
    }
    ~Policy();

    bool parse(const std::string& input, ErrorContainer& error);

    bool checkUserAgainstPolicy(const std::string& endpoint,
                                const HttpRequestType type,
                                const std::string& role);

   private:
    Policy();
    static Policy* instance;

    std::map<std::string, PolicyEntry> m_policyRules;

    bool checkEntry(const PolicyEntry& entry, const HttpRequestType type, const std::string& role);
    bool checkRuleList(const std::vector<std::string>& rules, const std::string& compare);

    friend Policy_Test;
};

}  // namespace Hanami

#endif  // POLICY_H
