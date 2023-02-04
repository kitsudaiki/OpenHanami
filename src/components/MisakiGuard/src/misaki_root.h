/**
 * @file        misaki_root.h
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

#ifndef MISAKIGUARD_MISAKIROOT_H
#define MISAKIGUARD_MISAKIROOT_H

#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiHanamiPolicies/policy.h>
#include <database/users_table.h>
#include <database/projects_table.h>

class MisakiRoot
{
public:
    MisakiRoot();

    bool init(Kitsunemimi::ErrorContainer &error);

    static Kitsunemimi::Jwt* jwt;
    static UsersTable* usersTable;
    static ProjectsTable* projectsTable;
    static Kitsunemimi::Sakura::SqlDatabase* database;
    static Kitsunemimi::Hanami::Policy* policies;

private:
    bool initDatabase(Kitsunemimi::ErrorContainer &error);
    bool initPolicies(Kitsunemimi::ErrorContainer &error);
    bool initJwt(Kitsunemimi::ErrorContainer &error);
};

#endif // MISAKIGUARD_MISAKIROOT_H
