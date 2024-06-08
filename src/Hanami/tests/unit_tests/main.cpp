/**
 * @file        main.cpp
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

#include "core/cluster_test.h"
#include "core/dataset_io_test.h"
#include "database/audit_log_table_test.h"
#include "database/checkpoint_table_test.h"
#include "database/cluster_table_test.h"
#include "database/dataset_table_test.h"
#include "database/error_log_table_test.h"
#include "database/projects_table_test.h"
#include "database/tempfile_table_test.h"
#include "database/users_table_test.h"

int
main()
{
    Hanami::initConsoleLogger(false);

    Cluster_Init_Test();
    DataSetIO_Test();

    ClusterTable_Test();
    CheckpointTable_Test();
    UserTable_Test();
    ProjectTable_Test();
    DataSetTable_Test();
    TempfileTable_Test();

    AuditLogTable_Test();
    ErrorLogTable_Test();

    return 0;
}
