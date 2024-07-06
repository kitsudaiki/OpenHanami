/**
 * @file        endpoint_init.h
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

#ifndef HANAMI_ENDPOINT_INIT_H
#define HANAMI_ENDPOINT_INIT_H

#include <api/endpoint_processing/http_processing/http_processing.h>
#include <api/endpoint_processing/http_server.h>
#include <api/http/auth/create_token.h>
#include <api/http/auth/renew_token.h>
#include <api/http/auth/validate_access.h>
#include <api/http/checkpoint/delete_checkpoint.h>
#include <api/http/checkpoint/get_checkpoint.h>
#include <api/http/checkpoint/list_checkpoint.h>
#include <api/http/cluster/create_cluster.h>
#include <api/http/cluster/delete_cluster.h>
#include <api/http/cluster/list_cluster.h>
#include <api/http/cluster/load_cluster.h>
#include <api/http/cluster/save_cluster.h>
#include <api/http/cluster/set_cluster_mode.h>
#include <api/http/cluster/show_cluster.h>
#include <api/http/cluster/switch_hosts.h>
#include <api/http/data_files/check_dataset.h>
#include <api/http/data_files/csv/create_csv_dataset.h>
#include <api/http/data_files/csv/finalize_csv_dataset.h>
#include <api/http/data_files/delete_dataset.h>
#include <api/http/data_files/get_dataset.h>
#include <api/http/data_files/get_progress_dataset.h>
#include <api/http/data_files/list_dataset.h>
#include <api/http/data_files/mnist/create_mnist_dataset.h>
#include <api/http/data_files/mnist/finalize_mnist_dataset.h>
#include <api/http/hosts/list_hosts.h>
#include <api/http/logs/get_audit_log.h>
#include <api/http/logs/get_error_log.h>
#include <api/http/measurements/power_consumption.h>
#include <api/http/measurements/speed.h>
#include <api/http/measurements/temperature_production.h>
#include <api/http/project/create_project.h>
#include <api/http/project/delete_project.h>
#include <api/http/project/get_project.h>
#include <api/http/project/list_projects.h>
#include <api/http/system_info/get_system_info.h>
#include <api/http/task/create_request_task.h>
#include <api/http/task/create_train_task.h>
#include <api/http/task/delete_task.h>
#include <api/http/task/list_task.h>
#include <api/http/task/show_task.h>
#include <api/http/threading/get_thread_mapping.h>
#include <api/http/user/add_project_to_user.h>
#include <api/http/user/create_user.h>
#include <api/http/user/delete_user.h>
#include <api/http/user/get_user.h>
#include <api/http/user/list_user_projects.h>
#include <api/http/user/list_users.h>
#include <api/http/user/remove_project_from_user.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_common/logger.h>
#include <hanami_config/config_handler.h>
#include <hanami_root.h>

const std::string authEndpointPath = "v1.0alpha/auth";
const std::string tokenEndpointPath = "v1.0alpha/token";

void initEndpoints();

#endif  // HANAMI_ENDPOINT_INIT_H
