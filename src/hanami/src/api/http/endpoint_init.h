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
#include <api/http/auth/create_token_v1_0.h>
#include <api/http/auth/renew_token_v1_0.h>
#include <api/http/checkpoint/delete_checkpoint_v1_0.h>
#include <api/http/checkpoint/get_checkpoint_v1_0.h>
#include <api/http/checkpoint/list_checkpoint_v1_0.h>
#include <api/http/cluster/create_cluster_v1_0.h>
#include <api/http/cluster/delete_cluster_v1_0.h>
#include <api/http/cluster/list_cluster_v1_0.h>
#include <api/http/cluster/load_cluster_v1_0.h>
#include <api/http/cluster/save_cluster_v1_0.h>
#include <api/http/cluster/set_cluster_mode_v1_0.h>
#include <api/http/cluster/show_cluster_v1_0.h>
#include <api/http/cluster/switch_hosts_v1_0.h>
#include <api/http/dataset/check_dataset_v1_0.h>
#include <api/http/dataset/csv/create_csv_dataset_v1_0.h>
#include <api/http/dataset/csv/finalize_csv_dataset_v1_0.h>
#include <api/http/dataset/delete_dataset_v1_0.h>
#include <api/http/dataset/download_dataset_content_v1_0.h>
#include <api/http/dataset/get_dataset_v1_0.h>
#include <api/http/dataset/get_progress_dataset_v1_0.h>
#include <api/http/dataset/list_dataset_v1_0.h>
#include <api/http/dataset/mnist/create_mnist_dataset_v1_0.h>
#include <api/http/dataset/mnist/finalize_mnist_dataset_v1_0.h>
#include <api/http/hosts/list_hosts_v1_0.h>
#include <api/http/logs/get_audit_log_v1_0.h>
#include <api/http/logs/get_error_log_v1_0.h>
#include <api/http/measurements/power_consumption_v1_0.h>
#include <api/http/measurements/speed_v1_0.h>
#include <api/http/measurements/temperature_production_v1_0.h>
#include <api/http/project/create_project_v1_0.h>
#include <api/http/project/delete_project_v1_0.h>
#include <api/http/project/get_project_v1_0.h>
#include <api/http/project/list_projects_v1_0.h>
#include <api/http/system_info/get_system_info_v1_0.h>
#include <api/http/task/create_request_task_v1_0.h>
#include <api/http/task/create_train_task_v1_0.h>
#include <api/http/task/delete_task_v1_0.h>
#include <api/http/task/list_task_v1_0.h>
#include <api/http/task/show_task_v1_0.h>
#include <api/http/threading/get_thread_mapping_v1_0.h>
#include <api/http/user/add_project_to_user_v1_0.h>
#include <api/http/user/create_user_v1_0.h>
#include <api/http/user/delete_user_v1_0.h>
#include <api/http/user/get_user_v1_0.h>
#include <api/http/user/list_user_projects_v1_0.h>
#include <api/http/user/list_users_v1_0.h>
#include <api/http/user/remove_project_from_user_v1_0.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_common/logger.h>
#include <hanami_config/config_handler.h>
#include <hanami_root.h>

const std::string tokenEndpointPath = "v1.0alpha/token";

void initEndpoints();

#endif  // HANAMI_ENDPOINT_INIT_H
