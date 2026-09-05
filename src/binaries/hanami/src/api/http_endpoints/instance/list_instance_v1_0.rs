// Copyright 2022-2026 Tobias Anker <tobias.anker@kitsunemimi.moe>

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use actix_web::web::Json;
use apistos::api_operation;

use crate::config;
use crate::database::meta_instance_table;

use ainari_api::common_functions::*;
use ainari_api::errors::ErrorResponse;
use ainari_api_structs::instance_structs::*;
use ainari_api_structs::user_context::UserContext;
use ainari_clients::endpoints::*;
use ainari_clients::proxy as proxy_clients;

#[api_operation(
    tag = "instance",
    summary = "List instance",
    description = r###"List basic information of all instance from the database."###,
    error_code = 401,
    error_code = 500
)]
pub async fn list_instance(context: UserContext) -> Result<Json<InstanceListResp>, ErrorResponse> {
    // get instances from db
    let instances = meta_instance_table::list_meta_instances(&context)
        .map_err(|e| map_db_list_error("hosts", e))?;

    // get endpoints from miko
    let miko_endpoint = &config::CONFIG.miko;
    let endpoints = get_endpoints(miko_endpoint, config::CONFIG.skip_tls_verification)
        .await
        .map_err(map_ainari_error_to_api_response)?;

    // prepare response
    let mut resp = InstanceListResp {
        instances: Vec::new(),
    };

    // fill reponse
    for instance in instances {
        let uuid = convert_uuid(&instance.uuid)?;

        // get port of the instance from torii
        let proxy_uuid = convert_uuid(&instance.proxy_uuid)?;
        let proxy_resp = proxy_clients::get_proxy(
            &endpoints.torii,
            &context.token,
            &proxy_uuid,
            config::CONFIG.skip_tls_verification,
        )
        .await
        .map_err(map_ainari_error_to_api_response)?;

        // add single object to the reponse-list
        let obj = InstanceBasicResp {
            uuid,
            name: instance.name.clone(),
            proxy_port: proxy_resp.port,
        };

        resp.instances.push(obj);
    }

    Ok(Json(resp))
}
