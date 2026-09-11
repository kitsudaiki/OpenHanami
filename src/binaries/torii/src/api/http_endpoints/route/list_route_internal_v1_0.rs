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

use crate::database::route_table;

use ainari_api::common_functions::*;
use ainari_api::errors::ErrorResponse;
use ainari_api_structs::route_structs::*;
use ainari_api_structs::user_context::UserContext;

#[api_operation(
    tag = "route",
    summary = "List route",
    description = r###"List basic information of all route from the database."###,
    error_code = 401,
    error_code = 500
)]
pub async fn list_route_internal(context: UserContext) -> Result<Json<RouteListResp>, ErrorResponse> {
    let routes = route_table::list_routes(&context).map_err(|e| map_db_list_error("routes", e))?;

    let mut resp = RouteListResp { routes: Vec::new() };

    for route in routes {
        let uuid = convert_uuid(&route.uuid)?;
        let instance_uuid = convert_uuid(&route.instance_uuid)?;
        let obj = RouteBasicResp {
            uuid,
            port: route.port as u16,
            target_address: route.target_address,
            instance_uuid,
        };

        resp.routes.push(obj);
    }

    Ok(Json(resp))
}
