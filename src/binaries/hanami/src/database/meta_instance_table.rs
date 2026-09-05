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

use chrono::Utc;
use diesel::connection::SimpleConnection;
use diesel::dsl::count_star;
use diesel::prelude::*;
use std::error::Error;
use uuid::Uuid;

use crate::database::db_handle;

use ainari_api_structs::user_context::UserContext;
use ainari_common::enums;

// Define the schema for meta_instances table
table! {
    meta_instances (uuid) {
        uuid -> Varchar,
        name -> Varchar,
        sakura_host_uuid -> Varchar,
        proxy_uuid -> Varchar,
        owner_id -> Varchar,
        project_id -> Varchar,
        status -> Varchar,
        created_at -> Varchar,
        created_by -> Varchar,
        updated_at -> Varchar,
        updated_by -> Varchar,
        deleted_at -> Nullable<Varchar>,
        deleted_by -> Nullable<Varchar>,
    }
}

/// Represents an entry in the meta_instances table.
/// This struct contains all the fields required to create, query, and update meta instance records.
#[derive(Insertable, Queryable, Selectable, Debug, PartialEq, Clone)]
#[diesel(table_name = meta_instances)]
pub struct MetaInstanceEntry {
    pub uuid: String,
    pub name: String,
    pub sakura_host_uuid: String,
    pub proxy_uuid: String,
    pub owner_id: String,
    pub project_id: String,
    pub status: String,
    pub created_at: String,
    pub created_by: String,
    pub updated_at: String,
    pub updated_by: String,
    pub deleted_at: Option<String>,
    pub deleted_by: Option<String>,
}

/// Initializes the meta_instances table in the database if it doesn't exist.
///
/// This function creates the table with the appropriate schema and constraints.
/// It's typically called during application startup to ensure the required tables exist.
pub fn init_meta_instance_table() -> Result<(), Box<dyn Error>> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    conn.batch_execute(
        "CREATE TABLE IF NOT EXISTS meta_instances (
        uuid VARCHAR(40) PRIMARY KEY,
        name VARCHAR(256),
        sakura_host_uuid VARCHAR(40),
        proxy_uuid VARCHAR(40),
        owner_id VARCHAR(256),
        project_id VARCHAR(256),
        status VARCHAR(8),
        created_at VARCHAR(64),
        created_by VARCHAR(256),
        updated_at VARCHAR(64),
        updated_by VARCHAR(256),
        deleted_at VARCHAR(64),
        deleted_by VARCHAR(256)
    );",
    )?;

    Ok(())
}

/// Adds a new meta instance to the database.
///
/// This function creates a new MetaInstanceEntry with the provided parameters and inserts it into the database.
/// The status is set to "ACTIVE" and timestamps are set to the current time.
///
/// # Arguments
/// * `meta_instance_uuid` - The unique identifier for the meta instance
/// * `instance_name` - The name of the meta instance
/// * `sakura_host_uuid` - The UUID of the Sakura host associated with this instance
/// * `proxy_uuid` - The UUID of the proxy associated with this instance
/// * `context` - The user context containing information about the user and project
///
/// # Returns
/// A QueryResult indicating the number of rows affected
pub fn add_new_meta_instance(
    meta_instance_uuid: &Uuid,
    instance_name: &str,
    sakura_host_uuid: &Uuid,
    proxy_uuid: &Uuid,
    context: &UserContext,
) -> QueryResult<usize> {
    let meta_instance = MetaInstanceEntry {
        uuid: meta_instance_uuid.to_string().clone(),
        name: instance_name.to_string().clone(),
        sakura_host_uuid: sakura_host_uuid.to_string().clone(),
        proxy_uuid: proxy_uuid.to_string().clone(),
        owner_id: context.user_id.clone(),
        project_id: context.project_id.clone(),
        status: "ACTIVE".to_string(),
        created_at: Utc::now().to_rfc3339(),
        created_by: context.user_id.clone(),
        updated_at: Utc::now().to_rfc3339(),
        updated_by: context.user_id.clone(),
        deleted_at: None,
        deleted_by: None,
    };

    add_meta_instance(&meta_instance)
}

/// Adds a meta instance to the database.
///
/// This is a helper function that performs the actual insertion of a MetaInstanceEntry into the database.
///
/// # Arguments
/// * `meta_instance` - The MetaInstanceEntry to be inserted
///
/// # Returns
/// A QueryResult indicating the number of rows affected
pub fn add_meta_instance(meta_instance: &MetaInstanceEntry) -> QueryResult<usize> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::meta_instances::dsl::*;
    diesel::insert_into(meta_instances)
        .values(meta_instance)
        .execute(&mut *conn)
}

/// Retrieves a meta instance from the database.
///
/// This function queries the database for a meta instance with the specified UUID and checks the user's permissions.
/// Only active instances are returned, and the query is filtered based on the user's role and project membership.
///
/// # Arguments
/// * `meta_instance_uuid` - The UUID of the meta instance to retrieve
/// * `context` - The user context containing information about the user and their permissions
///
/// # Returns
/// A Result containing the MetaInstanceEntry if found, or a DbError if not found or an error occurs
pub fn get_meta_instance(
    meta_instance_uuid: &Uuid,
    context: &UserContext,
) -> Result<MetaInstanceEntry, enums::DbError> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::meta_instances::dsl::*;

    let mut query = meta_instances
        .filter(
            uuid.eq(meta_instance_uuid.to_string())
                .and(status.eq("ACTIVE")),
        )
        .into_boxed();

    // Apply permission-based filtering
    if context.is_admin != true.to_string() {
        query = query.filter(project_id.eq(context.project_id.clone()));
        if context.is_project_admin != true.to_string() {
            query = query.filter(owner_id.eq(context.user_id.clone()));
        }
    }

    match query
        .select(MetaInstanceEntry::as_select())
        .first::<MetaInstanceEntry>(&mut *conn)
    {
        Ok(meta_instance) => Ok(meta_instance),
        Err(diesel::result::Error::NotFound) => Err(enums::DbError::NotFound),
        Err(e) => {
            log::error!("Database-error: {e:?}");
            Err(enums::DbError::InternalError)
        }
    }
}

/// Lists all meta instances that the user has access to.
///
/// This function retrieves all active meta instances and applies permission-based filtering.
/// The results are filtered based on the user's role and project membership.
///
/// # Arguments
/// * `context` - The user context containing information about the user and their permissions
///
/// # Returns
/// A QueryResult containing a vector of MetaInstanceEntry objects
#[allow(dead_code)]
pub fn list_meta_instances(context: &UserContext) -> QueryResult<Vec<MetaInstanceEntry>> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::meta_instances::dsl::*;

    let mut query = meta_instances.filter(status.eq("ACTIVE")).into_boxed();

    // Apply permission-based filtering
    if context.is_admin != true.to_string() {
        query = query.filter(project_id.eq(context.project_id.clone()));
        if context.is_project_admin != true.to_string() {
            query = query.filter(owner_id.eq(context.user_id.clone()));
        }
    }

    query
        .select(MetaInstanceEntry::as_select())
        .load(&mut *conn)
}

/// Counts the number of meta instances that the user has access to.
///
/// This function counts all active meta instances and applies permission-based filtering.
/// The count is filtered based on the user's role and project membership.
///
/// # Arguments
/// * `context` - The user context containing information about the user and their permissions
///
/// # Returns
/// A QueryResult containing the count of meta instances as an i64
pub fn count_meta_instances(context: &UserContext) -> QueryResult<i64> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::meta_instances::dsl::*;

    let mut query = meta_instances.filter(status.eq("ACTIVE")).into_boxed();

    // Apply permission-based filtering
    query = query.filter(project_id.eq(context.project_id.clone()));
    query = query.filter(owner_id.eq(context.user_id.clone()));

    query.select(count_star()).first::<i64>(&mut *conn)
}

/// Force deletes a meta instance from the database.
///
/// This function marks a meta instance as deleted without checking permissions.
/// It's intended for system-level operations where permission checks are not required.
///
/// # Arguments
/// * `meta_instance_uuid` - The UUID of the meta instance to delete
///
/// # Returns
/// A Result indicating success or an error
pub fn force_delete_meta_instance(meta_instance_uuid: &Uuid) -> Result<(), enums::DbError> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::meta_instances::dsl::*;
    match diesel::update(meta_instances.filter(uuid.eq(meta_instance_uuid.to_string())))
        .set((
            status.eq("DELETED"),
            deleted_at.eq(Utc::now().to_rfc3339()),
            deleted_by.eq("HOST_INIT"),
        ))
        .execute(&mut *conn)
    {
        Ok(_) => Ok(()),
        Err(diesel::result::Error::NotFound) => Err(enums::DbError::NotFound),
        Err(e) => {
            log::error!("Database-error: {e:?}");
            Err(enums::DbError::InternalError)
        }
    }
}

/// Deletes a meta instance from the database.
///
/// This function marks a meta instance as deleted after verifying that the user has permission to delete it.
/// It first checks if the instance exists and if the user has the necessary permissions.
///
/// # Arguments
/// * `meta_instance_uuid` - The UUID of the meta instance to delete
/// * `context` - The user context containing information about the user and their permissions
///
/// # Returns
/// A Result indicating success or an error
pub fn delete_meta_instance(
    meta_instance_uuid: &Uuid,
    context: &UserContext,
) -> Result<(), enums::DbError> {
    // Verify the meta instance exists and the user has permission to delete it
    get_meta_instance(meta_instance_uuid, context)?;

    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::meta_instances::dsl::*;
    match diesel::update(meta_instances.filter(uuid.eq(meta_instance_uuid.to_string())))
        .set((
            status.eq("DELETED"),
            deleted_at.eq(Utc::now().to_rfc3339()),
            deleted_by.eq(context.user_id.clone()),
        ))
        .execute(&mut *conn)
    {
        Ok(_) => Ok(()),
        Err(diesel::result::Error::NotFound) => Err(enums::DbError::NotFound),
        Err(e) => {
            log::error!("Database-error: {e:?}");
            Err(enums::DbError::InternalError)
        }
    }
}

/// Deletes all meta instances from the database.
///
/// This function marks all active meta instances as deleted without checking permissions.
/// It's intended for system-level operations where permission checks are not required.
///
/// # Returns
/// A Result indicating success or an error
#[allow(dead_code)]
pub fn delete_all_meta_instance() -> Result<(), enums::DbError> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::meta_instances::dsl::*;
    match diesel::update(meta_instances.filter(status.eq("ACTIVE")))
        .set((
            status.eq("DELETED"),
            deleted_at.eq(Utc::now().to_rfc3339()),
            deleted_by.eq("AINARI_START"),
        ))
        .execute(&mut *conn)
    {
        Ok(_) => Ok(()),
        Err(diesel::result::Error::NotFound) => Err(enums::DbError::NotFound),
        Err(e) => {
            log::error!("Database-error: {e:?}");
            Err(enums::DbError::InternalError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn hard_delete_meta_instance(meta_instance_uuid: &Uuid) {
        use self::meta_instances::dsl::*;
        let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
        let _ = diesel::delete(meta_instances.filter(uuid.eq(meta_instance_uuid.to_string())))
            .execute(&mut *conn);
    }

    #[test]
    #[serial]
    fn test_add_get_meta_instance() {
        let _ = init_meta_instance_table();
        let uuid1 = Uuid::new_v4();
        let name = "test-instance".to_string();
        let sakura_host_uuid1 = Uuid::new_v4();
        let proxy_uuid1 = Uuid::new_v4();

        let project_id = "test-project".to_string();
        let owner_id = "test-user".to_string();
        let context = UserContext {
            token: "".to_string(),
            user_id: owner_id.clone(),
            project_id: project_id.clone(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };

        let meta_instance = MetaInstanceEntry {
            uuid: uuid1.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_meta_instance(&uuid1);

        add_meta_instance(&meta_instance).unwrap();
        match get_meta_instance(&uuid1, &context) {
            Ok(retrieved_meta_instance) => {
                assert_eq!(retrieved_meta_instance.uuid, meta_instance.uuid);
                assert_eq!(retrieved_meta_instance.proxy_uuid, meta_instance.proxy_uuid);
                assert_eq!(
                    retrieved_meta_instance.sakura_host_uuid,
                    meta_instance.sakura_host_uuid
                );
                assert_eq!(retrieved_meta_instance.owner_id, meta_instance.owner_id);
                assert_eq!(retrieved_meta_instance.project_id, meta_instance.project_id);
                assert_eq!(retrieved_meta_instance.status, meta_instance.status);
                assert_eq!(retrieved_meta_instance.created_by, meta_instance.created_by);
                assert_eq!(retrieved_meta_instance.updated_by, meta_instance.updated_by);
                assert_eq!(retrieved_meta_instance.deleted_at, meta_instance.deleted_at);
                assert_eq!(retrieved_meta_instance.deleted_by, meta_instance.deleted_by);
            }
            Err(_) => {
                assert_eq!(true, false);
            }
        };

        hard_delete_meta_instance(&uuid1);
    }

    #[test]
    #[serial]
    fn test_list_meta_instances() {
        let _ = init_meta_instance_table();
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();
        let name = "test-instance".to_string();
        let sakura_host_uuid1 = Uuid::new_v4();
        let proxy_uuid1 = Uuid::new_v4();

        let project_id = "test-project".to_string();
        let owner_id = "test-user".to_string();
        let context = UserContext {
            token: "".to_string(),
            user_id: owner_id.clone(),
            project_id: project_id.clone(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };

        let meta_instance1 = MetaInstanceEntry {
            uuid: uuid1.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let meta_instance2 = MetaInstanceEntry {
            uuid: uuid2.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "DELETED".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_meta_instance(&uuid1);
        hard_delete_meta_instance(&uuid2);

        add_meta_instance(&meta_instance1).unwrap();
        add_meta_instance(&meta_instance2).unwrap();
        let meta_instances = list_meta_instances(&context).unwrap();
        assert_eq!(meta_instances.len(), 1);
        hard_delete_meta_instance(&uuid1);
        hard_delete_meta_instance(&uuid2);
    }

    #[test]
    #[serial]
    fn test_delete_meta_instance() {
        let _ = init_meta_instance_table();
        let uuid1 = Uuid::new_v4();
        let name = "test-instance".to_string();
        let sakura_host_uuid1 = Uuid::new_v4();
        let proxy_uuid1 = Uuid::new_v4();

        let project_id = "test-project".to_string();
        let owner_id = "test-user".to_string();
        let context = UserContext {
            token: "".to_string(),
            user_id: owner_id.clone(),
            project_id: project_id.clone(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };

        let meta_instance = MetaInstanceEntry {
            uuid: uuid1.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_meta_instance(&uuid1);

        add_meta_instance(&meta_instance).unwrap();
        let _ = delete_meta_instance(&uuid1, &context);
        let result = get_meta_instance(&uuid1, &context);
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_count_meta_instances() {
        let _ = init_meta_instance_table();
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();
        let uuid3 = Uuid::new_v4();
        let name = "test-instance".to_string();
        let sakura_host_uuid1 = Uuid::new_v4();
        let proxy_uuid1 = Uuid::new_v4();

        let project_id = "test-project".to_string();
        let owner_id = "test-user".to_string();
        let context = UserContext {
            token: "".to_string(),
            user_id: owner_id.clone(),
            project_id: project_id.clone(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };

        let meta_instance1 = MetaInstanceEntry {
            uuid: uuid1.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let meta_instance2 = MetaInstanceEntry {
            uuid: uuid2.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let meta_instance3 = MetaInstanceEntry {
            uuid: uuid3.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_meta_instance(&uuid1);
        hard_delete_meta_instance(&uuid2);
        hard_delete_meta_instance(&uuid3);

        add_meta_instance(&meta_instance1).unwrap();
        add_meta_instance(&meta_instance2).unwrap();
        add_meta_instance(&meta_instance3).unwrap();

        let number = count_meta_instances(&context).unwrap();
        assert_eq!(number, 3);

        hard_delete_meta_instance(&uuid1);
        hard_delete_meta_instance(&uuid2);
        hard_delete_meta_instance(&uuid3);
    }

    #[test]
    #[serial]
    fn test_meta_instances_permissions() {
        let _ = init_meta_instance_table();
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();
        let uuid3 = Uuid::new_v4();
        let name = "test-instance".to_string();
        let sakura_host_uuid1 = Uuid::new_v4();
        let proxy_uuid1 = Uuid::new_v4();

        let meta_instance1 = MetaInstanceEntry {
            uuid: uuid1.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let meta_instance2 = MetaInstanceEntry {
            uuid: uuid2.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: "test-user-43".to_string(),
            project_id: "test_permissions_1".to_string(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let meta_instance3 = MetaInstanceEntry {
            uuid: uuid3.to_string(),
            name: name.clone(),
            sakura_host_uuid: sakura_host_uuid1.to_string(),
            proxy_uuid: proxy_uuid1.to_string(),
            owner_id: "test-user-44".to_string(),
            project_id: "test_permissions_2".to_string(),
            status: "ACTIVE".to_string(),
            created_at: "2025-03-31".to_string(),
            created_by: "admin".to_string(),
            updated_at: "2025-03-31".to_string(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_meta_instance(&uuid1);
        hard_delete_meta_instance(&uuid2);
        hard_delete_meta_instance(&uuid3);

        add_meta_instance(&meta_instance1).unwrap();
        add_meta_instance(&meta_instance2).unwrap();
        add_meta_instance(&meta_instance3).unwrap();

        // list-test normal user
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };
        let meta_instances = list_meta_instances(&context).unwrap();
        assert_eq!(meta_instances.len(), 1);

        // list-test project-admin
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: true.to_string(),
        };
        let meta_instances = list_meta_instances(&context).unwrap();
        assert_eq!(meta_instances.len(), 2);

        // list-test admin
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: true.to_string(),
            is_project_admin: false.to_string(),
        };
        let meta_instances = list_meta_instances(&context).unwrap();
        assert_eq!(meta_instances.len(), 3);

        // get-test normal user
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };
        match get_meta_instance(&uuid1, &context) {
            Ok(retrieved_meta_instance) => {
                assert_eq!(retrieved_meta_instance.uuid, uuid1.to_string());
            }
            Err(_) => {
                assert_eq!(true, false);
            }
        };

        // get-test normal user false uuid
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };
        if get_meta_instance(&uuid3, &context).is_ok() {
            assert_eq!(true, false);
        };

        // delete-test normal user false uuid
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };
        if delete_meta_instance(&uuid3, &context).is_ok() {
            assert_eq!(true, false);
        };

        hard_delete_meta_instance(&uuid1);
        hard_delete_meta_instance(&uuid2);
        hard_delete_meta_instance(&uuid3);
    }
}
