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
use diesel::prelude::*;
use diesel::result::DatabaseErrorKind;
use uuid::Uuid;

use crate::database::db_handle;

use ainari_api_structs::user_context::UserContext;
use ainari_common::enums;

// Define the schema for the instances table
table! {
    instances (uuid) {
        uuid -> Varchar,
        name -> Varchar,
        is_created -> Bool,
        number_of_cores -> BigInt,
        size_of_memory -> BigInt,
        size_of_disk -> BigInt,
        image_uuid -> Varchar,
        seed_uuid -> Varchar,
        public_key_uuid -> Varchar,
        ip_addresses -> Varchar,
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

/// Represents a single entry in the instances table
#[derive(Insertable, Queryable, Selectable, Debug, PartialEq, Clone)]
#[diesel(table_name = instances)]
pub struct InstanceEntry {
    pub uuid: String,
    pub name: String,
    pub is_created: bool,
    pub number_of_cores: i64,
    pub size_of_memory: i64,
    pub size_of_disk: i64,
    pub image_uuid: String,
    pub seed_uuid: String,
    pub public_key_uuid: String,
    pub ip_addresses: String,
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

/// Initializes the instances table in the database if it doesn't already exist
///
/// # Returns
/// * `Ok(())` if the table was created or already exists
/// * An error if there was a problem creating the table
pub fn init_instance_table() -> Result<(), Box<dyn std::error::Error>> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    conn.batch_execute(
        "CREATE TABLE IF NOT EXISTS instances (
        uuid VARCHAR(40) PRIMARY KEY,
        name VARCHAR(256),
        is_created -> BOOL
        number_of_cores -> INTEGER
        size_of_memory -> INTEGER
        size_of_disk -> INTEGER
        image_uuid -> VARCHAR(40)
        seed_uuid -> VARCHAR(40)
        public_key_uuid -> VARCHAR(40)
        ip_addresses -> TEXT
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

/// Adds a new instance to the database with the provided information
///
/// # Arguments
/// * `instance_uuid` - Unique identifier for the new instance
/// * `instance_name` - Name for the new instance
/// * `instance_template` - Template content for the new instance
/// * `inputs` - Vector of input specifications
/// * `outputs` - Vector of output specifications
/// * `context` - User context containing authentication information
///
/// # Returns
/// * `Ok(usize)` with the number of rows inserted on success
/// * `Err` with an appropriate error on failure
pub fn add_new_instance(
    instance_uuid: &Uuid,
    instance_name: &str,
    number_of_cores: i64,
    size_of_memory: i64,
    size_of_disk: i64,
    image_uuid: &Uuid,
    seed_uuid: &Uuid,
    public_key_uuid: &Uuid,
    ip_addresses: &Vec<String>,
    context: &UserContext,
) -> QueryResult<usize> {
    // Serialize the input and output vectors to JSON strings
    let ip_addresses_str = match serde_json::to_string(&ip_addresses) {
        Ok(ip_addresses_str) => ip_addresses_str,
        Err(e) => {
            return Err(diesel::result::Error::DatabaseError(
                DatabaseErrorKind::SerializationFailure,
                Box::new(format!("Failed to serialize ip_addresses with error: {e}")),
            ));
        }
    };

    // Create the new instance entry
    let instance = InstanceEntry {
        uuid: instance_uuid.to_string().clone(),
        name: instance_name.to_owned(),
        is_created: false,
        number_of_cores: number_of_cores,
        size_of_memory: size_of_memory,
        size_of_disk: size_of_disk,
        image_uuid: image_uuid.to_string().clone(),
        seed_uuid: seed_uuid.to_string().clone(),
        public_key_uuid: public_key_uuid.to_string().clone(),
        ip_addresses: ip_addresses_str,
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

    // Insert the instance into the database
    add_instance(&instance)
}

/// Adds a instance entry to the database
///
/// # Arguments
/// * `instance` - The instance entry to insert
///
/// # Returns
/// * `Ok(usize)` with the number of rows inserted on success
/// * `Err` with an appropriate error on failure
pub fn add_instance(instance: &InstanceEntry) -> QueryResult<usize> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::instances::dsl::*;
    diesel::insert_into(instances)
        .values(instance)
        .execute(&mut *conn)
}

/// Retrieves a specific instance from the database
///
/// # Arguments
/// * `instance_uuid` - Unique identifier of the instance to retrieve
/// * `context` - User context containing authentication information
///
/// # Returns
/// * `Ok(InstanceEntry)` with the instance on success
/// * `Err(enums::DbError)` with an appropriate error on failure
pub fn get_instance(instance_uuid: &Uuid, context: &UserContext) -> Result<InstanceEntry, enums::DbError> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::instances::dsl::*;

    // Build the query with appropriate filters based on user permissions
    let mut query = instances
        .filter(uuid.eq(instance_uuid.to_string()).and(status.eq("ACTIVE")))
        .into_boxed();

    // Apply project and ownership filters for non-admin users
    if context.is_admin != true.to_string() {
        query = query.filter(project_id.eq(context.project_id.clone()));
        if context.is_project_admin != true.to_string() {
            query = query.filter(owner_id.eq(context.user_id.clone()));
        }
    }

    // Execute the query and return the result
    match query
        .select(InstanceEntry::as_select())
        .first::<InstanceEntry>(&mut *conn)
    {
        Ok(instance) => Ok(instance),
        Err(diesel::result::Error::NotFound) => Err(enums::DbError::NotFound),
        Err(e) => {
            log::error!("Database-error: {e:?}");
            Err(enums::DbError::InternalError)
        }
    }
}

/// Lists all deleted instances from the database
///
/// # Returns
/// * `Ok(Vec<InstanceEntry>)` with the list of deleted instances on success
/// * `Err` with an appropriate error on failure
pub fn list_deleted_instances() -> QueryResult<Vec<InstanceEntry>> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::instances::dsl::*;

    // Create a query to find all instances with "DELETED" status
    let query = instances.filter(status.eq("DELETED")).into_boxed();

    // Execute the query and return the results
    query.select(InstanceEntry::as_select()).load(&mut *conn)
}

/// Lists all active instances from the database, applying appropriate filters based on user permissions
///
/// # Arguments
/// * `context` - User context containing authentication information
///
/// # Returns
/// * `Ok(Vec<InstanceEntry>)` with the list of instances on success
/// * `Err` with an appropriate error on failure
pub fn list_instances(context: &UserContext) -> QueryResult<Vec<InstanceEntry>> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::instances::dsl::*;

    // Build the query with appropriate filters based on user permissions
    let mut query = instances.filter(status.eq("ACTIVE")).into_boxed();

    // Apply project and ownership filters for non-admin users
    if context.is_admin != true.to_string() {
        query = query.filter(project_id.eq(context.project_id.clone()));
        if context.is_project_admin != true.to_string() {
            query = query.filter(owner_id.eq(context.user_id.clone()));
        }
    }

    // Execute the query and return the results
    query.select(InstanceEntry::as_select()).load(&mut *conn)
}

/// Marks a specific instance as deleted in the database
///
/// # Arguments
/// * `instance_uuid` - Unique identifier of the instance to delete
/// * `context` - User context containing authentication information
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(enums::DbError)` with an appropriate error on failure
pub fn delete_instance(instance_uuid: &Uuid, context: &UserContext) -> Result<(), enums::DbError> {
    // First verify that the instance exists and the user has permission to delete it
    get_instance(instance_uuid, context)?;

    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::instances::dsl::*;

    // Update the instance's status to "DELETED" and set the deletion timestamp and user
    match diesel::update(instances.filter(uuid.eq(instance_uuid.to_string())))
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

/// Marks all active instances as deleted in the database
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(enums::DbError)` with an appropriate error on failure
pub fn delete_all_instance() -> Result<(), enums::DbError> {
    let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
    use self::instances::dsl::*;

    // Update all active instances to have "DELETED" status with a system user as the deleter
    match diesel::update(instances.filter(status.eq("ACTIVE")))
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

    fn hard_delete_instance(instance_uuid: &Uuid) {
        use self::instances::dsl::*;
        let mut conn = db_handle::DB_CONN.lock().expect("mutex poisoned");
        let _ = diesel::delete(instances.filter(uuid.eq(instance_uuid.to_string()))).execute(&mut *conn);
    }

    #[test]
    #[serial]
    fn test_add_get_instance() {
        let _ = init_instance_table();
        let uuid1 = Uuid::new_v4();

        let project_id = "test-project".to_string();
        let owner_id = "test-user".to_string();
        let context = UserContext {
            token: "".to_string(),
            user_id: owner_id.clone(),
            project_id: project_id.clone(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };

        let instance = InstanceEntry {
            uuid: uuid1.to_string(),
            name: "Alice".to_string(),
            is_created: false,
            number_of_cores: 2,
            size_of_memory: 4096,
            size_of_disk: 1024,
            image_uuid: Uuid::new_v4().to_string(),
            seed_uuid: Uuid::new_v4().to_string(),
            public_key_uuid: Uuid::new_v4().to_string(),
            ip_addresses: serde_json::to_string(&vec!["192.168.1.1".to_string()]).unwrap(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: Utc::now().to_rfc3339(),
            created_by: "admin".to_string(),
            updated_at: Utc::now().to_rfc3339(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_instance(&uuid1);

        add_instance(&instance).unwrap();
        match get_instance(&uuid1, &context) {
            Ok(retrieved_instance) => {
                assert_eq!(retrieved_instance.uuid, instance.uuid);
                assert_eq!(retrieved_instance.name, instance.name);
                assert_eq!(retrieved_instance.is_created, instance.is_created);
                assert_eq!(retrieved_instance.number_of_cores, instance.number_of_cores);
                assert_eq!(retrieved_instance.size_of_memory, instance.size_of_memory);
                assert_eq!(retrieved_instance.size_of_disk, instance.size_of_disk);
                assert_eq!(retrieved_instance.image_uuid, instance.image_uuid);
                assert_eq!(retrieved_instance.seed_uuid, instance.seed_uuid);
                assert_eq!(retrieved_instance.public_key_uuid, instance.public_key_uuid);
                assert_eq!(retrieved_instance.ip_addresses, instance.ip_addresses);
                assert_eq!(retrieved_instance.owner_id, instance.owner_id);
                assert_eq!(retrieved_instance.project_id, instance.project_id);
                assert_eq!(retrieved_instance.status, instance.status);
                assert_eq!(retrieved_instance.created_by, instance.created_by);
                assert_eq!(retrieved_instance.updated_by, instance.updated_by);
                assert_eq!(retrieved_instance.deleted_at, instance.deleted_at);
                assert_eq!(retrieved_instance.deleted_by, instance.deleted_by);
            }
            Err(_) => {
                assert_eq!(true, false);
            }
        };

        hard_delete_instance(&uuid1);
    }

    #[test]
    #[serial]
    fn test_list_instances() {
        let _ = init_instance_table();
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();

        let project_id = "test-project".to_string();
        let owner_id = "test-user".to_string();
        let context = UserContext {
            token: "".to_string(),
            user_id: owner_id.clone(),
            project_id: project_id.clone(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };

        let instance1 = InstanceEntry {
            uuid: uuid1.to_string(),
            name: "Alice".to_string(),
            is_created: false,
            number_of_cores: 2,
            size_of_memory: 4096,
            size_of_disk: 1024,
            image_uuid: Uuid::new_v4().to_string(),
            seed_uuid: Uuid::new_v4().to_string(),
            public_key_uuid: Uuid::new_v4().to_string(),
            ip_addresses: serde_json::to_string(&vec!["192.168.1.1".to_string()]).unwrap(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: Utc::now().to_rfc3339(),
            created_by: "admin".to_string(),
            updated_at: Utc::now().to_rfc3339(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let instance2 = InstanceEntry {
            uuid: uuid2.to_string(),
            name: "Bob".to_string(),
            is_created: false,
            number_of_cores: 2,
            size_of_memory: 4096,
            size_of_disk: 1024,
            image_uuid: Uuid::new_v4().to_string(),
            seed_uuid: Uuid::new_v4().to_string(),
            public_key_uuid: Uuid::new_v4().to_string(),
            ip_addresses: serde_json::to_string(&vec!["192.168.1.1".to_string()]).unwrap(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "DELETED".to_string(),
            created_at: Utc::now().to_rfc3339(),
            created_by: "admin".to_string(),
            updated_at: Utc::now().to_rfc3339(),
            updated_by: "admin".to_string(),
            deleted_at: Some(Utc::now().to_rfc3339()),
            deleted_by: Some("admin".to_string()),
        };

        hard_delete_instance(&uuid1);
        hard_delete_instance(&uuid2);

        add_instance(&instance1).unwrap();
        add_instance(&instance2).unwrap();
        let instances = list_instances(&context).unwrap();
        assert_eq!(instances.len(), 1);
        hard_delete_instance(&uuid1);
        hard_delete_instance(&uuid2);
    }

    #[test]
    #[serial]
    fn test_delete_instance() {
        let _ = init_instance_table();
        let uuid1 = Uuid::new_v4();

        let project_id = "test-project".to_string();
        let owner_id = "test-user".to_string();
        let context = UserContext {
            token: "".to_string(),
            user_id: owner_id.clone(),
            project_id: project_id.clone(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };

        let instance = InstanceEntry {
            uuid: uuid1.to_string(),
            name: "Alice".to_string(),
            is_created: false,
            number_of_cores: 2,
            size_of_memory: 4096,
            size_of_disk: 1024,
            image_uuid: Uuid::new_v4().to_string(),
            seed_uuid: Uuid::new_v4().to_string(),
            public_key_uuid: Uuid::new_v4().to_string(),
            ip_addresses: serde_json::to_string(&vec!["192.168.1.1".to_string()]).unwrap(),
            owner_id: owner_id.clone(),
            project_id: project_id.clone(),
            status: "ACTIVE".to_string(),
            created_at: Utc::now().to_rfc3339(),
            created_by: "admin".to_string(),
            updated_at: Utc::now().to_rfc3339(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_instance(&uuid1);

        add_instance(&instance).unwrap();
        let _ = delete_instance(&uuid1, &context);
        let result = get_instance(&uuid1, &context);
        assert!(result.is_err());
    }


    #[test]
    #[serial]
    fn test_instances_permissions() {
        let _ = init_instance_table();
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();
        let uuid3 = Uuid::new_v4();

        let instance1 = InstanceEntry {
            uuid: uuid1.to_string(),
            name: "Alice".to_string(),
            is_created: false,
            number_of_cores: 1,
            size_of_memory: 1024,
            size_of_disk: 20480,
            image_uuid: Uuid::new_v4().to_string(),
            seed_uuid: Uuid::new_v4().to_string(),
            public_key_uuid: Uuid::new_v4().to_string(),
            ip_addresses: "[]".to_string(),
            owner_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            status: "ACTIVE".to_string(),
            created_at: Utc::now().to_rfc3339(),
            created_by: "admin".to_string(),
            updated_at: Utc::now().to_rfc3339(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let instance2 = InstanceEntry {
            uuid: uuid2.to_string(),
            name: "Bob".to_string(),
            is_created: false,
            number_of_cores: 1,
            size_of_memory: 1024,
            size_of_disk: 20480,
            image_uuid: Uuid::new_v4().to_string(),
            seed_uuid: Uuid::new_v4().to_string(),
            public_key_uuid: Uuid::new_v4().to_string(),
            ip_addresses: "[]".to_string(),
            owner_id: "test-user-43".to_string(),
            project_id: "test_permissions_1".to_string(),
            status: "ACTIVE".to_string(),
            created_at: Utc::now().to_rfc3339(),
            created_by: "admin".to_string(),
            updated_at: Utc::now().to_rfc3339(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        let instance3 = InstanceEntry {
            uuid: uuid3.to_string(),
            name: "Poi".to_string(),
            is_created: false,
            number_of_cores: 1,
            size_of_memory: 1024,
            size_of_disk: 20480,
            image_uuid: Uuid::new_v4().to_string(),
            seed_uuid: Uuid::new_v4().to_string(),
            public_key_uuid: Uuid::new_v4().to_string(),
            ip_addresses: "[]".to_string(),
            owner_id: "test-user-44".to_string(),
            project_id: "test_permissions_2".to_string(),
            status: "ACTIVE".to_string(),
            created_at: Utc::now().to_rfc3339(),
            created_by: "admin".to_string(),
            updated_at: Utc::now().to_rfc3339(),
            updated_by: "admin".to_string(),
            deleted_at: None,
            deleted_by: None,
        };

        hard_delete_instance(&uuid1);
        hard_delete_instance(&uuid2);
        hard_delete_instance(&uuid3);

        add_instance(&instance1).unwrap();
        add_instance(&instance2).unwrap();
        add_instance(&instance3).unwrap();

        // list-test normal user
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };
        let instances = list_instances(&context).unwrap();
        assert_eq!(instances.len(), 1);

        // list-test project-admin
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: true.to_string(),
        };
        let instances = list_instances(&context).unwrap();
        assert_eq!(instances.len(), 2);

        // list-test admin
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: true.to_string(),
            is_project_admin: false.to_string(),
        };
        let instances = list_instances(&context).unwrap();
        assert_eq!(instances.len(), 3);

        // get-test normal user
        let context = UserContext {
            token: "".to_string(),
            user_id: "test-user-42".to_string(),
            project_id: "test_permissions_1".to_string(),
            is_admin: false.to_string(),
            is_project_admin: false.to_string(),
        };
        match get_instance(&uuid1, &context) {
            Ok(retrieved_instance) => {
                assert_eq!(retrieved_instance.uuid, uuid1.to_string());
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
        if get_instance(&uuid3, &context).is_ok() {
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
        if delete_instance(&uuid3, &context).is_ok() {
            assert_eq!(true, false);
        };

        hard_delete_instance(&uuid1);
        hard_delete_instance(&uuid2);
        hard_delete_instance(&uuid3);
    }
}
