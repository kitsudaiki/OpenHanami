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

use diesel::backend::Backend;
use diesel::connection::SimpleConnection;
use diesel::deserialize::{self, FromSql, FromSqlRow};
use diesel::expression::AsExpression;
use diesel::prelude::*;
use diesel::serialize::{self, Output, ToSql};
use diesel::sql_types::Varchar;
use diesel::sqlite::Sqlite;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

use super::constants::UNINIT_POINT_32;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Position {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Position {
    pub fn new() -> Self {
        Position {
            x: UNINIT_POINT_32,
            y: UNINIT_POINT_32,
            z: UNINIT_POINT_32,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.x != UNINIT_POINT_32 && self.y != UNINIT_POINT_32 && self.z != UNINIT_POINT_32
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[ {} , {} , {} ]", self.x, self.y, self.z)
    }
}

// Store a String here instead of a Uuid!
#[derive(Debug, Clone, PartialEq, AsExpression, FromSqlRow)]
#[diesel(sql_type = Varchar)]
pub struct DbUuid(String);

// ToSql now safely borrows the owned String
impl<DB: Backend> ToSql<Varchar, DB> for DbUuid
where
    String: ToSql<Varchar, DB>,
{
    fn to_sql<'b>(&'b self, out: &mut Output<'b, '_, DB>) -> serialize::Result {
        // self.0 is a String. We borrow it, satisfying the 'b lifetime!
        self.0.to_sql(out)
    }
}

// FromSql continues to read a String
impl<DB: Backend> FromSql<Varchar, DB> for DbUuid
where
    String: FromSql<Varchar, DB>,
{
    fn from_sql(bytes: DB::RawValue<'_>) -> deserialize::Result<Self> {
        let s = String::from_sql(bytes)?;
        Ok(DbUuid(s))
    }
}

// Convert Uuid -> DbUuid (Happens BEFORE ToSql)
impl From<Uuid> for DbUuid {
    fn from(uuid: Uuid) -> Self {
        // We allocate the String here, so it is owned by DbUuid
        DbUuid(uuid.to_string())
    }
}

// Convert DbUuid -> Uuid (Happens AFTER FromSql)
impl TryFrom<DbUuid> for Uuid {
    type Error = uuid::Error;
    fn try_from(db_uuid: DbUuid) -> Result<Self, Self::Error> {
        Uuid::parse_str(&db_uuid.0)
    }
}
