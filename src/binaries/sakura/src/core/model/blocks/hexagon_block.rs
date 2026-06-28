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

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use super::*;

use ainari_common::enums::*;
use ainari_common::error::AinariError;

// ==================================================================================================

#[derive(Debug, Serialize)]
pub struct HexagonBlock {
    pub uuid: Uuid,
    pub model_uuid: Uuid,

    pub block_io: BlockIoBuffer,

    #[serde(skip)]
    pub parent_block: Option<Arc<Mutex<HexagonBlock>>>,
}

impl PartialEq for HexagonBlock {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.model_uuid == other.model_uuid
            && self.block_io == other.block_io
    }
}

impl HexagonBlock {
    pub fn new(model_uuid: &Uuid, parent_block: Option<Arc<Mutex<HexagonBlock>>>) -> Self {
        HexagonBlock {
            uuid: Uuid::new_v4(),
            model_uuid: *model_uuid,
            block_io: BlockIoBuffer::new(0),
            parent_block: parent_block,
        }
    }
}

impl Block for HexagonBlock {
    fn process(&mut self) -> Result<bool, AinariError> {
        Ok(true)
    }

    fn get_free_input(&mut self) -> u8 {
        255
    }

    /// Gets the UUID of the block.
    ///
    /// # Returns
    ///
    /// The UUID of the block.
    fn get_uuid(&self) -> Uuid {
        self.uuid
    }

    /// Gets the model UUID of the block.
    ///
    /// # Returns
    ///
    /// The model UUID of the block.
    fn get_model_uuid(&self) -> Uuid {
        self.model_uuid
    }

    /// Gets the block I/O buffer.
    ///
    /// # Returns
    ///
    /// A mutable reference to the block I/O buffer.
    fn get_block_io(&mut self) -> &mut BlockIoBuffer {
        &mut self.block_io
    }

    /// Gets the type of the block.
    ///
    /// # Returns
    ///
    /// The type of the block.
    fn get_type(&self) -> ObjectType {
        ObjectType::HexagonBlock
    }

    fn get_parent_block(&self) -> Option<Arc<Mutex<HexagonBlock>>> {
        self.parent_block.clone()
    }

    /// Serializes the block to a byte vector.
    ///
    /// # Returns
    ///
    /// A byte vector containing the serialized block.
    fn serailize(&self) -> Vec<u8> {
        let cfg = bincode::config::standard();
        bincode::serde::encode_to_vec(self, cfg).expect("Failed to serialize")
    }
}
