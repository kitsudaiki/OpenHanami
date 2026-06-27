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
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use super::super::processing::worker_queue::*;
use super::axons::*;
use super::block_io::*;
use super::block_trait::*;

use crate::core::processing::worker_queue::WorkerTask;

use ainari_common::constants::*;
use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_common::functions::*;

// ==================================================================================================

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct StartEndBlock {
    pub uuid: Uuid,
    pub hexagon_uuid: Uuid,
    pub model_uuid: Uuid,

    pub block_io: BlockIoBuffer,
}

impl PartialEq for StartEndBlock {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.hexagon_uuid == other.hexagon_uuid
            && self.model_uuid == other.model_uuid
            && self.block_io == other.block_io
    }
}

impl StartEndBlock {
    pub fn new(hexagon_uuid: &Uuid, model_uuid: &Uuid) -> Self {
        let mut block = StartEndBlock {
            uuid: Uuid::new_v4(),
            hexagon_uuid: *hexagon_uuid,
            model_uuid: *model_uuid,

            block_io: BlockIoBuffer::default(),
        };

        block.block_io.output_buffer.push(AxonSection::default());

        block
    }
}

impl Block for StartEndBlock {
    fn process(&mut self, task_type: WorkerTaskType, cycle_number: u64) -> Result<(), AinariError> {
        send_forward(
            &mut self.block_io,
            task_type,
            cycle_number,
            &self.model_uuid,
            &self.hexagon_uuid,
            &self.uuid,
        );

        // TODO: trigger next cycle

        Ok(())
    }

    fn get_free_input(&mut self, axon_section: &mut AxonSection) -> bool {
        // self.block_io.input_buffer.append(axon_section.clone());
        // let pos = (self.block_io.input_buffer.len()-1) as u16;

        // axon_section.target_block_uuid = self.uuid;
        // axon_section.target_hexagon_uuid = self.hexagon_uuid;
        // axon_section.target_pos = pos;

        // self.block_io.inputs_in_use = self.block_io.input_buffer.len() as u64;

        return true;
    }

    /// Gets the UUID of the block.
    ///
    /// # Returns
    ///
    /// The UUID of the block.
    fn get_uuid(&self) -> Uuid {
        self.uuid
    }

    /// Gets the hexagon UUID of the block.
    ///
    /// # Returns
    ///
    /// The hexagon UUID of the block.
    fn get_hexagon_uud(&self) -> Uuid {
        self.hexagon_uuid
    }

    /// Gets the model UUID of the block.
    ///
    /// # Returns
    ///
    /// The model UUID of the block.
    fn get_model_uud(&self) -> Uuid {
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
        ObjectType::StartEndBlock
    }

    /// Sets the model UUID of the block.
    ///
    /// # Arguments
    ///
    /// * `new_model_uuid` - The new model UUID.
    fn set_model_uuid(&mut self, new_model_uuid: &Uuid) {
        self.model_uuid = *new_model_uuid;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_transfer() {}

    #[test]
    fn test_serialize_deserialize() {
        let original = StartEndBlock::default();

        let cfg = bincode::config::standard();
        let serialized: Vec<u8> =
            bincode::serde::encode_to_vec(&original, cfg).expect("Failed to serialize");
        let deserialized: StartEndBlock = bincode::serde::decode_from_slice(&serialized, cfg)
            .expect("Failed to deserialize")
            .0;
        println!("size: {}", serialized.len());

        assert_eq!(original, deserialized);
    }
}
