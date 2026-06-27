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
use serde_big_array::BigArray;
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use super::super::processing::worker_queue::*;
use super::axons::*;
use super::block_io::*;
use super::block_trait::*;

use ainari_common::constants::*;
use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_common::functions::*;

// ==================================================================================================

#[derive(Default, PartialEq, Debug, Serialize, Deserialize)]
pub struct InputSynapse {
    pub upper_next: u32,
    pub lower_next: u32,
    pub border: f32,
    pub target: u16,
    pub level: u8,
    pub power: u8,
}

// check that InputSynapse takes really only 8 byte of memory
const _: () = {
    assert!(size_of::<InputSynapse>() == 16);
};

/// Represents an input block in the neural network model.
/// This block is responsible for receiving and processing input data.
///
/// # Fields
///
/// * `uuid` - Unique identifier for the block.
/// * `hexagon_uuid` - Identifier for the hexagon this block belongs to.
/// * `model_uuid` - Identifier for the model this block belongs to.
/// * `block_io` - Input/output buffer for the block.
/// * `name` - Name of the block.
/// * `input_links` - Vector of input links to other blocks.
/// * `local_finish_counter` - Local counter for tracking completion status.
/// * `finish_counter_mutex` - Shared counter for tracking completion status across threads.
#[derive(Debug, Serialize, Deserialize)]
pub struct InputBlock {
    pub uuid: Uuid,
    pub hexagon_uuid: Uuid,
    pub model_uuid: Uuid,

    pub block_io: BlockIoBuffer,

    pub name: String,

    #[serde(with = "BigArray")]
    pub input_values: [f32; 128],
    pub input_links: Vec<InputSynapse>,
}

impl PartialEq for InputBlock {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.hexagon_uuid == other.hexagon_uuid
            && self.model_uuid == other.model_uuid
            && self.block_io == other.block_io
            && self.name == other.name
            && self.input_values == other.input_values
            && self.input_links == other.input_links
    }
}

impl InputBlock {
    /// Creates a new InputBlock instance.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the block.
    /// * `hexagon_uuid` - Identifier for the hexagon this block belongs to.
    /// * `model_uuid` - Identifier for the model this block belongs to.
    /// * `finish_counter` - Shared counter for tracking completion status across threads.
    ///
    /// # Returns
    ///
    /// A new InputBlock instance.
    pub fn new(name: &str, hexagon_uuid: &Uuid, model_uuid: &Uuid) -> Self {
        let mut block = InputBlock {
            uuid: Uuid::new_v4(),
            hexagon_uuid: *hexagon_uuid,
            model_uuid: *model_uuid,

            name: name.to_owned(),

            block_io: BlockIoBuffer::default(),

            input_values: [0.0f32; 128],
            input_links: Vec::new(),
        };

        block.block_io.output_buffer.push(AxonSection::default());

        block
    }

    // ==================================================================================================

    /// Applies input data to the input block.
    ///
    /// # Arguments
    ///
    /// * `input_ptr` - Pointer to the input data.
    pub fn apply_input(&mut self, input_ptr: &[f32], offset: usize) {
        for (i, val) in input_ptr.iter().enumerate().skip(offset).take(128) {
            self.input_values[i - offset] = *val;
        }
    }
}

impl Default for InputBlock {
    fn default() -> Self {
        let mut block = InputBlock {
            uuid: Uuid::new_v4(),
            hexagon_uuid: Uuid::new_v4(),
            model_uuid: Uuid::new_v4(),

            name: "".to_owned(),

            block_io: BlockIoBuffer::default(),

            input_values: [0.0f32; 128],
            input_links: Vec::new(),
        };

        block.block_io.output_buffer.push(AxonSection::default());

        block
    }
}

impl Block for InputBlock {
    fn process(&mut self, task_type: WorkerTaskType, cycle_number: u64) -> Result<(), AinariError> {
        send_forward(
            &mut self.block_io,
            task_type,
            cycle_number,
            &self.model_uuid,
            &self.hexagon_uuid,
            &self.uuid,
        );
        Ok(())
    }

    /// Gets a free input axon section.
    ///
    /// # Arguments
    ///
    /// * `_` - Unused parameter (reserved for future use).
    ///
    /// # Returns
    ///
    /// `true` if a free input was found, `false` otherwise.
    fn get_free_input(&mut self, _: &mut AxonSection) -> bool {
        false
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
        ObjectType::InputBlock
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
    fn test_apply_input() {
        // let name = "test-input".to_string();
        // let hexagon_uuid = Uuid::new_v4();
        // let model_uuid = Uuid::new_v4();
        // let mut input_block = InputBlock::new(&name, &hexagon_uuid, &model_uuid, 0);

        // let input_values = vec![1.0, 2.0, -3.0, 4.0];
        // input_block.apply_input(&input_values);

        // // check size of the resized buffers
        // assert_eq!(input_block.input_links.len(), 16);
        // assert_eq!(input_block.block_io.output_buffer.len(), 1);

        // // check input-links
        // assert_eq!(input_block.input_links[4], 0);
        // assert_eq!(input_block.input_links[5], UNINIT_STATE_64);
        // assert_eq!(input_block.input_links[6], 1);
        // assert_eq!(input_block.input_links[7], UNINIT_STATE_64);
        // assert_eq!(input_block.input_links[8], UNINIT_STATE_64);
        // assert_eq!(input_block.input_links[9], 2);
        // assert_eq!(input_block.input_links[10], 3);
        // assert_eq!(input_block.input_links[11], UNINIT_STATE_64);

        // // check axons
        // assert_eq!(
        //     input_block.block_io.output_buffer[0].data.axons[0].potential,
        //     1.0
        // );
        // assert_eq!(
        //     input_block.block_io.output_buffer[0].data.axons[1].potential,
        //     2.0
        // );
        // assert_eq!(
        //     input_block.block_io.output_buffer[0].data.axons[2].potential,
        //     3.0
        // );
        // assert_eq!(
        //     input_block.block_io.output_buffer[0].data.axons[3].potential,
        //     4.0
        // );
    }

    #[test]
    fn test_serialize_deserialize() {
        let original = InputBlock::default();

        let cfg = bincode::config::standard();
        let serialized: Vec<u8> =
            bincode::serde::encode_to_vec(&original, cfg).expect("Failed to serialize");
        let deserialized: InputBlock = bincode::serde::decode_from_slice(&serialized, cfg)
            .expect("Failed to deserialize")
            .0;
        println!("size: {}", serialized.len());

        assert_eq!(original, deserialized);
    }
}
