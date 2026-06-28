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

use super::hexagon_block::*;
use super::*;

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
/// * `input_links` - Vector of input links to other blocks.
#[derive(Debug, Serialize)]
pub struct InputBlock {
    pub uuid: Uuid,
    pub model_uuid: Uuid,
    #[serde(skip)]
    pub parent_block: Arc<Mutex<HexagonBlock>>,

    pub is_processed: bool,

    pub block_io: BlockIoBuffer,

    #[serde(with = "BigArray")]
    pub input_values: [f32; 128],
    pub input_links: Vec<InputSynapse>,
}

impl PartialEq for InputBlock {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.model_uuid == other.model_uuid
            && self.block_io == other.block_io
            && self.input_values == other.input_values
            && self.input_links == other.input_links
    }
}

impl InputBlock {
    /// Creates a new InputBlock instance.
    ///
    /// # Arguments
    ///
    /// * `hexagon_uuid` - Identifier for the hexagon this block belongs to.
    /// * `model_uuid` - Identifier for the model this block belongs to.
    ///
    /// # Returns
    ///
    /// A new InputBlock instance.
    pub fn new(model_uuid: &Uuid, parent_block: Arc<Mutex<HexagonBlock>>) -> Self {
        InputBlock {
            uuid: Uuid::new_v4(),
            model_uuid: *model_uuid,
            parent_block: parent_block,

            is_processed: false,

            block_io: BlockIoBuffer::new(1),

            input_values: [0.0f32; 128],
            input_links: Vec::new(),
        }
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

impl Block for InputBlock {
    fn process(&mut self) -> Result<bool, AinariError> {
        if !self.is_processed {}

        self.is_processed = true;

        let is_finished = self.block_io.send_forward()?;
        if is_finished {
            self.is_processed = false;
        }
        Ok(is_finished)
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
        ObjectType::InputBlock
    }

    fn get_parent_block(&self) -> Option<Arc<Mutex<HexagonBlock>>> {
        Some(self.parent_block.clone())
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
    fn test_serialize_deserialize() {
        // let original = InputBlock::default();

        // let cfg = bincode::config::standard();
        // let serialized: Vec<u8> =
        //     bincode::serde::encode_to_vec(&original, cfg).expect("Failed to serialize");
        // let deserialized: InputBlock = bincode::serde::decode_from_slice(&serialized, cfg)
        //     .expect("Failed to deserialize")
        //     .0;
        // println!("size: {}", serialized.len());

        // assert_eq!(original, deserialized);
    }
}
