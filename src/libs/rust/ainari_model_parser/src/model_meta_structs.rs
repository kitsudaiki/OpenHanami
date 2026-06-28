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

use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_common::objects::*;

/// Configuration settings for the neural network model
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub neuron_cooldown: f32,
    pub refractory_time: u32,
    pub max_connection_distance: u32,
}

/// Metadata for connections between neurons (axons)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxonMeta {
    pub from: Position,
    pub to: Position,
}

/// Metadata for hexagonal neurons in the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HexagonMeta {
    pub uuid: Uuid,
    pub position: Position,
    pub name: String,

    pub is_input: bool,
    pub is_output: bool,

    pub axon_target: Uuid,

    pub neighbors: [Uuid; 12],
}

impl HexagonMeta {
    pub fn new(position: Position) -> Self {
        let new_uuid = Uuid::new_v4();
        HexagonMeta {
            uuid: new_uuid,
            position,
            name: "".to_string(),

            axon_target: new_uuid, // Default to its own UUID

            is_input: false,
            is_output: false,

            neighbors: [Uuid::nil(); 12],
        }
    }

    pub fn get_neighbor_position(&self, side: usize) -> Position {
        return get_neighbor_pos(&self.position, side);
    }

    pub fn get_neighbor_uuid(&self, side: usize) -> Uuid {
        return self.neighbors[side].clone();
    }
}

/// Calculates the position of a neighboring cell in a hexagonal grid.
///
/// Given a source position and a side number (0-11), returns the position of the adjacent cell.
/// The side numbering follows a specific pattern used in hexagonal grid algorithms.
///
/// # Arguments
///
/// * `source_pos` - The position of the source cell
/// * `side` - The side number (0-11) indicating which neighbor to get
///
/// # Panics
///
/// Panics if the side value is out of the valid range (0-11).
pub fn get_neighbor_pos(source_pos: &Position, side: usize) -> Position {
    let mut result = Position { x: 0, y: 0, z: 0 };

    match side {
        0 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z - 1;
        }
        1 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z - 1;
        }
        2 => {
            result.x = source_pos.x;
            result.y = source_pos.y;
            result.z = source_pos.z - 1;
        }
        3 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z;
        }
        4 => {
            result.x = source_pos.x + 1;
            result.y = source_pos.y;
            result.z = source_pos.z;
        }
        5 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z;
        }
        6 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z;
        }
        7 => {
            result.x = source_pos.x - 1;
            result.y = source_pos.y;
            result.z = source_pos.z;
        }
        8 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z;
        }
        9 => {
            result.x = source_pos.x;
            result.y = source_pos.y;
            result.z = source_pos.z + 1;
        }
        10 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z + 1;
        }
        11 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z + 1;
        }
        _ => panic!("Invalid side value: {side}"),
    }

    result
}

/// Gets the next five side numbers in a hexagonal grid traversal pattern.
///
/// Given a side number (0-11), returns an array of five side numbers that follow
/// a specific pattern used in hexagonal grid algorithms.
///
/// # Arguments
///
/// * `side` - The starting side number (0-11)
///
/// # Panics
///
/// Panics if the side value is out of the valid range (0-11).
pub fn get_next_sides(side: usize) -> [u8; 5] {
    match side {
        0 => [1, 4, 11, 5, 2],
        1 => [2, 8, 10, 7, 0],
        2 => [0, 6, 9, 3, 1],
        3 => [5, 2, 8, 10, 7],
        4 => [8, 10, 7, 0, 6],
        5 => [7, 0, 6, 9, 3],
        6 => [4, 11, 5, 2, 8],
        7 => [3, 1, 4, 11, 5],
        8 => [6, 9, 3, 1, 4],
        9 => [11, 5, 2, 8, 10],
        10 => [9, 3, 1, 4, 11],
        11 => [10, 7, 0, 6, 9],
        _ => panic!("Invalid side value: {side}; This should never happen!"),
    }
}

/// Metadata for input connections to the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMeta {
    pub uuid: Uuid,
    pub hexagon_uuid: Uuid,
    pub name: String,
    pub position: Position,
}

impl InputMeta {
    /// Creates a new InputMeta instance
    ///
    /// # Arguments
    /// * `name` - The name of the input
    /// * `position` - The position of the input in the network
    ///
    /// # Returns
    /// A new InputMeta instance with a generated UUID and nil hexagon UUID
    pub fn new(name: String, position: Position) -> Self {
        InputMeta {
            uuid: Uuid::new_v4(),
            hexagon_uuid: Uuid::nil(), // Not connected to a hexagon by default
            name,
            position,
        }
    }
}

/// Metadata for output connections from the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMeta {
    pub uuid: Uuid,
    pub hexagon_uuid: Uuid,
    pub name: String,
    pub position: Position,
    pub output_type: OutputType,
}

impl OutputMeta {
    /// Creates a new OutputMeta instance
    ///
    /// # Arguments
    /// * `name` - The name of the output
    /// * `position` - The position of the output in the network
    /// * `output_type` - The type of data this output produces
    ///
    /// # Returns
    /// A new OutputMeta instance with a generated UUID and nil hexagon UUID
    pub fn new(name: String, position: Position, output_type: OutputType) -> Self {
        OutputMeta {
            uuid: Uuid::new_v4(),
            hexagon_uuid: Uuid::nil(), // Not connected to a hexagon by default
            name,
            position,
            output_type,
        }
    }
}

/// Metadata for the entire neural network model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMeta {
    pub uuid: Uuid,
    pub name: String,
    pub version: i32,
    pub settings: Settings,
    pub hexagons: HashMap<Uuid, HexagonMeta>,
    pub axons: Vec<AxonMeta>,
    pub inputs: Vec<InputMeta>,
    pub outputs: Vec<OutputMeta>,
}

impl ModelMeta {
    pub fn get_hexagon_meta(&self, hexagon_uuid: &Uuid) -> Result<&HexagonMeta, AinariError> {
        if let Some(hexagon_meta) = self.hexagons.get(hexagon_uuid) {
            return Ok(&hexagon_meta);
        } else {
            let msg = format!("Hexagon with uuid '{hexagon_uuid}' not found.");
            return Err(AinariError::InvalidInput(msg));
        }
    }
}
