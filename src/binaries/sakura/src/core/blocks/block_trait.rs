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

use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use ainari_common::enums::*;
use ainari_common::error::AinariError;

use crate::core::processing::worker_queue::WorkerTaskType;

use super::axons::*;
use super::block_io::*;

pub trait Block: Send + Sync + Debug {
    fn process(&mut self, task_type: WorkerTaskType, cycle_number: u64) -> Result<(), AinariError>;

    fn get_free_input(&mut self, axon_section: &mut AxonSection) -> bool;
    fn get_uuid(&self) -> Uuid;
    fn get_hexagon_uud(&self) -> Uuid;
    fn get_model_uud(&self) -> Uuid;
    #[allow(dead_code)]
    fn set_model_uuid(&mut self, new_model_uuid: &Uuid);
    #[allow(dead_code)]
    fn get_type(&self) -> ObjectType;
    #[allow(dead_code)]
    fn serailize(&self) -> Vec<u8>;

    fn get_block_io(&mut self) -> &mut BlockIoBuffer;
}
