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

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;

use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_model_parser::model_meta_structs::*;

use crate::core::blocks::core_block::*;
use crate::core::blocks::input_block::*;
use crate::core::blocks::output_block::*;
use crate::core::model_content::*;
use crate::core::processing::output_buffer::OutputBuffer;

use super::blocks::block_trait::Block;

lazy_static::lazy_static! {
    /// Global singleton for model data handling.
    ///
    /// This provides thread-safe access to all models and their components.
    pub static ref MODEL_HANDLER: RwLock<ModelDataHandler> = RwLock::new(init_model_data_handler());
}

// ==================================================================================================

/// Main handler for managing multiple models and their components.
///
/// This struct provides functionality for creating, accessing, and manipulating models
/// and their associated blocks, inputs, and outputs.
pub struct ModelDataHandler {
    /// Map of model UUIDs to their corresponding ModelContent instances.
    pub models: HashMap<Uuid, Arc<Mutex<ModelContent>>>,
}

// ==================================================================================================

/// Initializes a new empty ModelDataHandler instance.
///
/// # Returns
/// A new ModelDataHandler with an empty models map.
pub fn init_model_data_handler() -> ModelDataHandler {
    ModelDataHandler {
        models: HashMap::new(),
    }
}

// ==================================================================================================

impl ModelDataHandler {
    /// Initializes a new model with the given metadata and UUID.
    ///
    /// This creates a complete model structure including all blocks, inputs, and outputs.
    ///
    /// # Arguments
    /// * `model_uuid` - UUID of the model to initialize.
    /// * `parsed_model` - Metadata describing the model's structure and configuration.
    ///
    /// # Returns
    /// * `Ok(())` on success.
    /// * `Err(AinariError)` if the model already exists or if initialization fails.
    pub fn init_new_model(
        &mut self,
        model_uuid: &Uuid,
        parsed_model: &ModelMeta,
    ) -> Result<(), AinariError> {
        if self.models.contains_key(&model_uuid) {
            let msg = format!("Model with uuid '{}' already exist.", model_uuid);
            return Err(AinariError::InvalidInput(msg));
        }

        let mut content = init_new_model(model_uuid, parsed_model.clone())?;
        self.models
            .insert(model_uuid.clone(), Arc::new(Mutex::new(content)));

        Ok(())
    }

    /// Gets an immutable reference to a model by its UUID.
    ///
    /// # Arguments
    /// * `model_uuid` - UUID of the model to retrieve.
    ///
    /// # Returns
    /// * `Ok(&ModelContent)` on success.
    /// * `Err(AinariError)` if the model doesn't exist.
    pub fn get_model(&self, model_uuid: &Uuid) -> Result<Arc<Mutex<ModelContent>>, AinariError> {
        if let Some(model) = self.models.get(model_uuid) {
            Ok(model.clone())
        } else {
            let msg = format!("Model with uuid '{model_uuid}' not found.");
            Err(AinariError::InvalidInput(msg))
        }
    }

    /// Retrieves a block from a model's hexagon.
    ///
    /// This method finds a specific block within a hexagon of a model using the provided UUIDs.
    ///
    /// # Arguments
    ///
    /// * `model_uuid` - A reference to the UUID of the model
    /// * `hexagon_uuid` - A reference to the UUID of the hexagon containing the block
    /// * `block_uuid` - A reference to the UUID of the block to retrieve
    ///
    /// # Returns
    ///
    /// * `Result<Arc<Mutex<dyn Block>>, AinariError>` - The block if found, Err otherwise
    pub fn get_block(
        &self,
        model_uuid: &Uuid,
        hexagon_uuid: &Uuid,
        block_uuid: &Uuid,
    ) -> Result<Arc<Mutex<dyn Block>>, AinariError> {
        // let model_link = self.get_model(model_uuid)?;

        // let binding = model_link.hexagon_data.read().expect("mutex poisoned");
        // let hexagon_link = if let Some(h) = binding.get(hexagon_uuid) {
        //     h.lock().expect("mutex poisoned")
        // } else {
        //     let msg = format!("Hexagon with uuid '{hexagon_uuid}' not found.");
        //     return Err(AinariError::InvalidInput(msg));
        // };

        // if let Some(block_mutex) = hexagon_link.blocks.get(block_uuid) {
        //     return Ok(block_mutex.clone());
        // }

        let msg = format!("Block with uuid '{block_uuid}' not found.");
        Err(AinariError::InvalidInput(msg))
    }

    /// Deletes a model from the collection.
    ///
    /// This method removes a model and all its associated data from the collection.
    ///
    /// # Arguments
    ///
    /// * `model_uuid` - A reference to the UUID of the model to delete
    ///
    /// # Returns
    ///
    /// * `Result<(), AinariError>` - Ok if the model was deleted successfully, Err otherwise
    pub fn delete_model(&mut self, model_uuid: &Uuid) -> Result<(), AinariError> {
        if !self.models.contains_key(model_uuid) {
            let msg: String = format!("Model with uuid '{model_uuid}' not found.");
            return Err(AinariError::InvalidInput(msg));
        }

        self.models.remove(model_uuid);

        Ok(())
    }

    /// Resets all output buffers in a model.
    ///
    /// This method calls the reset method on all output buffers in the specified model.
    ///
    /// # Arguments
    ///
    /// * `model_uuid` - A reference to the UUID of the model
    ///
    /// # Returns
    ///
    /// * `Result<(), AinariError>` - Ok if all outputs were reset successfully, Err otherwise
    pub fn reset_outputs(&self, model_uuid: &Uuid) -> Result<(), AinariError> {
        // let model_link = self.get_model(model_uuid)?;
        // let outputs = model_link.outputs.read().expect("mutex poisoned");
        // for output_mutex in outputs.values() {
        //     let mut output = output_mutex.lock().expect("mutex poisoned");
        //     output.reset_output();
        // }

        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use ainari_common::enums::*;
    use ainari_model_parser::model_meta_structs::Settings;
    use ainari_model_parser::model_parser::parse_model_template;
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn test_create_model() {
        // let model_uuid = Uuid::new_v4();
        // let name = "test_model".to_string();
        // let template = "version: 1
        // settings:
        //     neuron_cooldown: 1000000000.0;
        //     refractory_time: 1;
        //     max_connection_distance: 1;
        // hexagons:
        //     1,1,1;
        //     2,2,2;
        // axons:
        //     1,1,1 -> 2,2,2;
        // inputs:
        //     key1: 1,1,1;
        // outputs:
        //     key2: 2,2,2;"
        //     .to_string();

        // let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        // root_handler.models.clear();

        // {
        //     let mut parsed_model = parse_model_template(&name, &template).unwrap();
        //     parsed_model.uuid = model_uuid;
        //     let ret = root_handler.init_new_model(&model_uuid, &parsed_model);
        //     assert!(ret.is_ok());
        //     assert_eq!(root_handler.models.len(), 1);
        //     assert!(root_handler.models.contains_key(&model_uuid));

        //     let model = root_handler.models.get(&model_uuid).unwrap();
        //     assert!(model.model_interface.is_some());
        //     assert_eq!(model.model_meta.uuid, model_uuid);

        //     // check initial state of hexagon-data
        //     let hexagons = model.hexagon_data.read().expect("mutex poisoned");
        //     assert_eq!(hexagons.len(), 1);
        // }

        // assert!(root_handler.delete_model(&model_uuid).is_ok());
        // assert!(root_handler.delete_model(&model_uuid).is_err());
    }

    #[test]
    #[serial]
    fn test_add_blocks_to_model() {
        // let finish_counter = Arc::new(Mutex::new(FinishCounter::default()));
        // let model_uuid = Uuid::new_v4();
        // let hexagon_uuid0;
        // let hexagon_uuid1;
        // let model_name = "test_model".to_string();
        // let input_name = "test_input".to_string();
        // let output_name = "test_output".to_string();
        // let template = "version: 1
        // settings:
        //     neuron_cooldown: 1000000000.0;
        //     refractory_time: 1;
        //     max_connection_distance: 1;
        // hexagons:
        //     1,1,1;
        //     2,2,2;
        // axons:
        //     1,1,1 -> 2,2,2;
        // inputs:
        //     test_input: 1,1,1;
        // outputs:
        //     test_output: 2,2,2;"
        //     .to_string();

        // let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        // root_handler.models.clear();
        // let mut parsed_model = parse_model_template(&model_name, &template).unwrap();
        // parsed_model.uuid = model_uuid;
        // let _ = root_handler.init_new_model(&model_uuid, &parsed_model);

        // {
        //     let model = root_handler.models.get(&model_uuid).unwrap();
        //     if model.model_meta.hexagons.values().next().unwrap().is_input {
        //         hexagon_uuid0 = *model.model_meta.hexagons.keys().next().unwrap();
        //         hexagon_uuid1 = *model.model_meta.hexagons.keys().nth(1).unwrap();
        //     } else {
        //         hexagon_uuid1 = *model.model_meta.hexagons.keys().next().unwrap();
        //         hexagon_uuid0 = *model.model_meta.hexagons.keys().nth(1).unwrap();
        //     }
        // }

        // // prepare new blocks
        // let settings = Settings::default();
        // let core_block = Arc::new(Mutex::new(CoreBlock::new(
        //     &hexagon_uuid0,
        //     &model_uuid,
        //     &settings,
        // )));
        // let input_block = Arc::new(Mutex::new(InputBlock::new(
        //     &input_name,
        //     &hexagon_uuid0,
        //     &model_uuid,
        //     &
        //     0,
        // )));
        // let output_block = Arc::new(Mutex::new(OutputBlock::new(
        //     &hexagon_uuid1,
        //     &model_uuid,
        //     &output_name,
        // )));
        // let output_buffer = Arc::new(Mutex::new(OutputBuffer::new(
        //     &output_name,
        //     &hexagon_uuid1,
        //     &model_uuid,
        //     &OutputType::PlainOutput,
        // )));

        // let core_block_uuid = core_block.lock().unwrap().uuid;
        // let output_block_uuid = output_block.lock().unwrap().uuid;

        // // input-block and output-buffer are already added by initilizing of the model, so the names can not be added again
        // assert!(root_handler.add_output_buffer(&output_buffer).is_err());
        // assert!(root_handler.add_input_block(&input_block).is_err());

        // // add blocks to model
        // assert!(
        //     root_handler
        //         .add_core_block(&model_uuid, &hexagon_uuid0, &core_block_uuid, &core_block)
        //         .is_ok()
        // );
        // assert!(
        //     root_handler
        //         .add_output_block(
        //             &model_uuid,
        //             &hexagon_uuid1,
        //             &output_block_uuid,
        //             &output_block
        //         )
        //         .is_ok()
        // );
        // {
        //     let model = root_handler.models.get(&model_uuid).unwrap();
        //     let hexagons = model.hexagon_data.read().expect("mutex poisoned");
        //     assert_eq!(hexagons.len(), 2);
        //     // check hexagon 0
        //     {
        //         let hexagon0 = hexagons.get(&hexagon_uuid0).unwrap();
        //         assert_eq!(hexagon0.lock().expect("mutex poisoned").blocks.len(), 2);
        //         let inputs = model.inputs.read().expect("mutex poisoned");
        //         assert!(inputs.contains_key(&input_name));
        //     }

        //     // check hexagon 1
        //     {
        //         let hexagon1 = hexagons.get(&hexagon_uuid1).unwrap();
        //         assert_eq!(hexagon1.lock().expect("mutex poisoned").blocks.len(), 1);
        //         let outputs = model.outputs.read().expect("mutex poisoned");
        //         assert!(outputs.contains_key(&output_name));
        //     }
        // }

        // // check add blocks with the same ids again
        // assert!(
        //     root_handler
        //         .add_core_block(&model_uuid, &hexagon_uuid0, &core_block_uuid, &core_block)
        //         .is_err()
        // );
        // assert!(root_handler.add_input_block(&input_block).is_err());
        // assert!(
        //     root_handler
        //         .add_output_block(
        //             &model_uuid,
        //             &hexagon_uuid1,
        //             &output_block_uuid,
        //             &output_block
        //         )
        //         .is_err()
        // );
        // assert!(root_handler.add_output_buffer(&output_buffer).is_err());

        // // check getter
        // assert!(
        //     root_handler
        //         .get_input_block(&model_uuid, &input_name)
        //         .is_ok()
        // );
        // assert!(
        //     root_handler
        //         .get_input_block(&model_uuid, &output_name)
        //         .is_err()
        // );
        // assert!(
        //     root_handler
        //         .get_output_buffer(&model_uuid, &input_name)
        //         .is_err()
        // );
        // assert!(
        //     root_handler
        //         .get_output_buffer(&model_uuid, &output_name)
        //         .is_ok()
        // );
        // assert!(
        //     root_handler
        //         .get_block(
        //             &model_uuid,
        //             &hexagon_uuid0,
        //             &core_block.lock().expect("mutex poisoned").uuid
        //         )
        //         .is_ok()
        // );
        // assert!(
        //     root_handler
        //         .get_block(&model_uuid, &hexagon_uuid1, &Uuid::new_v4())
        //         .is_err()
        // );

        // // delete block and check again
        // {
        //     let _ = root_handler.delete_block(
        //         &model_uuid,
        //         &hexagon_uuid0,
        //         &core_block.lock().expect("mutex poisoned").uuid,
        //     );
        //     let model = root_handler.models.get(&model_uuid).unwrap();
        //     let hexagons = model.hexagon_data.read().expect("mutex poisoned");
        //     let hexagon0 = hexagons.get(&hexagon_uuid0).unwrap();
        //     assert_eq!(hexagon0.lock().expect("mutex poisoned").blocks.len(), 1);
        // }
    }

    #[test]
    #[serial]
    fn test_create_restore_checkpoint() {
        // let file_path = "/tmp/test_checkpoint".to_string();
        // let _ = fs::remove_file(&file_path).is_ok();
        // let finish_counter = Arc::new(Mutex::new(FinishCounter::default()));
        // let model_uuid = Uuid::new_v4();
        // let model_uuid_new = Uuid::new_v4();
        // let hexagon_uuid0;
        // let hexagon_uuid1;
        // let model_name = "test_model".to_string();
        // let input_name = "test_input".to_string();
        // let output_name = "test_output".to_string();
        // let template = "version: 1
        // settings:
        //     neuron_cooldown: 1000000000.0;
        //     refractory_time: 1;
        //     max_connection_distance: 1;
        // hexagons:
        //     1,1,1;
        //     2,2,2;
        // axons:
        //     1,1,1 -> 2,2,2;
        // inputs:
        //     test_input: 1,1,1;
        // outputs:
        //     test_output: 2,2,2;"
        //     .to_string();

        // let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        // root_handler.models.clear();
        // let mut parsed_model = parse_model_template(&model_name, &template).unwrap();
        // parsed_model.uuid = model_uuid;
        // let _ = root_handler.init_new_model(&model_uuid, &parsed_model);

        // {
        //     let model = root_handler.models.get(&model_uuid).unwrap();
        //     if model.model_meta.hexagons.values().next().unwrap().is_input {
        //         hexagon_uuid0 = *model.model_meta.hexagons.keys().next().unwrap();
        //         hexagon_uuid1 = *model.model_meta.hexagons.keys().nth(1).unwrap();
        //     } else {
        //         hexagon_uuid1 = *model.model_meta.hexagons.keys().next().unwrap();
        //         hexagon_uuid0 = *model.model_meta.hexagons.keys().nth(1).unwrap();
        //     }
        // }

        // // prepare new blocks
        // let settings = Settings::default();
        // let core_block_mutex = Arc::new(Mutex::new(CoreBlock::new(
        //     &hexagon_uuid0,
        //     &model_uuid,
        //     &settings,
        // )));
        // let input_block_mutex = Arc::new(Mutex::new(InputBlock::new(
        //     &input_name,
        //     &hexagon_uuid0,
        //     &model_uuid,
        //     0,
        // )));
        // let output_block_mutex = Arc::new(Mutex::new(OutputBlock::new(
        //     &hexagon_uuid1,
        //     &model_uuid,
        //     &output_name,
        // )));
        // let output_buffer_mutex = Arc::new(Mutex::new(OutputBuffer::new(
        //     &output_name,
        //     &hexagon_uuid1,
        //     &model_uuid,
        //     &OutputType::PlainOutput,
        // )));

        // let core_block_uuid = core_block_mutex.lock().unwrap().uuid;
        // let output_block_uuid = output_block_mutex.lock().unwrap().uuid;

        // // add blocks to model
        // let _ = root_handler.add_core_block(
        //     &model_uuid,
        //     &hexagon_uuid0,
        //     &core_block_uuid,
        //     &core_block_mutex,
        // );
        // let _ = root_handler.add_input_block(&input_block_mutex);
        // let _ = root_handler.add_output_block(
        //     &model_uuid,
        //     &hexagon_uuid1,
        //     &output_block_uuid,
        //     &output_block_mutex,
        // );
        // let _ = root_handler.add_output_buffer(&output_buffer_mutex);

        // // save and restore
        // let _ = root_handler.create_checkpoint(&model_uuid, &file_path);
        // let _ = root_handler.restore_checkpoint(&model_uuid_new, &file_path);

        // {
        //     let model = root_handler.models.get(&model_uuid_new).unwrap();
        //     let hexagons = model.hexagon_data.read().expect("mutex poisoned");
        //     assert_eq!(hexagons.len(), 2);
        //     // check hexagon 0
        //     {
        //         let hexagon0 = hexagons.get(&hexagon_uuid0).unwrap();
        //         assert_eq!(hexagon0.lock().expect("mutex poisoned").blocks.len(), 2);
        //         let inputs = model.inputs.read().expect("mutex poisoned");
        //         assert!(inputs.contains_key(&input_name));
        //     }

        //     // check hexagon 1
        //     {
        //         let hexagon1 = hexagons.get(&hexagon_uuid1).unwrap();
        //         assert_eq!(hexagon1.lock().expect("mutex poisoned").blocks.len(), 1);
        //         let outputs = model.outputs.read().expect("mutex poisoned");
        //         assert!(outputs.contains_key(&output_name));
        //     }
        // }

        // // check getter
        // assert!(
        //     root_handler
        //         .get_input_block(&model_uuid_new, &input_name)
        //         .is_ok()
        // );
        // assert!(
        //     root_handler
        //         .get_input_block(&model_uuid_new, &output_name)
        //         .is_err()
        // );
        // assert!(
        //     root_handler
        //         .get_output_buffer(&model_uuid_new, &input_name)
        //         .is_err()
        // );
        // assert!(
        //     root_handler
        //         .get_output_buffer(&model_uuid_new, &output_name)
        //         .is_ok()
        // );
        // assert!(
        //     root_handler
        //         .get_block(
        //             &model_uuid_new,
        //             &hexagon_uuid0,
        //             &core_block_mutex.lock().expect("mutex poisoned").uuid
        //         )
        //         .is_ok()
        // );
        // assert!(
        //     root_handler
        //         .get_block(&model_uuid_new, &hexagon_uuid1, &Uuid::new_v4())
        //         .is_err()
        // );
    }
}
