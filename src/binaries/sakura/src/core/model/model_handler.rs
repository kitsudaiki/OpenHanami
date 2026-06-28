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

use std::collections::HashMap;
// use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;

use ainari_common::error::AinariError;
use ainari_model_parser::model_meta_structs::*;

use crate::core::model::model::*;

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
    pub models: HashMap<Uuid, Arc<Mutex<Model>>>,
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

        let content = init_new_model(parsed_model.clone())?;
        self.models.insert(model_uuid.clone(), content);

        Ok(())
    }

    pub fn get_model(&self, model_uuid: &Uuid) -> Result<Arc<Mutex<Model>>, AinariError> {
        if let Some(model) = self.models.get(model_uuid) {
            Ok(model.clone())
        } else {
            let msg = format!("Model with uuid '{model_uuid}' not found.");
            Err(AinariError::InvalidInput(msg))
        }
    }

    pub fn delete_model(&mut self, model_uuid: &Uuid) -> Result<(), AinariError> {
        if !self.models.contains_key(model_uuid) {
            let msg: String = format!("Model with uuid '{model_uuid}' not found.");
            return Err(AinariError::InvalidInput(msg));
        }

        self.models.remove(model_uuid);

        Ok(())
    }

    // pub fn create_checkpoint(
    //     &self,
    //     model_uuid: &Uuid,
    //     file_path: &Path,
    // ) -> Result<(), AinariError> {
    //     Ok(())
    // }

    // pub fn restore_checkpoint(
    //     &mut self,
    //     model_uuid: &Uuid,
    //     file_path: &Path,
    // ) -> Result<(), AinariError> {
    //     Ok(())
    // }
}

#[cfg(test)]
mod tests {
    use ainari_model_parser::model_parser::parse_model_template;
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn test_create_model() {
        let model_uuid = Uuid::new_v4();
        let name = "test_model".to_string();
        let template = "version: 1
        settings:
            neuron_cooldown: 1000000000.0;
            refractory_time: 1;
            max_connection_distance: 1;
        hexagons:
            1,1,1;
            2,2,2;
        axons:
            1,1,1 -> 2,2,2;
        inputs:
            key1: 1,1,1;
        outputs:
            key2: 2,2,2;"
            .to_string();

        let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        root_handler.models.clear();

        {
            let mut parsed_model = parse_model_template(&name, &template).unwrap();
            parsed_model.uuid = model_uuid;
            let ret = root_handler.init_new_model(&model_uuid, &parsed_model);
            assert!(ret.is_ok());
            assert_eq!(root_handler.models.len(), 1);
            assert!(root_handler.models.contains_key(&model_uuid));

            let model_mutex = root_handler.get_model(&model_uuid).unwrap();
            let model = model_mutex.lock().expect("mutex poisoned");
            assert_eq!(model.uuid, model_uuid);
            // assert_eq!(model.hexagon_data.len(), 2);
            // assert_eq!(model.input_groups.len(), 1);
            // assert_eq!(model.output_groups.len(), 1);
        }

        assert!(root_handler.delete_model(&model_uuid).is_ok());
        assert!(root_handler.delete_model(&model_uuid).is_err());
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
