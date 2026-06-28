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

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path};

/// Computes the SHA-256 hash of the given input string and returns it as a hexadecimal string.
///
/// This function takes a string slice as input, processes it through the SHA-256 hashing algorithm,
/// and returns the resulting hash as a hexadecimal string.
pub fn sha256_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let result = hasher.finalize();
    hex::encode(result) // Convert hash bytes to a hexadecimal String
}

/// Extracts the token part from a Bearer token string.
///
/// This function splits the input string by spaces and checks if the first part is "Bearer".
/// If so, it returns the second part as Some(&str). Otherwise, it returns None.
pub fn split_bearer_token(token: &str) -> Option<&str> {
    let parts: Vec<&str> = token.splitn(2, ' ').collect();
    if parts.len() == 2 && parts[0] == "Bearer" {
        Some(parts[1])
    } else {
        None
    }
}

/// Creates a SHA-256 hash of the input string and returns it as a formatted hexadecimal string.
///
/// Similar to `sha256_hash`, but uses a different formatting approach for the result.
pub fn create_sha256_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// Checks if a given path is a safe subpath.
///
/// A safe subpath is one that doesn't contain parent directory references ("..") or
/// absolute path components. This is useful for preventing directory traversal attacks.
pub fn is_safe_subpath(path: &Path) -> bool {
    // Reject absolute paths immediately
    if path.is_absolute() {
        return false;
    }

    // Check each component of the path
    for comp in path.components() {
        match comp {
            Component::ParentDir => return false, // contains ".."
            Component::RootDir => return false,   // starts with "/"
            _ => {}
        }
    }

    true
}

/// Clears all files and subdirectories from the specified directory.
///
/// This function removes all contents of the given directory but leaves the directory itself intact.
///
/// # Arguments
///
/// * `dir` - A path to the directory to be cleared, which can be any type that implements `AsRef<Path>`
///
/// # Errors
///
/// Returns an `std::io::Error` if any operation fails during the directory clearing process.
pub fn clear_directory<P: AsRef<Path>>(dir: P) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            fs::remove_dir_all(&path)?;
        } else {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

pub fn push_chunked<T>(v: &mut Vec<T>, value: T) {
    if v.len() == v.capacity() {
        v.reserve(32);
    }
    v.push(value);
}

/// Computes a PCG (Permuted Congruential Generator) hash of the given u32 value.
///
/// This is a fast, non-cryptographic hash function suitable for general-purpose use.
/// The function updates the input value in place and returns the computed hash.
#[inline]
pub fn pcg_hash(input: &mut u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    *input = (word >> 22) ^ word;
    *input
}
