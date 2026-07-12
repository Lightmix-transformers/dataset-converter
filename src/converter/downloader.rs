use anyhow::{Context, Result};
use glob::glob;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;

use crate::converter::schema::detect_split_from_path;

use super::schema::DatasetSchema;

/// Resolve source files according to the schema's `source_type`.
///
/// Dispatches on the configured source type:
/// - `"local"` → resolve glob patterns on disk
/// - `"hf_dataset"` → download from HuggingFace Hub
///
/// Returns a vector of `(entity_name, split_name, path)` tuples for each
/// resolved file. The split name is auto-detected from the file path
/// (e.g. `train/annotations.json` → `"train"`).
pub fn fetch_source_files(schema: &DatasetSchema) -> Result<Vec<(String, String, PathBuf)>> {
    match schema.source.source_type.as_str() {
        "local" => fetch_local_files(schema),
        "hf_dataset" => fetch_hf_files_sync(schema),
        _ => Err(anyhow::anyhow!(
            "Unsupported source type: '{}'. Use 'local' or 'hf_dataset'",
            schema.source.source_type
        )),
    }
}

/// Resolve local source files from the schema's file configuration.
///
/// For each entry in `schema.files`:
/// - Resolves relative paths against `source.path` (absolute paths used as-is)
/// - Expands glob patterns (`*`, `**`) via the `glob` crate
/// - Detects split name from the file path (e.g. `train/` → `"train"`)
///
/// Non-existent files and empty glob matches emit warnings but do not
/// abort — they are silently skipped.
fn fetch_local_files(schema: &DatasetSchema) -> Result<Vec<(String, String, PathBuf)>> {
    let base_path = schema
        .source
        .path
        .as_deref()
        .context("Local source requires 'path' field")?;

    let mut files = Vec::new();
    for file_config in &schema.files {
        let full_path = if file_config.path.starts_with('/') {
            file_config.path.clone()
        } else {
            format!("{}/{}", base_path.trim_end_matches('/'), file_config.path)
        };

        // Handle glob patterns
        if full_path.contains('*') {
            let matches: Vec<PathBuf> = glob(&full_path)
                .context(format!("Invalid glob pattern '{}'", file_config.path))?
                .filter_map(|r| r.ok())
                .collect();

            if matches.is_empty() {
                println!(
                    "Warning: no files matched '{}' in '{}'",
                    file_config.path, base_path
                );
            } else {
                for path in matches {
                    let split = detect_split_from_path(path.to_str().unwrap())
                        .unwrap_or_else(|| "default".to_string());
                    files.push((file_config.entity.clone(), split, path));
                }
            }
        } else {
            let path = PathBuf::from(&full_path);
            if !path.exists() {
                println!("Warning: file '{}' does not exist, skipping", full_path);
                continue;
            }
            let split = detect_split_from_path(&full_path).unwrap_or_else(|| "default".to_string());
            files.push((file_config.entity.clone(), split, path));
        }
    }

    Ok(files)
}

/// Download source files from the HuggingFace Hub.
///
/// For each file in `schema.files`:
/// - Constructs a direct download URL from `source.dataset` and `file.path`
/// - Downloads to `.cache/hf_datasets/{dataset}/` preserving directory structure
/// - Skips image directories (`.jpg`, `.png`) — only annotation JSON files are fetched
/// - Detects split name from the configured file path
fn fetch_hf_files_sync(schema: &DatasetSchema) -> Result<Vec<(String, String, PathBuf)>> {
    let dataset = schema
        .source
        .dataset
        .as_deref()
        .context("HF dataset source requires 'dataset' field")?;

    // Create download directory
    let cache_dir = format!(".cache/hf_datasets/{}", dataset.replace('/', "_"));
    create_dir_all(&cache_dir)
        .context(format!("Failed to create cache directory '{}'", cache_dir))?;

    let mut files = Vec::new();
    for file_config in &schema.files {
        // Skip image directories - we only need annotation JSON files
        if file_config.path.contains(".jpg") || file_config.path.contains(".png") {
            println!("Skipping image directory: {}", file_config.path);
            continue;
        }

        let hf_url = format!(
            "https://huggingface.co/datasets/{}/resolve/main/{}",
            dataset, file_config.path
        );

        let dest_path = PathBuf::from(&cache_dir).join(&file_config.path);
        if let Some(parent) = dest_path.parent() {
            create_dir_all(parent)?;
        }

        println!("Downloading: {} -> {}", hf_url, dest_path.display());
        download_file(&hf_url, &dest_path)?;

        let split =
            detect_split_from_path(&file_config.path).unwrap_or_else(|| "default".to_string());
        files.push((file_config.entity.clone(), split, dest_path));
    }

    Ok(files)
}

/// Download a single file from `url` and write it to `dest`.
///
/// Uses a blocking HTTP GET. The entire response body is loaded into memory
fn download_file(url: &str, dest: &PathBuf) -> Result<()> {
    let response =
        reqwest::blocking::get(url).with_context(|| format!("Failed to connect to '{}'", url))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Download failed with status: {}",
            response.status()
        ));
    }

    let total_bytes = response.content_length().unwrap_or(0);
    println!("  Size: {} bytes", total_bytes);

    let mut file =
        File::create(dest).with_context(|| format!("Failed to create '{}'", dest.display()))?;

    let bytes = response
        .bytes()
        .map_err(|e| anyhow::anyhow!("Failed to read response body: {}", e))?;

    // Write all at once for simplicity - progress tracking on large files can be added later
    file.write_all(&bytes).context("Failed to write data")?;

    println!("  Done: {} bytes", bytes.len());
    Ok(())
}
