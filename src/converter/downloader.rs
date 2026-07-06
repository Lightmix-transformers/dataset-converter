use anyhow::{Context, Result};
use glob::glob;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;

use super::schema::DatasetSchema;

pub fn fetch_source_files(schema: &DatasetSchema) -> Result<Vec<(String, PathBuf)>> {
    match schema.source.source_type.as_str() {
        "local" => fetch_local_files(schema),
        "hf_dataset" => fetch_hf_files_sync(schema),
        _ => Err(anyhow::anyhow!(
            "Unsupported source type: '{}'. Use 'local' or 'hf_dataset'",
            schema.source.source_type
        )),
    }
}

fn fetch_local_files(schema: &DatasetSchema) -> Result<Vec<(String, PathBuf)>> {
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
        if file_config.path.contains('*') {
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
                    files.push((file_config.entity.clone(), path));
                }
            }
        } else {
            let path = PathBuf::from(&full_path);
            if !path.exists() {
                println!("Warning: file '{}' does not exist, skipping", full_path);
                continue;
            }
            files.push((file_config.entity.clone(), path));
        }
    }

    Ok(files)
}

fn fetch_hf_files_sync(schema: &DatasetSchema) -> Result<Vec<(String, PathBuf)>> {
    let dataset = schema
        .source
        .dataset
        .as_deref()
        .context("HF dataset source requires 'dataset' field")?;

    // Create download directory
    let cache_dir = format!(".cache/hf_datasets/{}", dataset.replace('/', "_"));
    std::fs::create_dir_all(&cache_dir)
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

        files.push((file_config.entity.clone(), dest_path));
    }

    Ok(files)
}

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
