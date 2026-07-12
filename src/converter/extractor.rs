use anyhow::{Context, Result};
use image::ImageReader;
use jsonpath_lib::JsonPathError;
use polars::prelude::*;
use serde_json::Value;
use std::fs::{File, create_dir_all, read, read_to_string};
use std::io::Cursor;
use std::path::PathBuf;
use std::{collections::HashMap, path::Path};

use crate::converter::merger::merge_entities;
use crate::formats::{Compression, DatasetFormat, write_dataset};

use super::schema::{DatasetSchema, FieldConfig, detect_split_from_path};

/// Top-level entry point for single-file extraction (split_strategy: none).
///
/// Reads a source file, extracts fields per entity, merges entities via
/// configured joins, and writes the result to `output_path`.
///
/// Dispatches on file extension:
/// - `.csv` → chunked CSV reader with binary resolution
/// - other (JSON) → eager parse + JSONPath queries
///
/// # Arguments
/// * `format` — output format (Parquet or Arrow IPC)
/// * `path` — path to the source file
/// * `schema` — dataset schema defining fields, entities, and joins
/// * `output_path` — destination path for the written file
/// * `chunk_size` — rows per chunk when processing binary data
/// * `compression` — compression algorithm for the output
pub fn extract_and_write(
    format: DatasetFormat,
    path: &str,
    schema: &DatasetSchema,
    output_path: &str,
    chunk_size: usize,
    compression: Compression,
) -> Result<()> {
    let mut entity_fields: HashMap<&str, Vec<&FieldConfig>> = HashMap::new();
    for field in &schema.fields {
        let entity = field.entity.as_deref().unwrap_or("data");
        entity_fields.entry(entity).or_default().push(field);
    }

    if path.ends_with(".csv") {
        extract_csv_incremental(
            format,
            path,
            &entity_fields,
            output_path,
            chunk_size,
            compression,
        )
    } else {
        // JSON path stays eager; metadata files are small.
        let contents =
            read_to_string(path).with_context(|| format!("Failed to read '{}'", path))?;
        let json = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse '{}'", path))?;

        let mut entity_lfs: Vec<(String, LazyFrame)> = Vec::new();
        for (entity, fields) in &entity_fields {
            let lf = extract_entity_data(&json, fields, path)?;
            entity_lfs.push((entity.to_string(), lf));
        }

        let mut merged = merge_entities(entity_lfs, schema)?
            .collect()
            .map_err(|e| anyhow::anyhow!("Failed to collect: {}", e))
            .unwrap();

        let file = File::create(output_path)
            .with_context(|| format!("Failed to create '{}'", output_path))?;

        write_dataset(
            format.clone(),
            file.try_clone().unwrap(),
            compression.clone(),
            &mut merged,
            path,
        );

        Ok(())
    }
}

/// Iterate over entities in a CSV file and extract each with binary support.
///
/// For every entity defined in the schema, delegates to
/// `extract_csv_with_binary_chunked` so large image payloads are
/// processed incrementally rather than loaded entirely into memory.
fn extract_csv_incremental(
    format: DatasetFormat,
    path: &str,
    entity_fields: &HashMap<&str, Vec<&FieldConfig>>,
    output_path: &str,
    chunk_size: usize,
    compression: Compression,
) -> Result<()> {
    for (entity, fields) in entity_fields {
        if fields.is_empty() {
            continue;
        }

        println!(
            "  -> Extracting '{}' with binary fields in chunks...",
            entity
        );
        extract_csv_with_binary_chunked(
            format.clone(),
            path,
            entity,
            fields,
            output_path,
            chunk_size,
            compression.clone(),
        )?;
    }

    Ok(())
}

/// Resolve which CSV column holds image paths for binary field matching.
///
/// Priority:
/// 1. A `column_name` containing "path" or "file" (case-insensitive)
/// 2. The first String-typed column among configured fields
///
/// Returns an error if no candidate can be found or the column is missing.
fn resolve_path_column(df: &DataFrame, fields: &[&FieldConfig]) -> Result<String> {
    let name_hint = fields.iter().find_map(|f| {
        f.column_name.as_ref().filter(|c| {
            let lower = c.to_lowercase();
            lower.contains("path") || lower.contains("file")
        })
    });

    let dtype_fallback = || {
        fields.iter().find_map(|f| {
            f.column_name.as_ref().filter(|c| {
                df.column(c)
                    .map(|col| col.dtype() == &DataType::String)
                    .unwrap_or(false)
            })
        })
    };

    let candidate = name_hint
        .or_else(dtype_fallback)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Cannot find path column for binary field matching"))?;

    df.column(&candidate)
        .with_context(|| format!("Path column '{}' not found in CSV", candidate))?;

    Ok(candidate)
}

/// Build a mapping from DataFrame row index → resolved filesystem path.
///
/// For each path resolved by the binary field's glob pattern, looks up the
/// corresponding CSV row indices in `path_to_indices` and records the
/// association. Used during chunked binary resolution to find which file
/// belongs to which row.
fn build_row_to_file_map(
    file_paths: &[PathBuf],
    path_to_indices: &HashMap<String, Vec<usize>>,
) -> HashMap<usize, PathBuf> {
    let mut row_to_path = HashMap::new();
    for file_path in file_paths {
        let filename = file_path.to_string_lossy().to_string();
        if let Some(indices) = path_to_indices.get(&filename) {
            for &idx in indices {
                row_to_path.entry(idx).or_insert_with(|| file_path.clone());
            }
        }
    }
    row_to_path
}

/// Read a file from disk, optionally decoding it as an image.
///
/// Uses a per-chunk LRU-style cache to avoid re-reading the same file
/// multiple times within a single chunk. Returns `None` if the file
/// cannot be read or decoded.
fn load_binary(
    path: &Path,
    cache: &mut HashMap<PathBuf, Vec<u8>>,
    decode: bool,
    image_mode: &str,
) -> Option<Vec<u8>> {
    if let Some(bytes) = cache.get(path) {
        return Some(bytes.clone());
    }

    let raw = read(path).ok()?;
    let bytes = if decode {
        decode_image(&raw, image_mode).ok()?
    } else {
        raw
    };

    cache.insert(path.to_path_buf(), bytes.clone());
    Some(bytes)
}

/// Load binary data for a range of rows within a chunk.
///
/// Iterates `start..end`, looking up each row index in `row_to_path`.
/// Rows with no matching file get `None`. Returns a vector aligned
/// to the chunk (length = `end - start`).
fn fill_binary_chunk(
    start: usize,
    end: usize,
    row_to_path: &HashMap<usize, PathBuf>,
    decode: bool,
    image_mode: &str,
) -> Vec<Option<Vec<u8>>> {
    let mut values = vec![None; end - start];
    let mut cache = HashMap::<PathBuf, Vec<u8>>::new();

    for (local_idx, row_idx) in (start..end).enumerate() {
        if let Some(path) = row_to_path.get(&row_idx) {
            values[local_idx] = load_binary(path, &mut cache, decode, image_mode);
        }
    }

    values
}

/// Build a bidirectional index from path strings → DataFrame row indices.
///
/// For each non-null value in the path column, registers two keys:
/// the raw string (e.g. `"train/cat/001.JPEG"`) and the base_dir-prefixed
/// form (e.g. `"data/train/cat/001.JPEG"`). This allows matching against
/// both relative CSV paths and absolute glob-resolved filesystem paths.
fn build_path_to_indices(
    str_chunked: &StringChunked,
    base_dir: &str,
) -> HashMap<String, Vec<usize>> {
    let mut path_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, opt) in str_chunked.iter().enumerate() {
        if let Some(path_str) = opt {
            path_to_indices
                .entry(path_str.to_string())
                .or_default()
                .push(i);
            path_to_indices
                .entry(format!("{}/{}", base_dir, path_str))
                .or_default()
                .push(i);
        }
    }
    path_to_indices
}

/// Chunked CSV extraction with binary field resolution (no split partitioning).
///
/// Reads the entire CSV eagerly, builds path→index and index→file mappings,
/// then processes rows in chunks.
fn extract_csv_with_binary_chunked(
    format: DatasetFormat,
    path: &str,
    entity: &str,
    fields: &[&FieldConfig],
    output_path: &str,
    chunk_size: usize,
    compression: Compression,
) -> Result<()> {
    let df_original = read_csv_file_eager(path)?;
    let path_col_name = resolve_path_column(&df_original, fields)?;

    let mut non_binary_fields: Vec<&FieldConfig> = Vec::new();
    let mut binary_fields: Vec<(&FieldConfig, &str)> = Vec::new();
    for field in fields {
        if field.dtype == "binary" {
            let source = field
                .source
                .as_ref()
                .context(format!("Binary field '{}' requires 'source'", field.name))?;
            binary_fields.push((field, source));
        } else {
            non_binary_fields.push(field);
        }
    }

    let base_dir = Path::new(path)
        .parent()
        .map(|p| p.to_str().unwrap_or("."))
        .unwrap_or(".");

    let s = df_original.column(&path_col_name)?.as_materialized_series();
    let str_chunked = s.str().context("Path column must be string type")?;
    let path_to_indices = build_path_to_indices(str_chunked, base_dir);

    let mut row_to_path_by_field = Vec::with_capacity(binary_fields.len());
    for (field, source) in &binary_fields {
        let file_paths = resolve_patterns(source, base_dir)?;
        if field.decode {
            println!(
                "  Decoding {} images ({})...",
                file_paths.len(),
                field.image_mode
            );
        }
        row_to_path_by_field.push(build_row_to_file_map(&file_paths, &path_to_indices));
    }

    let mut non_binary_exprs: Vec<Expr> = Vec::new();
    for field in &non_binary_fields {
        let source_col = field.column_name.clone().or_else(|| {
            field
                .jsonpath
                .as_ref()
                .map(|jp| jp.strip_prefix("$[*].").unwrap_or(jp).to_string())
        });
        if let Some(source_col) = source_col {
            non_binary_exprs.push(col(&source_col).alias(&field.name));
        }
    }
    let renamed_df = match non_binary_exprs.is_empty() {
        true => None,
        false => Some(
            df_original
                .clone()
                .lazy()
                .select(non_binary_exprs)
                .collect()
                .map_err(|e| anyhow::anyhow!("Failed to select: {}", e))?,
        ),
    };

    let total_rows = df_original.height();
    println!(
        "  Processing {} rows in chunks of {}...",
        total_rows, chunk_size
    );

    // Process binary data in chunks to manage memory, collect all chunks, then write once.
    let mut chunks: Vec<DataFrame> = Vec::new();

    let file =
        File::create(output_path).with_context(|| format!("Failed to create '{}'", output_path))?;
    for start in (0..total_rows).step_by(chunk_size) {
        let end = (start + chunk_size).min(total_rows);
        println!("    Chunk {}-{}...", start, end);

        let mut chunk = match renamed_df {
            Some(ref renamed_df) => renamed_df.slice(start as i64, end - start),
            None => DataFrame::empty(),
        };

        for (bidx, (field, _)) in binary_fields.iter().enumerate() {
            let row_to_path = &row_to_path_by_field[bidx];
            let binary_values =
                fill_binary_chunk(start, end, row_to_path, field.decode, &field.image_mode);
            chunk.with_column(Column::new(field.name.as_str().into(), &binary_values))?;
        }

        println!("  Writing {} rows...", chunk.height());
        write_dataset(
            format.clone(),
            file.try_clone().context("Failed to clone file handle")?,
            compression.clone(),
            &mut chunk,
            output_path,
        );

        chunks.push(chunk);
    }

    println!("  -> Done extracting '{}'", entity);
    Ok(())
}

fn read_csv_file_eager(path: &str) -> Result<DataFrame> {
    let options = CsvReadOptions::default().with_has_header(true);
    options
        .try_into_reader_with_file_path(Some(path.into()))
        .map_err(|e| anyhow::anyhow!("Failed to create CSV reader '{}': {}", path, e))?
        .finish()
        .with_context(|| format!("Failed to read CSV '{}'", path))
}

/// Extract fields for a single entity from a parsed JSON document.
///
/// For each field config:
/// - If `jsonpath` is set → query the JSON tree and convert to a typed column
/// - If `dtype == "binary"` → resolve glob patterns, read files, decode images
/// - Otherwise → error (field must have jsonpath or be binary)
///
/// Single-value columns are broadcast to match the longest column so all
/// columns align for DataFrame construction.
fn extract_entity_data(
    json: &Value,
    fields: &[&FieldConfig],
    source_path: &str,
) -> Result<LazyFrame> {
    if fields.is_empty() {
        return Ok(DataFrame::empty().lazy());
    }

    let base_dir = Path::new(source_path)
        .parent()
        .map(|p| p.to_str().unwrap_or("."))
        .unwrap_or(".");

    let mut columns: Vec<Column> = Vec::with_capacity(fields.len());
    for field in fields {
        let column = if let Some(ref jsonpath) = field.jsonpath {
            let values = query_path(json, jsonpath)?;
            build_column(&field.name, &values, &field.dtype)?
        } else if field.dtype == "binary" {
            let source = field.source.as_ref().context(format!(
                "Binary field '{}' requires 'source' attribute",
                field.name
            ))?;
            load_binary_field(
                &field.name,
                source,
                base_dir,
                field.decode,
                &field.image_mode,
            )
            .context(format!("Failed to load binary field '{}'", field.name))?
        } else {
            return Err(anyhow::anyhow!(
                "Field '{}': must have 'jsonpath' or 'column_name' for non-binary types",
                field.name
            ));
        };
        columns.push(column);
    }

    // Broadcast single-value columns up to the others' length.
    if !columns.is_empty() {
        let len = columns[0].len();
        for col in &mut columns[1..] {
            if col.len() == 1 && len > 1 {
                *col = broadcast_column(col, len)?;
            }
        }
    }

    let df = DataFrame::new_infer_height(columns)
        .context("Failed to build DataFrame from extracted fields")?;
    Ok(df.lazy())
}

/// Execute a JSONPath query against a parsed JSON document.
///
/// Returns a vector of wrapped `Value`s — one per match. Non-matching
/// positions are represented as `Some(Value)` since jsonpath_lib only
/// returns hits.
fn query_path(json: &Value, path: &str) -> Result<Vec<Option<Value>>> {
    let results = jsonpath_lib::select(json, path)
        .map_err(|e: JsonPathError| anyhow::anyhow!("JSONPath '{}' error: {}", path, e))?;

    Ok(results.into_iter().map(|v| Some(v.clone())).collect())
}

/// Resolve glob patterns into filesystem paths and read binary data.
///
/// `pattern` supports comma-separated globs, e.g. "train/**/*.JPEG,val/**/*.JPEG".
/// Files are sorted for deterministic ordering. If `decode` is true,
/// each file is decoded as an image in the specified mode; otherwise
/// raw bytes are returned.
fn load_binary_field(
    name: &str,
    pattern: &str,
    base_dir: &str,
    decode: bool,
    image_mode: &str,
) -> Result<Column> {
    let mut matched_paths = resolve_patterns(pattern, base_dir)?;
    matched_paths.sort();

    let bytes: Vec<Vec<u8>> = matched_paths
        .iter()
        .filter_map(|p| read(p).ok())
        .filter_map(|raw| {
            if decode {
                decode_image(&raw, image_mode).ok()
            } else {
                Some(raw)
            }
        })
        .collect();

    Ok(Column::new(name.into(), &bytes))
}

/// Decode raw image bytes into a flat pixel array.
///
/// Supports `"rgb"` and `"grayscale"`/`"gray"` modes. The output is
/// a contiguous `Vec<u8>` (R,G,B,R,G,B,... or G,G,G,...) suitable for
/// embedding directly into Parquet binary columns.
fn decode_image(bytes: &[u8], mode: &str) -> Result<Vec<u8>> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("Failed to detect image format")?
        .decode()
        .context("Failed to decode image")?;

    match mode.to_lowercase().as_str() {
        "grayscale" | "gray" => Ok(img.into_luma8().into_raw()),
        "rgb" => Ok(img.into_rgb8().into_raw()),
        _ => Err(anyhow::anyhow!(
            "Invalid image mode '{}'. Use 'grayscale' or 'rgb'",
            mode
        )),
    }
}

/// Resolve one or more glob patterns into a list of filesystem paths.
///
/// Supports comma-separated patterns (e.g. `"train/**/*.JPEG,val/**/*.JPEG"`).
/// Patterns containing `**` use a custom recursive resolver; others
/// delegate to the `glob` crate. Each pattern is resolved relative to
/// `base_dir` unless it starts with `/`.
fn resolve_patterns(pattern: &str, base_dir: &str) -> Result<Vec<PathBuf>> {
    let mut results = Vec::new();

    for part in pattern.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        let full_path = resolve_path(part, base_dir);

        if full_path.contains("**") {
            results.extend(resolve_recursive(&full_path));
        } else if let Ok(entries) = glob::glob(&full_path) {
            results.extend(entries.filter_map(|r| r.ok()));
        }
    }

    Ok(results)
}

/// Resolve a relative path against `base_dir`, or return as-is if absolute.
fn resolve_path(path: &str, base_dir: &str) -> String {
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("{}/{}", base_dir, path)
    }
}

/// Walk a glob pattern and collect all matching paths.
///
/// Used as a fallback for `**` patterns where the standard `glob` crate
/// may not behave consistently across platforms.
fn resolve_recursive(pattern: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(paths) = glob::glob(pattern) {
        for path in paths.filter_map(|r| r.ok()) {
            results.push(path);
        }
    }
    results
}

/// Filter and cast JSON values to a numeric type.
///
/// Iterates `values`, keeping only entries that are `Some(Number)`,
/// applying converter `f` (e.g. `|n| n.as_u64()`). Returns a dense
/// vector with no gaps — nulls are silently dropped.
fn collect_numeric<T>(
    values: &[Option<Value>],
    f: impl Fn(&serde_json::Number) -> Option<T>,
) -> Vec<T> {
    values
        .iter()
        .filter_map(|v| v.as_ref()?.as_number().and_then(&f))
        .collect()
}

/// Convert a slice of JSON values into a typed Polars Column.
///
/// Dispatches on `dtype`:
/// - Numeric (`u32`, `u64`, `i32`, `i64`, `f32`, `f64`) → filter + cast
/// - `bool` → extract boolean values
/// - `binary` → encode strings as raw bytes
/// - anything else (default: `string`) → serialize to string
fn build_column(name: &str, values: &[Option<Value>], dtype: &str) -> Result<Column> {
    let column = match dtype {
        "u32" => Column::new(
            name.into(),
            &collect_numeric(values, |n| n.as_u64().map(|x| x as u32)),
        ),
        "u64" => Column::new(name.into(), &collect_numeric(values, |n| n.as_u64())),
        "i32" => Column::new(
            name.into(),
            &collect_numeric(values, |n| n.as_i64().map(|x| x as i32)),
        ),
        "i64" => Column::new(name.into(), &collect_numeric(values, |n| n.as_i64())),
        "f32" => Column::new(
            name.into(),
            &collect_numeric(values, |n| n.as_f64().map(|x| x as f32)),
        ),
        "f64" => Column::new(name.into(), &collect_numeric(values, |n| n.as_f64())),
        "bool" => Column::new(
            name.into(),
            &values
                .iter()
                .filter_map(|v| v.as_ref().and_then(|val| val.as_bool()))
                .collect::<Vec<_>>(),
        ),
        "binary" => Column::new(
            name.into(),
            &values
                .iter()
                .filter_map(|v| {
                    v.as_ref()
                        .and_then(|val| val.as_str())
                        .map(|s| s.as_bytes().to_vec())
                })
                .collect::<Vec<_>>(),
        ),
        _ => Column::new(
            name.into(),
            &values
                .iter()
                .filter_map(|v| {
                    v.as_ref().map(|val| match val {
                        Value::String(s) => s.clone(),
                        _ => serde_json::to_string(val).unwrap_or_default(),
                    })
                })
                .collect::<Vec<_>>(),
        ),
    };

    Ok(column)
}

/// Broadcast a single-value column to match the length of others.
///
/// Used when an entity has scalar metadata fields (e.g. a dataset-level
/// `source` tag) alongside array fields (e.g. image paths). Replicates
/// index 0 across `target_len` rows.
fn broadcast_column(col: &Column, target_len: usize) -> Result<Column> {
    let series = col
        .as_series()
        .ok_or_else(|| anyhow::anyhow!("Column has no series"))?;
    Ok(Column::from(series.new_from_index(0, target_len)))
}

/// Extract data for one split under the `file_path` split strategy.
///
/// Given a list of (entity, file_path) pairs for a single split:
/// 1. Groups schema fields by entity
/// 2. Reads each entity's JSON file and extracts fields via JSONPath
/// 3. Merges entities via configured join rules
/// 4. Writes the merged DataFrame to `output_path`
///
/// # Arguments
/// * `split_files` — (entity_name, file_path) pairs for this split
/// * `schema` — dataset schema defining fields, entities, and joins
/// * `output_path` — destination path for the written file
/// * `format` — output format (Parquet or Arrow IPC)
/// * `compression` — compression algorithm for the output
pub fn extract_split(
    split_files: &[(String, PathBuf)],
    schema: &DatasetSchema,
    output_path: &str,
    format: DatasetFormat,
    compression: Compression,
) -> Result<()> {
    // Group fields by entity
    let mut entity_fields: HashMap<&str, Vec<&FieldConfig>> = HashMap::new();
    for field in &schema.fields {
        let entity = field.entity.as_deref().unwrap_or("data");
        entity_fields.entry(entity).or_default().push(field);
    }

    // Build lookup: entity -> file path for this split
    let mut entity_to_file: HashMap<String, &PathBuf> = HashMap::new();
    for (entity, path) in split_files {
        entity_to_file.insert(entity.clone(), path);
    }

    // Extract each entity from its corresponding file
    let mut entity_lfs: Vec<(String, LazyFrame)> = Vec::new();
    for (entity, fields) in &entity_fields {
        let file_path = entity_to_file
            .get(*entity)
            .ok_or_else(|| anyhow::anyhow!("No source file found for entity '{}'", entity))?;

        let path_str = file_path.to_str().context("Invalid file path")?;
        let contents =
            read_to_string(file_path).with_context(|| format!("Failed to read '{}'", path_str))?;
        let json = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse '{}'", path_str))?;

        let lf = extract_entity_data(&json, fields, path_str)?;
        entity_lfs.push((entity.to_string(), lf));
    }

    // Merge all entities via configured joins
    let merged = merge_entities(entity_lfs, schema)?
        .collect()
        .map_err(|e| anyhow::anyhow!("Failed to collect merged DataFrame: {}", e))?;

    // Write output
    let file =
        File::create(output_path).with_context(|| format!("Failed to create '{}'", output_path))?;

    let mut df = merged;
    write_dataset(
        format.clone(),
        file.try_clone().context("Failed to clone file handle")?,
        compression.clone(),
        &mut df,
        output_path,
    );

    Ok(())
}

/// Extract CSV data and partition into per-split outputs (row_column strategy).
///
/// 1. Reads the entire CSV eagerly
/// 2. Detects split name for each row from the configured `split_col`
///    (e.g. path `"train/cat/001.JPEG"` → split `"train"`)
/// 3. Groups row indices by split name
/// 4. For each split: selects rows, resolves binary fields, writes output
///
/// # Arguments
/// * `format` — output format (Parquet or Arrow IPC)
/// * `path` — path to the source CSV file
/// * `schema` — dataset schema defining fields and output config
/// * `split_col` — name of the column containing split-detectable values
/// * `output_dir` — parent directory; per-split files written to `{output_dir}/{split}/`
/// * `chunk_size` — rows per chunk when processing binary data
/// * `compression` — compression algorithm for the output
pub fn extract_csv_by_split(
    format: DatasetFormat,
    path: &str,
    schema: &DatasetSchema,
    split_col: &str,
    output_dir: &str,
    chunk_size: usize,
    compression: Compression,
) -> Result<()> {
    let df = read_csv_file_eager(path)?;

    // Verify split column exists and is string type
    let col_series = df
        .column(split_col)
        .with_context(|| format!("Split column '{}' not found in CSV", split_col))?;
    let col_str = col_series
        .str()
        .with_context(|| format!("Split column '{}' is not a string type", split_col))?;

    // Group row indices by detected split name
    let mut split_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, opt) in col_str.iter().enumerate() {
        if let Some(path_value) = opt {
            let split = detect_split_from_path(path_value).unwrap_or_else(|| "default".to_string());
            split_indices.entry(split).or_default().push(i);
        }
    }

    if split_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No splits detected from column '{}'. Check that paths contain train/val/test keywords.",
            split_col
        ));
    }

    // Get fields for this entity (CSV sources typically have one entity)
    let mut entity_fields: HashMap<&str, Vec<&FieldConfig>> = HashMap::new();
    for field in &schema.fields {
        let entity = field.entity.as_deref().unwrap_or("data");
        entity_fields.entry(entity).or_default().push(field);
    }

    // Write per-split output
    for (split_name, indices) in &split_indices {
        println!("  Split '{}': {} rows", split_name, indices.len());

        create_dir_all(&output_dir)
            .with_context(|| format!("Failed to create directory '{}'", output_dir))?;

        let output_path = format!(
            "{}/{}.{}",
            output_dir.trim_end_matches('/'),
            split_name,
            format
        );

        // Select rows for this split and write with binary field support
        let split_df = select_rows_by_indices(&df, indices)
            .with_context(|| format!("Failed to select rows for split '{}'", split_name))?;

        for (entity, fields) in &entity_fields {
            if fields.is_empty() {
                continue;
            }
            extract_split_csv_chunked(
                format.clone(),
                path,
                entity,
                fields,
                &split_df,
                &output_path,
                chunk_size,
                compression.clone(),
            )?;
        }
    }

    Ok(())
}

/// Select specific rows from a DataFrame by integer indices.
///
/// Converts indices to a UInt32 ChunkedArray and uses Polars' `take`
/// operation. Order is preserved — the output rows appear in the same
/// order as the input indices.
fn select_rows_by_indices(df: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
    let values: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let ca = UInt32Chunked::from_slice("idx".into(), &values);
    df.take(&ca)
        .map_err(|e| anyhow::anyhow!("Failed to select rows: {}", e))
}

/// Write a pre-filtered DataFrame (one split) with binary fields resolved.
///
/// Takes a DataFrame that has already been filtered to one split's rows,
/// resolves non-binary field renames, builds path→index and index→file
/// mappings for binary fields, then processes in chunks to keep memory
/// bounded. All chunks are collected into a `Vec<DataFrame>`, concatenated,
/// and written once.
fn extract_split_csv_chunked(
    format: DatasetFormat,
    source_path: &str,
    entity: &str,
    fields: &[&FieldConfig],
    split_df: &DataFrame,
    output_path: &str,
    chunk_size: usize,
    compression: Compression,
) -> Result<()> {
    let path_col_name = resolve_path_column(split_df, fields)?;

    let mut non_binary_fields: Vec<&FieldConfig> = Vec::new();
    let mut binary_fields: Vec<(&FieldConfig, &str)> = Vec::new();
    for field in fields {
        if field.dtype == "binary" {
            let source = field
                .source
                .as_ref()
                .context(format!("Binary field '{}' requires 'source'", field.name))?;
            binary_fields.push((field, source));
        } else {
            non_binary_fields.push(field);
        }
    }

    let base_dir = Path::new(source_path)
        .parent()
        .map(|p| p.to_str().unwrap_or("."))
        .unwrap_or(".");

    // Build rename expressions for non-binary fields.
    // Also track what the path column gets renamed to.
    let mut exprs: Vec<Expr> = Vec::new();
    let mut effective_path_col = path_col_name.clone();
    for f in &non_binary_fields {
        let source = f
            .column_name
            .as_ref()
            .map(|s| s.to_string())
            .unwrap_or_else(|| f.name.clone());
        exprs.push(col(&source).alias(&f.name));
        // If this field's source is the path column, track its new name
        if source == path_col_name {
            effective_path_col = f.name.clone();
        }
    }

    let renamed_df = if !exprs.is_empty() {
        split_df
            .clone()
            .lazy()
            .select(exprs)
            .collect()
            .map_err(|e| anyhow::anyhow!("Failed to select: {}", e))?
    } else {
        split_df.clone()
    };

    // Resolve binary field mappings using the renamed DataFrame's path column.
    let s = renamed_df
        .column(&effective_path_col)
        .with_context(|| {
            format!(
                "Path column '{}' not found after renaming (was '{}')",
                effective_path_col, path_col_name
            )
        })?
        .as_materialized_series();
    let str_chunked = s.str().context("Path column must be string type")?;
    let path_to_indices = build_path_to_indices(str_chunked, base_dir);

    let mut row_to_path_by_field = Vec::with_capacity(binary_fields.len());
    for (field, source) in &binary_fields {
        let file_paths = resolve_patterns(source, base_dir)?;
        if field.decode {
            println!(
                "  Decoding {} images ({})...",
                file_paths.len(),
                field.image_mode
            );
        }
        row_to_path_by_field.push(build_row_to_file_map(&file_paths, &path_to_indices));
    }

    let total_rows = renamed_df.height();
    println!(
        "  Processing {} rows in chunks of {}...",
        total_rows, chunk_size
    );

    let file =
        File::create(output_path).with_context(|| format!("Failed to create '{}'", output_path))?;
    for start in (0..total_rows).step_by(chunk_size) {
        let end = (start + chunk_size).min(total_rows);
        println!("    Chunk {}-{}...", start, end);

        let mut chunk = renamed_df.slice(start as i64, end - start);

        for (bidx, (field, _)) in binary_fields.iter().enumerate() {
            let row_to_path = &row_to_path_by_field[bidx];
            let mut binary_values = vec![None; end - start];
            let mut cache = HashMap::<PathBuf, Vec<u8>>::new();

            for (local_idx, chunk_row) in (start..end).enumerate() {
                if let Some(row_path) = renamed_df
                    .column(&effective_path_col)?
                    .str()?
                    .get(chunk_row)
                {
                    // Use the full path value as lookup key (matches path_to_indices keys)
                    if let Some(indices) = path_to_indices.get(row_path) {
                        for &idx in indices {
                            if let Some(fp) = row_to_path.get(&idx) {
                                binary_values[local_idx] =
                                    load_binary(fp, &mut cache, field.decode, &field.image_mode);
                                break;
                            }
                        }
                    }
                }
            }
            chunk.with_column(Column::new(field.name.as_str().into(), &binary_values))?;
        }

        println!("  Writing {} rows...", chunk.height());

        write_dataset(
            format.clone(),
            file.try_clone().unwrap(),
            compression.clone(),
            &mut chunk,
            output_path,
        );
    }

    println!("  -> Done extracting '{}' for split", entity);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{env::temp_dir, fs::create_dir_all};

    use super::*;

    #[test]
    fn build_row_to_file_map_matches_raw_and_prefixed_keys() {
        let base_dir = "data";
        let file_a = PathBuf::from("data/images/a.jpg");
        let file_b = PathBuf::from("data/images/b.jpg");
        let file_unmatched = PathBuf::from("data/images/unused.jpg");
        let file_paths = vec![file_a.clone(), file_b.clone(), file_unmatched];

        let mut path_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        path_to_indices.insert("images/a.jpg".to_string(), vec![0]);
        path_to_indices.insert(format!("{}/images/a.jpg", base_dir), vec![0]);
        path_to_indices.insert("data/images/b.jpg".to_string(), vec![1]);

        let row_to_path = build_row_to_file_map(&file_paths, &path_to_indices);

        assert_eq!(row_to_path.get(&0), Some(&file_a));
        assert_eq!(row_to_path.get(&1), Some(&file_b));
        assert_eq!(row_to_path.len(), 2);
    }

    #[test]
    fn fill_binary_chunk_preserves_row_alignment_with_gaps() {
        let dir = temp_dir().join(format!("extract_test_{}_{}", std::process::id(), line!()));
        create_dir_all(&dir).unwrap();

        let file0 = dir.join("row0.bin");
        let file2 = dir.join("row2.bin");
        std::fs::write(&file0, b"AAA").unwrap();
        std::fs::write(&file2, b"CCC").unwrap();

        let mut row_to_path: HashMap<usize, PathBuf> = HashMap::new();
        row_to_path.insert(0, file0);
        row_to_path.insert(2, file2);

        let values = fill_binary_chunk(0, 3, &row_to_path, false, "rgb");

        assert_eq!(
            values,
            vec![Some(b"AAA".to_vec()), None, Some(b"CCC".to_vec())]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
