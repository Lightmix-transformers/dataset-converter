use anyhow::{Context, Result};
use image::ImageReader;
use jsonpath_lib::JsonPathError;
use polars::prelude::*;
use serde_json::Value;
use std::fs::{File, OpenOptions, read, read_to_string};
use std::io::Cursor;
use std::path::PathBuf;
use std::{collections::HashMap, path::Path};

use crate::converter::merger::merge_entities;
use crate::formats::OutputFormat;
use crate::formats::arrow_io::write_chunk_arrow;
use crate::formats::parquet_io::write_chunk_parquet;

use super::schema::{DatasetSchema, FieldConfig};

pub fn extract_and_write_incremental(
    format: OutputFormat,
    path: &str,
    schema: &DatasetSchema,
    output_path: &str,
    chunk_size: usize,
    compression: ParquetCompression,
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
        write_chunk_parquet(file, compression, &mut merged, output_path).unwrap();
        Ok(())
    }
}

fn extract_csv_incremental(
    format: OutputFormat,
    path: &str,
    entity_fields: &HashMap<&str, Vec<&FieldConfig>>,
    output_path: &str,
    chunk_size: usize,
    compression: ParquetCompression,
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
            compression,
        )?;
    }

    Ok(())
}

/// Picks the path column: prefer a field name containing "path"/"file",
/// else the first String-typed column. Validated against the real schema.
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

/// Indexes a path column by both its raw value and its base_dir-prefixed
/// form, to match however a field's glob-resolved paths turn out to be shaped.
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

fn extract_csv_with_binary_chunked(
    format: OutputFormat,
    path: &str,
    entity: &str,
    fields: &[&FieldConfig],
    output_path: &str,
    chunk_size: usize,
    compression: ParquetCompression,
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

    let mut file =
        File::create(output_path).with_context(|| format!("Failed to create '{}'", output_path))?;
    for start in (0..total_rows).step_by(chunk_size) {
        let end = (start + chunk_size).min(total_rows);
        println!("    Chunk {}-{}...", start, end);

        let mut result_chunk = match renamed_df {
            Some(ref renamed_df) => renamed_df.slice(start as i64, end - start),
            None => DataFrame::empty(),
        };

        for (bidx, (field, _)) in binary_fields.iter().enumerate() {
            let row_to_path = &row_to_path_by_field[bidx];
            let binary_values =
                fill_binary_chunk(start, end, row_to_path, field.decode, &field.image_mode);
            result_chunk.with_column(Column::new(field.name.as_str().into(), &binary_values))?;
        }

        println!("    Writing {} rows...", result_chunk.height());
        match format {
            OutputFormat::Parquet => {
                write_chunk_parquet(
                    file.try_clone().unwrap(),
                    compression,
                    &mut result_chunk,
                    output_path,
                )
                .unwrap();
            }
            _ => {
                write_chunk_arrow(
                    &mut file,
                    IpcCompression::LZ4,
                    &mut result_chunk,
                    output_path,
                )
                .unwrap();
            }
        }

        drop(result_chunk);
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

fn query_path(json: &Value, path: &str) -> Result<Vec<Option<Value>>> {
    let results = jsonpath_lib::select(json, path)
        .map_err(|e: JsonPathError| anyhow::anyhow!("JSONPath '{}' error: {}", path, e))?;

    Ok(results.into_iter().map(|v| Some(v.clone())).collect())
}

/// `pattern` supports comma-separated globs, e.g. "train/**/*.JPEG,val/**/*.JPEG".
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

/// Falls back to explicit directory walking for "**" patterns.
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

fn resolve_path(path: &str, base_dir: &str) -> String {
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("{}/{}", base_dir, path)
    }
}

fn resolve_recursive(pattern: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(paths) = glob::glob(pattern) {
        for path in paths.filter_map(|r| r.ok()) {
            results.push(path);
        }
    }
    results
}

fn collect_numeric<T>(
    values: &[Option<Value>],
    f: impl Fn(&serde_json::Number) -> Option<T>,
) -> Vec<T> {
    values
        .iter()
        .filter_map(|v| v.as_ref()?.as_number().and_then(&f))
        .collect()
}

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

fn broadcast_column(col: &Column, target_len: usize) -> Result<Column> {
    let series = col
        .as_series()
        .ok_or_else(|| anyhow::anyhow!("Column has no series"))?;
    Ok(Column::from(series.new_from_index(0, target_len)))
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
