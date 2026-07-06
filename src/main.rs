mod cli;
mod converter;
mod formats;

use anyhow::{Context, Result};
use clap::Parser;
use converter::{
    downloader::fetch_source_files, extract_and_write_incremental, schema::DatasetSchema,
};
use std::path::{Path, PathBuf};

use crate::formats::Compression;

fn main() -> Result<()> {
    let cli = cli::Cli::parse();

    match cli.command {
        cli::Commands::Convert(args) => run_convert(&args),
        cli::Commands::ListSchemas => list_schemas(),
    }
}

/// Resolve a schema path from user input (supports bundled names and file paths).
fn resolve_schema_path(name: &str) -> Result<PathBuf> {
    if Path::new(name).exists() || name.ends_with(".yaml") || name.ends_with(".yml") {
        Ok(PathBuf::from(name))
    } else {
        let bundled = format!("schemas/{}.yaml", name);
        if Path::new(&bundled).exists() {
            Ok(PathBuf::from(&bundled))
        } else {
            Err(anyhow::anyhow!(
                "Schema '{}' not found. Try: dataset-converter list-schemas",
                name
            ))
        }
    }
}

fn run_convert(args: &cli::ConvertArgs) -> Result<()> {
    println!("Converting {} dataset...", args.input);
    println!("  Source: {}", args.source);
    println!("  Output: {}", args.output);
    println!("  Format: {}", args.format);
    println!("  Chunk size: {}", args.chunk_size);
    if args.decode_images {
        println!("  Decode images: {} mode", args.image_mode);
    }

    let mut schema = load_schema(&args.schema, &args.input)?;
    if args.source == "local" {
        schema.source.path = Some(args.input.clone());
    }

    if args.decode_images {
        for field in &mut schema.fields {
            if field.dtype == "binary" {
                field.decode = true;
                field.image_mode = args.image_mode.clone();
            }
        }
    }

    let issues = schema.validate();
    if !issues.is_empty() {
        eprintln!("Schema validation issues:");
        for issue in &issues {
            eprintln!("  - {}", issue);
        }
        return Err(anyhow::anyhow!("Schema validation failed"));
    }

    let files = fetch_source_files(&schema)?;
    println!("Found {} source file(s)", files.len());

    let compression = Compression::from(schema.output.compression.algorithm.as_str());

    let mut processed_paths = std::collections::HashSet::new();

    println!("Using incremental extraction for binary fields...");
    let output_path = match args.format.as_str() {
        "parquet" => format!("{}/output.parquet", args.output.trim_end_matches('/')),
        _ => format!("{}/output.arrow", args.output.trim_end_matches('/')),
    };

    for (_, path) in &files {
        if !processed_paths.insert(path.clone()) {
            continue;
        }
        println!("Extracting from '{}'", path.display());
        let path_str = path
            .to_str()
            .context(format!("Invalid path: {}", path.display()))?;

        extract_and_write_incremental(
            args.format.as_str().into(),
            path_str,
            &schema,
            &output_path,
            args.chunk_size,
            compression.clone(),
        )
        .unwrap();
    }

    let output_path = match args.format.as_str() {
        "parquet" => format!("{}/output.parquet", args.output.trim_end_matches('/')),
        _ => format!("{}/output.arrow", args.output.trim_end_matches('/')),
    };
    println!("Done! Output written to '{}'", output_path);
    Ok(())
}

fn list_schemas() -> Result<()> {
    let schema_dirs = [
        PathBuf::from("schemas"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schemas"),
    ];

    let mut schemas = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for dir in &schema_dirs {
        if !dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .map(|e| e == "yaml" || e == "yml")
                .unwrap_or(false)
                && let Some(name) = path.file_stem().map(|s| s.to_string_lossy().into_owned())
                && seen.insert(name.clone())
            {
                schemas.push((name, path.display().to_string()));
            }
        }
    }

    schemas.sort_by(|a, b| a.0.cmp(&b.0));
    println!("Available schemas:");
    for (name, _) in &schemas {
        println!("  - {}", name);
    }
    println!("\nUsage: dataset-converter preview <schema-name> [--input <file>]");
    Ok(())
}

fn load_schema(schema_arg: &Option<String>, input: &str) -> Result<DatasetSchema> {
    match schema_arg {
        Some(path) => {
            let resolved = resolve_schema_path(path)?;
            DatasetSchema::from_path(resolved.to_str().context("Invalid schema path")?)
        }
        None => {
            // Try to detect from input path
            let path_buf = PathBuf::from(input);
            let stem = path_buf
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("coco");

            let resolved = resolve_schema_path(stem)?;
            DatasetSchema::from_path(resolved.to_str().context("Invalid schema path")?)
        }
    }
}
