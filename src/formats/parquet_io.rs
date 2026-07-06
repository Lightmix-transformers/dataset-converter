use std::fs::File;

use anyhow::{Context, Result};
use polars::prelude::*;

pub fn write_chunk_parquet(
    file: File,
    compression: ParquetCompression,
    chunk: &mut DataFrame,
    path: &str,
) -> Result<u64, anyhow::Error> {
    ParquetWriter::new(file.try_clone().unwrap())
        .with_compression(compression)
        .finish(chunk)
        .with_context(|| format!("Failed to write parquet '{}'", path))
}
