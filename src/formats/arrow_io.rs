use std::fs::File;

use anyhow::{Context, Result};
use polars::prelude::*;

pub fn write_chunk_arrow(
    file: File,
    compression: Option<IpcCompression>,
    chunk: &mut DataFrame,
    path: &str,
) -> Result<(), anyhow::Error> {
    match compression.is_some() {
        true => IpcWriter::new(file)
            .with_compression(compression)
            .finish(chunk)
            .with_context(|| format!("Failed to write parquet '{}'", path)),
        false => IpcWriter::new(file)
            .finish(chunk)
            .with_context(|| format!("Failed to write parquet '{}'", path)),
    }
}
