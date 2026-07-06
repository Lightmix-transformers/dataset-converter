use std::fs::File;

use anyhow::{Context, Result};
use polars::prelude::*;

pub fn write_chunk_arrow(
    file: &mut File,
    compression: IpcCompression,
    chunk: &mut DataFrame,
    path: &str,
) -> Result<(), anyhow::Error> {
    IpcWriter::new(file)
        .with_compression(Some(compression))
        .finish(chunk)
        .with_context(|| format!("Failed to write parquet '{}'", path))
}
