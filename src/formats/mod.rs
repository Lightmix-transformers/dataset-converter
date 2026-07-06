use std::fs::File;

use polars::{
    frame::DataFrame,
    io::{ipc::IpcCompression, parquet::write::ParquetCompression},
};

use crate::formats::{arrow_io::write_chunk_arrow, parquet_io::write_chunk_parquet};

pub mod arrow_io;
pub mod parquet_io;

#[derive(Clone)]
pub enum DatasetFormat {
    Parquet,
    Arrow,
}

impl From<&str> for DatasetFormat {
    fn from(format: &str) -> Self {
        match format {
            "parquet" => DatasetFormat::Parquet,
            "arrow" => DatasetFormat::Arrow,
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone)]
pub enum Compression {
    Zstd,
    Gzip,
    Snappy,
    Lz4,
    None,
}

impl Compression {
    pub fn to_parquet(&self) -> anyhow::Result<ParquetCompression> {
        match self {
            Self::Zstd => Ok(ParquetCompression::Zstd(None)),
            Self::Gzip => Ok(ParquetCompression::Gzip(None)),
            Self::Snappy => Ok(ParquetCompression::Snappy),
            Self::None => Ok(ParquetCompression::Uncompressed),
            _ => Err(anyhow::anyhow!("Wrong compression format for parquet")),
        }
    }

    pub fn to_arrow(&self) -> anyhow::Result<IpcCompression> {
        match self {
            Self::Lz4 => Ok(IpcCompression::LZ4),
            _ => Err(anyhow::anyhow!("Wrong compression format for arrow")),
        }
    }
}

impl From<&str> for Compression {
    fn from(format: &str) -> Self {
        match format {
            "zstd" => Self::Zstd,
            "gzip" => Self::Gzip,
            "snappy" => Self::Snappy,
            "none" => Self::None,
            _ => unimplemented!(),
        }
    }
}

pub fn write_dataset(
    format: DatasetFormat,
    file: File,
    compression: Compression,
    chunk: &mut DataFrame,
    path: &str,
) {
    match format {
        DatasetFormat::Parquet => {
            write_chunk_parquet(file, compression.to_parquet().unwrap(), chunk, path).unwrap();
        }
        DatasetFormat::Arrow => {
            write_chunk_arrow(file, compression.to_arrow().unwrap(), chunk, path).unwrap();
        }
        _ => unimplemented!(),
    }
}
