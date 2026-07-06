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
    pub fn to_parquet(&self) -> anyhow::Result<Option<ParquetCompression>> {
        match self {
            Self::Zstd => Ok(Some(ParquetCompression::Zstd(None))),
            Self::Gzip => Ok(Some(ParquetCompression::Gzip(None))),
            Self::Snappy => Ok(Some(ParquetCompression::Snappy)),
            Self::None => Ok(Some(ParquetCompression::Uncompressed)),
            _ => Err(anyhow::anyhow!("Wrong compression format for parquet")),
        }
    }

    pub fn to_arrow(&self) -> anyhow::Result<Option<IpcCompression>> {
        match self {
            Self::Lz4 => Ok(Some(IpcCompression::LZ4)),
            Self::None => Ok(None),
            _ => Err(anyhow::anyhow!("Wrong compression format for arrow")),
        }
    }

    pub fn formats() -> [&'static str; 4] {
        ["zstd", "gzip", "snappy", "none"]
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
            write_chunk_parquet(
                file,
                compression.to_parquet().unwrap().unwrap(),
                chunk,
                path,
            )
            .unwrap();
        }
        DatasetFormat::Arrow => {
            let arrow_compression = compression.to_arrow().unwrap();
            if arrow_compression.is_some() {
                write_chunk_arrow(file, arrow_compression, chunk, path).unwrap();
            } else {
                write_chunk_arrow(file, None, chunk, path).unwrap();
            }
        }
        _ => unimplemented!(),
    }
}
