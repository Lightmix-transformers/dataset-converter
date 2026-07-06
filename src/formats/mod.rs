pub mod arrow_io;
pub mod parquet_io;

#[derive(Clone)]
pub enum OutputFormat {
    Parquet,
    Arrow,
}

impl From<&str> for OutputFormat {
    fn from(format: &str) -> Self {
        match format {
            "parquet" => OutputFormat::Parquet,
            "arrow" => OutputFormat::Arrow,
            _ => unimplemented!(),
        }
    }
}
