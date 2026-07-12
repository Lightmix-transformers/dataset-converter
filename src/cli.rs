use clap::{Parser, Subcommand};

/// Dataset converter tool - convert datasets between formats.
#[derive(Parser)]
#[command(name = "dataset-converter")]
#[command(version = "0.0.1")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Convert a dataset to parquet/arrow format.
    Convert(ConvertArgs),

    /// List available bundled schemas.
    ListSchemas,
}

#[derive(Debug, clap::Args)]
pub struct ConvertArgs {
    /// Source type: `hf` for HuggingFace datasets, `local` for local files.
    pub source: String,

    /// Input path or HF dataset identifier (e.g., `cocodataset/coco_2017`).
    pub input: String,

    /// Output directory for converted files.
    #[clap(short, long)]
    pub output: String,

    /// Schema file to use. Defaults to bundled schema if format is detected.
    #[clap(long)]
    pub schema: Option<String>,

    /// Output format: `parquet` or `arrow`.
    #[clap(long, default_value = "parquet")]
    pub format: String,

    /// Chunk size for incremental writing (default: 10000).
    #[clap(long, default_value_t = 10000)]
    pub chunk_size: usize,

    /// Decode images to pixel arrays instead of storing raw bytes.
    #[clap(long)]
    pub decode_images: bool,

    /// Image mode for decoding: `grayscale` or `rgb` (default: grayscale).
    #[clap(long, default_value = "grayscale")]
    pub image_mode: String,

    /// Preserve dataset splits — write separate files per split.
    #[clap(long)]
    pub preserve_splits: bool,
}
