use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Top-level dataset schema configuration loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSchema {
    /// Human-readable name for this dataset format.
    pub name: String,
    /// Schema version for compatibility tracking.
    #[serde(default = "default_version")]
    pub version: String,
    /// Source configuration (HF dataset or local path).
    pub source: SourceConfig,
    /// Input files and their entity mappings.
    pub files: Vec<FileConfig>,
    /// Field extraction rules with JSONPath selectors.
    pub fields: Vec<FieldConfig>,
    /// Join configuration for multi-entity denormalization.
    #[serde(default)]
    pub joins: Vec<JoinConfig>,
    /// Output format settings.
    #[serde(default)]
    pub output: OutputConfig,
}

fn default_version() -> String {
    "1.0".to_string()
}

/// Source origin for dataset files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source type: `hf_dataset` or `local`.
    #[serde(rename = "type")]
    pub source_type: String,
    /// For HF datasets: `<owner>/<repo>` identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// For local sources: base directory path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// File mapping within a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileConfig {
    /// Relative path or glob pattern for input files.
    pub path: String,
    /// Logical entity name (e.g., `annotations`, `images`).
    #[serde(default = "default_entity")]
    pub entity: String,
}

fn default_entity() -> String {
    "data".to_string()
}


/// Field extraction rule with JSONPath selector, column name, or file path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfig {
    /// Output column name in the resulting parquet table.
    pub name: String,
    /// JSONPath expression to extract this field from JSON data. Mutually exclusive with `column_name` and `source`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jsonpath: Option<String>,
    /// Column name for CSV data extraction. Alternative to `jsonpath` for tabular sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column_name: Option<String>,
    /// Direct source path/glob for binary data (images). Mutually exclusive with `jsonpath` and `column_name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Target data type for the extracted value.
    #[serde(default = "default_type")]
    pub dtype: String,
    /// Source entity this field belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    /// Decode binary image data to pixel arrays. Applies to `binary` dtype fields only.
    #[serde(default)]
    pub decode: bool,
    /// Image mode for decoding: `grayscale` or `rgb`. Only applies when `decode: true`.
    #[serde(default = "default_image_mode")]
    pub image_mode: String,
}

fn default_image_mode() -> String {
    "grayscale".to_string()
}

fn default_type() -> String {
    "string".to_string()
}

/// Join rule for denormalizing multiple entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinConfig {
    /// Left side join key (e.g., `annotations.image_id`).
    pub left_field: String,
    /// Right side join key (e.g., `images.id`).
    pub right_field: String,
    /// Join strategy: `left`, `inner`, `right`, `outer`.
    #[serde(default = "default_strategy")]
    pub strategy: String,
}

fn default_strategy() -> String {
    "left".to_string()
}

/// Output format and processing settings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    /// Output format: `parquet` or `arrow`.
    #[serde(default = "default_format")]
    pub format: String,
    /// Rows per chunk for incremental writing.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Image compression settings.
    #[serde(default)]
    pub compression: CompressionConfig,
}

/// Image compression configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompressionConfig {
    /// Compress images in parquet by default.
    #[serde(default = "default_compress")]
    pub compress: bool,
    /// Compression algorithm: `zstd` or `zlib`.
    #[serde(default = "default_algo")]
    pub algorithm: String,
}

fn default_compress() -> bool {
    true
}

fn default_algo() -> String {
    "zstd".into()
}

fn default_format() -> String {
    "parquet".into()
}

fn default_chunk_size() -> usize {
    10_000
}

impl DatasetSchema {
    /// Load a schema from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let schema: DatasetSchema = serde_yaml::from_str(yaml)?;
        Ok(schema)
    }

    /// Load a schema from a file path.
    pub fn from_path(path: &str) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_yaml(&contents)
    }

    /// Validate schema structure and report issues.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.name.is_empty() {
            issues.push("Schema name cannot be empty".into());
        }

        if self.files.is_empty() {
            issues.push("At least one file must be specified".into());
        }

        if self.fields.is_empty() {
            issues.push("At least one field must be defined".into());
        }

        let valid_types = ["u32", "u64", "i32", "i64", "f32", "f64", "string", "bool", "binary"];

        let valid_modes = ["grayscale", "gray", "rgb"];
        for field in &self.fields {
            if !valid_types.contains(&field.dtype.as_str()) {
                issues.push(format!(
                    "Field '{}': invalid type '{}'. Must be one of: {:?}",
                    field.name, field.dtype, valid_types
                ));
            }
            let has_source =
                field.jsonpath.is_some() || field.column_name.is_some() || field.source.is_some();
            if !has_source {
                issues.push(format!(
                    "Field '{}': must specify 'jsonpath', 'column_name', or 'source'",
                    field.name
                ));
            }

            // Validate decode/image_mode settings for binary fields
            if field.decode && field.dtype != "binary" {
                issues.push(format!(
                    "Field '{}': 'decode' can only be used with 'binary' dtype",
                    field.name
                ));
            }
            if field.decode && !valid_modes.contains(&field.image_mode.as_str()) {
                issues.push(format!(
                    "Field '{}': invalid image_mode '{}'. Must be one of: {:?}",
                    field.name, field.image_mode, valid_modes
                ));
            }
        }

        let valid_strategies = ["left", "inner", "right", "outer"];
        for join in &self.joins {
            if !valid_strategies.contains(&join.strategy.as_str()) {
                issues.push(format!(
                    "Join '{} -> {}': invalid strategy '{}'. Must be one of: {:?}",
                    join.left_field, join.right_field, join.strategy, valid_strategies
                ));
            }
        }

        let valid_formats = ["parquet", "arrow"];
        if !valid_formats.contains(&self.output.format.as_str()) {
            issues.push(format!(
                "Output format '{}': invalid. Must be one of: {:?}",
                self.output.format, valid_formats
            ));
        }

        let valid_algos = ["zstd", "zlib"];
        if !valid_algos.contains(&self.output.compression.algorithm.as_str()) {
            issues.push(format!(
                "Compression algorithm '{}': invalid. Must be one of: {:?}",
                self.output.compression.algorithm, valid_algos
            ));
        }

        issues
    }
}
