pub mod downloader;
pub mod extractor;
pub mod merger;
pub mod schema;

pub use extractor::extract_and_write_incremental;
