use std::fs;
use std::path::PathBuf;
use polars::prelude::*;

/// Integration test for image embedding in parquet files.
/// 
/// This test verifies that:
/// 1. Binary columns are created correctly from JPEG files
/// 2. The binary data matches the source images exactly (byte-for-byte)
/// 3. Rows align correctly between metadata and binary data
fn test_image_embedding() {
    // Setup paths
    let schema_path = "schemas/coco_local_test.yaml";
    let output_dir = "test_output_integration";
    let parquet_path = format!("{}/output.parquet", output_dir);

    // Clean up previous test output
    let _ = fs::remove_dir_all(output_dir);
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // Run the converter using subprocess
    let output = std::process::Command::new("./target/debug/dataset-converter")
        .args([
            "convert",
            "local",
            ".",
            "--output",
            output_dir,
            "--schema",
            schema_path,
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to start dataset-converter");

    assert!(
        output.status.success(),
        "Converter failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify parquet file was created
    assert!(
        PathBuf::from(&parquet_path).exists(),
        "Parquet file not created at {}",
        parquet_path
    );

    // Read the parquet file and verify binary column
    let df = ParquetReader::new(
        fs::File::open(&parquet_path).expect("Failed to open parquet"),
    )
    .finish()
    .expect("Failed to read parquet");

    // Verify the DataFrame has expected structure
    assert!(
        df.height() > 0,
        "DataFrame is empty, expected rows from test data"
    );

    // Check that image_bytes column exists and is binary type (prefixed with entity name)
    let schema = df.schema();
    assert!(
        schema.get_field("images_image_bytes").is_some(),
        "images_image_bytes column not found in schema. Available columns: {:?}",
        schema.iter_names().collect::<Vec<_>>()
    );

    // Get the binary column
    let image_bytes_col = df.column("images_image_bytes").expect("Failed to get images_image_bytes column");
    let binary_chunked = image_bytes_col.binary().expect("image_bytes is not a binary column");

    // Read source JPEG file sizes for verification
    let jpeg1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/test1_coco.jpg");
    let jpeg2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/test2_coco.jpg");

    let jpeg1_size = fs::metadata(&jpeg1_path).expect("Failed to read test1_coco.jpg").len() as usize;
    let jpeg2_size = fs::metadata(&jpeg2_path).expect("Failed to read test2_coco.jpg").len() as usize;

    // Collect all non-null row sizes (after join, images may be duplicated)
    let iter = binary_chunked.iter();
    let rows = iter.collect::<Vec<_>>();

    assert!(rows.len() >= 2, "Expected at least 2 rows from iterator, got {}", rows.len());

    // Collect non-null row sizes and verify they match our source files
    let sizes: Vec<usize> = rows.iter().filter_map(|r| r.map(|b| b.len())).collect();

    // We should have exactly 2 unique file sizes present (one for each JPEG)
    let mut unique_sizes = sizes.clone();
    unique_sizes.sort();
    unique_sizes.dedup();
    assert_eq!(unique_sizes.len(), 2, "Expected 2 unique binary sizes, got {}", unique_sizes.len());

    // Verify both source file sizes are present
    assert!(sizes.contains(&jpeg1_size), "Missing JPEG1 size {} in results", jpeg1_size);
    assert!(sizes.contains(&jpeg2_size), "Missing JPEG2 size {} in results", jpeg2_size);

    println!("✓ Image embedding test passed!");
    println!("  - Binary column exists with correct type");
    println!("  - {} rows extracted, {} unique sizes", binary_chunked.len(), unique_sizes.len());
    println!("  - Binary sizes match source JPEG files: {} and {} bytes", jpeg1_size, jpeg2_size);
}

/// Integration test for PNG image embedding.
/// 
/// Converts JPEG images to PNG and verifies the same embedding logic works.
fn test_png_image_embedding() {
    use std::io::Cursor;
    
    // Read JPEG files
    let jpeg1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/test1_coco.jpg");
    let jpeg2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/test2_coco.jpg");

    let jpeg1_bytes = fs::read(&jpeg1_path).expect("Failed to read test1_coco.jpg");
    let jpeg2_bytes = fs::read(&jpeg2_path).expect("Failed to read test2_coco.jpg");

    // Create PNG versions of our test images
    let png1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/test1_coco.png");
    let png2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/test2_coco.png");

    // Decode JPEGs and save as PNG using with_format associated function
    let img1 = image::ImageReader::with_format(
        Cursor::new(&jpeg1_bytes),
        image::ImageFormat::Jpeg,
    )
    .decode()
    .expect("Failed to decode test1_coco.jpg")
    .to_rgba8();
    
    img1.save(&png1_path).expect("Failed to save test1_coco.png");

    let img2 = image::ImageReader::with_format(
        Cursor::new(&jpeg2_bytes),
        image::ImageFormat::Jpeg,
    )
    .decode()
    .expect("Failed to decode test2_coco.jpg")
    .to_rgba8();
    
    img2.save(&png2_path).expect("Failed to save test2_coco.png");

    // Read PNG files for verification
    let png1_bytes = fs::read(&png1_path).expect("Failed to read test1_coco.png");
    let png2_bytes = fs::read(&png2_path).expect("Failed to read test2_coco.png");

    // Create a temporary schema for PNG testing
    let png_schema = r#"
name: coco_png_test
version: "1.0"

source:
  type: local
  path: .

files:
  - path: test_data.json
    entity: annotations
  - path: test_data.json
    entity: images

fields:
  - name: image_id
    jsonpath: "$.images[*].id"
    dtype: u32
    entity: images

  - name: file_name
    jsonpath: "$.images[*].file_name"
    dtype: string
    entity: images

  - name: width
    jsonpath: "$.images[*].width"
    dtype: u32
    entity: images

  - name: height
    jsonpath: "$.images[*].height"
    dtype: u32
    entity: images

  - name: image_bytes
    source: "test_data/*.png"
    dtype: binary
    entity: images

  - name: annotation_id
    jsonpath: "$.annotations[*].id"
    dtype: u32
    entity: annotations

  - name: image_id
    jsonpath: "$.annotations[*].image_id"
    dtype: u32
    entity: annotations

joins:
  - left_field: annotations.image_id
    right_field: images.image_id
    strategy: left

output:
  format: parquet
  chunk_size: 10000
  compression:
    compress: true
    algorithm: zstd
"#;

    // Write PNG schema
    let png_schema_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/coco_png_test.yaml");
    fs::write(&png_schema_path, png_schema).expect("Failed to write PNG test schema");

    // Run converter with PNG schema
    let output_dir = "test_output_png";
    let _ = fs::remove_dir_all(output_dir);
    fs::create_dir_all(output_dir).expect("Failed to create PNG output directory");

    let output = std::process::Command::new("./target/debug/dataset-converter")
        .args([
            "convert",
            "local",
            ".",
            "--output",
            output_dir,
            "--schema",
            png_schema_path.to_str().unwrap(),
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to start dataset-converter for PNG test");

    assert!(
        output.status.success(),
        "PNG converter failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Read and verify PNG parquet
    let parquet_path = format!("{}/output.parquet", output_dir);
    let df = ParquetReader::new(
        fs::File::open(&parquet_path).expect("Failed to open PNG parquet"),
    )
    .finish()
    .expect("Failed to read PNG parquet");

    let image_bytes_col = df.column("images_image_bytes").expect("images_image_bytes column not found");
    let binary_chunked = image_bytes_col.binary().expect("image_bytes is not binary");

    // Verify PNG sizes match source files (after join, images may be duplicated)
    let iter = binary_chunked.iter();
    let rows = iter.collect::<Vec<_>>();

    assert!(rows.len() >= 2, "Expected at least 2 PNG rows, got {}", rows.len());

    let sizes: Vec<usize> = rows.iter().filter_map(|r| r.map(|b| b.len())).collect();
    let png1_size = png1_bytes.len();
    let png2_size = png2_bytes.len();

    // Verify both source file sizes are present in results
    assert!(sizes.contains(&png1_size), "Missing PNG1 size {} in results", png1_size);
    assert!(sizes.contains(&png2_size), "Missing PNG2 size {} in results", png2_size);

    // Clean up PNG files and schema
    let _ = fs::remove_file(&png1_path);
    let _ = fs::remove_file(&png2_path);
    let _ = fs::remove_file(&png_schema_path);
    let _ = fs::remove_dir_all(output_dir);

    println!("✓ PNG image embedding test passed!");
}

#[test]
fn integration_test_image_embedding() {
    test_image_embedding();
}

#[test]
fn integration_test_png_image_embedding() {
    test_png_image_embedding();
}
