# Dataset converter tool to parquet/arrow
This tool was developed in order to unify my experiments with different datasets into a single format - arrow, which is very efficient for reading from the disk. Althought, initially it was made for the arrow format, it supports convertion to parquet with or without compressing as well.

The tool in its initial development stage, so many things may not work well.

## Architecture

It works approximately like that (diagram below). I'm still experimenting with scheme definitions, so changes are expected.
```text
                         CLI args + YAML schema
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Schema Load           │
                    ├──────────────────────────────┤
                    │ • fields                     │
                    │ • entities                   │
                    │ • joins                      │
                    │ • split_strategy             │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Source Fetch          │
                    ├──────────────────────────────┤
                    │ local glob                   │
                    │           or                 │
                    │ HuggingFace download         │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Split Route           │
                    ├──────────────────────────────┤
                    │ none │ file_path │ row_column│
                    └──────────────┬───────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼

        ┌──────────────────────┐      ┌──────────────────────┐
        │         JSON         │      │         CSV          │
        ├──────────────────────┤      ├──────────────────────┤
        │ • JSONPath           │      │ • Column mapping     │
        │ • Binary resolve     │      │ • Binary resolve     │
        │                      │      │   (path → file)      │
        └──────────┬───────────┘      │ • Partition          │
                   │                  └──────────┬───────────┘
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │        Write Output          │
                    ├──────────────────────────────┤
                    │ Parquet                      │
                    │ or Arrow IPC                 │
                    │ (+ compression)              │
                    └──────────────────────────────┘
```

## Usage
To use the tool, specify dataset location, output dir, output format, and optionally compressing and memory handling options. Example for imagenette2 dataset:
```bash
cargo run --release convert --output /storage/experiments-ml/datasets/imagenette2 local /storage/experiments-ml/datasets/raw/imagenette2/ --schema schemas/imagenette.yaml  --decode-images --image-mode rgb --chunk-size 1024 --format parque
```


