# Data Source and Provenance

## Dataset

Store Sales - Time Series Forecasting

## Source

The raw files were obtained from the official Kaggle competition
"Store Sales - Time Series Forecasting".

The dataset represents grocery sales from Corporacion Favorita stores
located in Ecuador.

## Access Date

2026-06-23

## Project Use

This project uses the data to develop and evaluate:

- retail demand forecasting models;
- rolling time-series backtesting;
- promotion and holiday effect analysis;
- replenishment decision-support simulations.

## Data Integrity

The project records:

- required file names;
- expected schemas;
- row counts;
- file sizes;
- SHA-256 checksums.

Validation outputs are stored in:

- `reports/data/raw_data_manifest.csv`
- `reports/data/raw_data_validation.json`

## Repository Policy

Raw competition files are not redistributed through this repository.
Users must obtain the data from the original Kaggle competition source.

## Important Limitations

The dataset represents a specific grocery retailer and country.

Results should not automatically be generalised to:

- other countries;
- non-grocery retailers;
- current Favorita operations;
- businesses with different promotion or supply-chain processes.

The original dataset does not provide complete operational inventory
information such as current stock, supplier lead times, ordering costs,
holding costs, or minimum order quantities.

Therefore, replenishment outputs in this project are decision-support
simulations based on clearly stated user assumptions. They are not actual
Favorita inventory recommendations.
