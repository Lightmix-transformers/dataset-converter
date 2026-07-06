use anyhow::{Context, Result};
use polars::prelude::*;
use std::collections::HashMap;

use super::schema::DatasetSchema;

pub fn merge_entities(
    entity_dfs: Vec<(String, LazyFrame)>,
    schema: &DatasetSchema,
) -> Result<LazyFrame> {
    if entity_dfs.is_empty() {
        return Ok(DataFrame::empty().lazy());
    }

    let mut lf_map: HashMap<String, LazyFrame> = entity_dfs.into_iter().collect();

    // Start with the first entity's LazyFrame
    let first_entity = schema
        .files
        .first()
        .map(|f| f.entity.clone())
        .unwrap_or("data".to_string());
    let mut result_lf = lf_map.remove(&first_entity).context(format!(
        "Entity '{}' not found in extracted data",
        first_entity
    ))?;

    // Process each join - chain operations lazily without collecting
    for join in &schema.joins {
        let (_left_entity, left_col) = parse_field_ref(&join.left_field)?;
        let (right_entity, right_col) = parse_field_ref(&join.right_field)?;

        // Get the right LazyFrame and rename its columns to avoid conflicts
        let right_lf = lf_map.remove(&right_entity).context(format!(
            "Entity '{}' not found for join '{} -> {}'",
            right_entity, join.left_field, join.right_field
        ))?;

        let renamed_right = rename_columns_lf(&right_lf, &right_entity)?;

        // Find the actual column names in both LazyFrames
        let left_actual_col = find_column_in_lf(&result_lf, &left_col)
            .context(format!("Column '{}' not found in left LazyFrame", left_col))?;
        let right_actual_col =
            find_column_in_lf(&renamed_right, &format!("{}_{}", right_entity, right_col)).context(
                format!(
                    "Column for '{}' not found in renamed right LazyFrame",
                    right_col
                ),
            )?;

        // Perform the join lazily (no collection until merge is complete)
        result_lf = perform_join(
            &result_lf,
            &renamed_right,
            &left_actual_col,
            &right_actual_col,
            &join.strategy,
        )?;
    }

    Ok(result_lf)
}

fn perform_join(
    left: &LazyFrame,
    right: &LazyFrame,
    left_col: &str,
    right_col: &str,
    strategy: &str,
) -> Result<LazyFrame> {
    let join_type = match strategy {
        "left" => JoinType::Left,
        "inner" => JoinType::Inner,
        "right" => JoinType::Right,
        _ => JoinType::Left,
    };

    Ok(left
        .clone()
        .join_builder()
        .with(right.clone())
        .how(join_type)
        .left_on(vec![col(left_col)])
        .right_on(vec![col(right_col)])
        .finish())
}

fn rename_columns_lf(lf: &LazyFrame, entity: &str) -> Result<LazyFrame> {
    let schema = lf
        .clone()
        .collect_schema()
        .map_err(|e| anyhow::anyhow!("Failed to get schema for rename: {}", e))?;

    // Build rename expressions from the output schema
    let mut exprs: Vec<Expr> = Vec::new();
    for col_name in schema.iter_names() {
        let new_name = format!("{}_{}", entity, col_name);
        exprs.push(col(col_name.as_str()).alias(&new_name));
    }

    if exprs.is_empty() {
        return Ok(lf.clone());
    }

    Ok(lf.clone().select(exprs))
}

fn parse_field_ref(field: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = field.split('.').collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!(
            "Field '{}' must be in format 'entity.column'",
            field
        ));
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

fn find_column_in_lf(lf: &LazyFrame, target: &str) -> Option<String> {
    let schema = match lf.clone().collect_schema() {
        Ok(s) => s,
        Err(_) => return None,
    };
    let cols: Vec<String> = schema.iter_names().map(|n| n.to_string()).collect();
    cols.iter()
        .find(|c| c.as_str() == target || c.ends_with(&format!("_{}", target)))
        .cloned()
}
