//! Runtime conformance for the checked Mobius workflow packages.
//!
//! This file is copied into `onnx-genai-engine/tests` by Mobius CI so every
//! package is executed by the authoritative ONNX GenAI workflow runtime.

#![allow(clippy::field_reassign_with_default)]

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest, pipeline::WorkflowOutputRole,
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;

fn root(name: &str) -> anyhow::Result<PathBuf> {
    let root = std::env::var_os("MOBIUS_WORKFLOW_CONFORMANCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/onnx_genai_workflows")
        });
    Ok(root.join(name))
}

fn options(max_new_tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::default();
    options.max_new_tokens = max_new_tokens;
    options.seed = Some(7);
    options
}

fn decoder_batch_request(
    input_ids: &[i64],
    batch: i64,
    sequence: i64,
    prompt_lengths: &[i64],
    active: &[bool],
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let bool_bytes = active.iter().map(|value| u8::from(*value)).collect();
    let zeros = vec![0_i64; usize::try_from(batch)?];
    let ones = vec![1_i64; usize::try_from(batch)?];
    let slot_ids = (0..batch).collect::<Vec<_>>();
    let row_ids = (100..100 + batch).collect::<Vec<_>>();
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![0]),
        options: options(max_new_tokens),
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(input_ids, &[batch, sequence])?,
    )
    .with_input(
        "request.prompt_lengths",
        Value::from_slice_i64(prompt_lengths, &[batch])?,
    )
    .with_input(
        "package.active",
        Value::from_raw_bytes(bool_bytes, &[batch], DataType::Bool)?,
    )
    .with_input(
        "package.not_done",
        Value::from_raw_bytes(vec![0; usize::try_from(batch)?], &[batch], DataType::Bool)?,
    )
    .with_input("package.one_token", Value::from_slice_i64(&ones, &[batch])?)
    .with_input(
        "package.slot_ids",
        Value::from_slice_i64(&slot_ids, &[batch])?,
    )
    .with_input(
        "request.row_ids",
        Value::from_slice_i64(&row_ids, &[batch])?,
    )
    .with_input(
        "package.cache_lengths",
        Value::from_slice_i64(&zeros, &[batch])?,
    )
    .with_input(
        "package.zero_batch",
        Value::from_slice_i64(&zeros, &[batch])?,
    ))
}

#[test]
fn mobius_decoder_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("decoder")?, EngineConfig::default())?;
    let output = engine.run_pipeline_outputs(
        PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![4, 5]),
            options: options(3),
        })
        .with_input("request.row_ids", Value::from_slice_i64(&[0], &[1])?),
    )?;
    assert_eq!(
        engine
            .structured_output_for_role(&output, WorkflowOutputRole::Tokens)
            .expect("decoder must emit tokens")
            .to_vec_i64()?
            .len(),
        3
    );
    Ok(())
}

#[test]
fn mobius_decoder_rows_match_independent_runs_and_dynamic_batch_replay() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("decoder")?, EngineConfig::default())?;
    let generated = engine.generate_with_pipeline_request(decoder_batch_request(
        &[4, 5],
        1,
        2,
        &[2],
        &[true],
        3,
    )?)?;
    assert_eq!(generated.token_ids.len(), 3);

    let first = decoder_batch_request(&[4, 5], 1, 2, &[2], &[true], 3)?;
    let first_output = engine.run_pipeline_outputs(first)?;
    let first_tokens = engine
        .structured_output_for_role(&first_output, WorkflowOutputRole::Tokens)
        .expect("batch-one decoder must emit tokens")
        .to_vec_i64()?;

    let multi_row_error = engine
        .generate_with_pipeline_request(decoder_batch_request(
            &[4, 5, 6, 0],
            2,
            2,
            &[2, 1],
            &[true, true],
            3,
        )?)
        .expect_err("generate must not flatten multiple semantic rows");
    assert!(
        multi_row_error
            .to_string()
            .contains("multi-row ragged output")
    );

    let batched = decoder_batch_request(&[4, 5, 6, 0], 2, 2, &[2, 1], &[true, true], 3)?;
    let batched_output = engine.run_pipeline_outputs(batched)?;
    let rows = engine.output_rows_for_role(&batched_output, WorkflowOutputRole::Tokens);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 100);
    assert_eq!(rows[0].1.to_vec_i64()?, first_tokens);

    let mut independent = Engine::from_pipeline_dir(&root("decoder")?, EngineConfig::default())?;
    let second = decoder_batch_request(&[6, 0], 1, 2, &[1], &[true], 3)?;
    let second_output = independent.run_pipeline_outputs(second)?;
    let second_tokens = independent
        .structured_output_for_role(&second_output, WorkflowOutputRole::Tokens)
        .expect("independent second row must emit tokens")
        .to_vec_i64()?;
    assert_eq!(rows[1].0, 101);
    assert_eq!(rows[1].1.to_vec_i64()?, second_tokens);

    let inactive = decoder_batch_request(&[4, 5, 6, 0], 2, 2, &[2, 1], &[true, false], 3)?;
    let inactive_output = engine.run_pipeline_outputs(inactive)?;
    let inactive_rows = engine.output_rows_for_role(&inactive_output, WorkflowOutputRole::Tokens);
    assert_eq!(inactive_rows.len(), 1);
    assert_eq!(inactive_rows[0].0, 100);
    assert_eq!(inactive_rows[0].1.to_vec_i64()?, first_tokens);

    let first_inactive = decoder_batch_request(&[4, 5, 6, 0], 2, 2, &[2, 1], &[false, true], 3)?;
    let first_inactive_output = engine.run_pipeline_outputs(first_inactive)?;
    let first_inactive_rows =
        engine.output_rows_for_role(&first_inactive_output, WorkflowOutputRole::Tokens);
    assert_eq!(first_inactive_rows.len(), 1);
    assert_eq!(first_inactive_rows[0].0, 101);
    assert_eq!(first_inactive_rows[0].1.to_vec_i64()?, second_tokens);
    assert_eq!(
        engine
            .structured_output_for_role(&first_inactive_output, WorkflowOutputRole::Tokens)
            .expect("semantic lookup must return the first emitted row")
            .to_vec_i64()?,
        second_tokens
    );

    let replay = decoder_batch_request(&[4, 5], 1, 2, &[2], &[true], 3)?;
    let replay_output = engine.run_pipeline_outputs(replay)?;
    assert_eq!(
        engine
            .structured_output_for_role(&replay_output, WorkflowOutputRole::Tokens)
            .expect("batch-one replay must emit tokens")
            .to_vec_i64()?,
        first_tokens
    );
    Ok(())
}

#[test]
fn mobius_vlm_workflow_executes_complete_image_path() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("vlm")?, EngineConfig::default())?;
    let png = vec![
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2,
        0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 0, 0,
        3, 1, 1, 0, 201, 254, 146, 239, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    let png_len = i64::try_from(png.len())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5]),
        options: options(2),
    })
    .with_input(
        "request.image",
        Value::from_raw_bytes(png, &[png_len], DataType::Uint8)?,
    )
    .with_input("request.row_ids", Value::from_slice_i64(&[0], &[1])?);
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(
        engine
            .structured_output_for_role(&output, WorkflowOutputRole::Tokens)
            .expect("VLM must emit tokens")
            .shape(),
        [1, 2]
    );
    Ok(())
}

#[test]
fn mobius_euler_diffusion_workflow_executes_complete_path() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("diffusion")?, EngineConfig::default())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: options(2),
    })
    .with_input("latent", Value::from_slice_f32(&[1.0; 64], &[1, 4, 4, 4])?);
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["image"].shape(), [1, 3, 4, 4]);
    assert!(
        output["image"]
            .to_vec_f32()?
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn mobius_masked_diffusion_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("masked")?, EngineConfig::default())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![0, 0]),
        options: options(3),
    })
    .with_input(
        "masked_positions",
        Value::from_raw_bytes(vec![1, 0], &[1, 2], DataType::Bool)?,
    )
    .with_input("rng_offset", Value::from_slice_i64(&[0], &[1])?);
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["tokens"].shape(), [1, 2]);
    Ok(())
}

#[test]
fn mobius_codec_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("codec")?, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "request.waveform",
                Value::from_slice_f32(&[0.25, -0.5], &[1, 1, 2])?,
            );
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["waveform"].to_vec_f32()?, [0.25, -0.5]);
    Ok(())
}

#[test]
fn mobius_tts_workflow_executes_real_producer_graphs() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("tts")?, EngineConfig::default())?;
    let output = engine.run_pipeline_outputs(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: options(1),
    }))?;
    assert_eq!(output["waveform"].shape()[..2], [1, 1]);
    assert!(!output["waveform"].to_vec_f32()?.is_empty());
    Ok(())
}

#[test]
fn mobius_speculative_workflow_executes_rejection_and_correction() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("speculative")?, EngineConfig::default())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2, 3, 4]),
        options: options(1),
    })
    .with_input("serving.slot_ids", Value::from_slice_i64(&[0], &[1])?)
    .with_input("serving.row_ids", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "verifier.past_key_values.0.key",
        Value::from_slice_f32(&[], &[1, 2, 0, 8])?,
    )
    .with_input("grammar.initial_state", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "grammar.transition_table",
        Value::from_slice_i64(&[0; 32], &[1, 32])?,
    )
    .with_input("adaptive.current_k", Value::from_slice_i64(&[4], &[1])?)
    .with_input(
        "adaptive.estimates",
        Value::from_slice_f32(&[0.0; 24], &[1, 24])?,
    )
    .with_input("telemetry.draft_ms", Value::from_slice_f32(&[1.0], &[1])?)
    .with_input("telemetry.target_ms", Value::from_slice_f32(&[1.0], &[1])?);
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["tokens.row.0"].to_vec_i64()?, [1, 31]);
    Ok(())
}
