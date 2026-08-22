use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai_server::{AppState, app};
use serde_json::json;
use std::path::PathBuf;
use tower::ServiceExt;

fn package_dir() -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(std::env::var("MOBIUS_WORKFLOW_CONFORMANCE_DIR")?).join("hierarchical_audio"))
}

#[tokio::test]
async fn mobius_hierarchical_audio_accepts_raw_speech_request() -> anyhow::Result<()> {
    let state = AppState::load(&package_dir()?, Some("hierarchical-audio".to_string()))?;
    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "hierarchical-audio",
                        "input": "[Verse] Hello",
                        "instructions": "Music",
                        "response_format": "wav",
                        "stream": false,
                        "max_output_units": 3
                    })
                    .to_string(),
                ))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(content_type, "audio/wav");
    let wav = to_bytes(response.into_body(), usize::MAX).await?;
    assert!(wav.len() > 44);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
    assert_eq!(
        u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
        32000
    );
    assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16);
    Ok(())
}
