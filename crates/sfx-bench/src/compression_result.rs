use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncodingMode {
    Independent,
    IntraSequence,
}

impl EncodingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::IntraSequence => "intra-sequence",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionBenchmarkResult {
    pub bench_name: String,
    pub split: String,
    pub ae_operating_points: AeOperatingPoints,
    #[serde(default)]
    pub codec_rd_curves: Vec<CodecRdCurve>,
    #[serde(default)]
    pub matched_quality: Vec<MatchedQualitySummary>,
    #[serde(default)]
    pub metadata: CompressionMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecRdCurve {
    pub codec: String,
    pub encoding_mode: EncodingMode,
    #[serde(default)]
    pub rd_points: Vec<CodecRdPoint>,
    #[serde(default)]
    pub lossless_anchor: Option<LosslessAnchor>,
    #[serde(default)]
    pub caveats: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecRdPoint {
    pub quality_control: String,
    pub rate_bpp: f32,
    #[serde(default)]
    pub modality_summary: Vec<ModalitySummary>,
    pub combined_summary: CombinedSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LosslessAnchor {
    pub rate_bpp: f32,
    #[serde(default)]
    pub modality_summary: Vec<ModalitySummary>,
    pub combined_summary: CombinedSummary,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AeOperatingPoints {
    pub raw_f32: AeOperatingPoint,
    pub quantized: AeOperatingPoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AeOperatingPoint {
    pub encoding_mode: EncodingMode,
    pub rate_bpp: f32,
    #[serde(default)]
    pub modality_summary: Vec<ModalitySummary>,
    pub combined_summary: CombinedSummary,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedQualitySummary {
    pub quality_metric: String,
    pub target_value: f32,
    pub ae_raw_f32: MatchedQualityPoint,
    pub ae_quantized: MatchedQualityPoint,
    #[serde(default)]
    pub codec_points: Vec<CodecMatchedQualityPoint>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecMatchedQualityPoint {
    pub codec: String,
    pub encoding_mode: EncodingMode,
    pub quality_control: String,
    pub point: MatchedQualityPoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedQualityPoint {
    pub encoding_mode: EncodingMode,
    pub rate_bpp: f32,
    #[serde(default)]
    pub modality_summary: Vec<ModalitySummary>,
    pub combined_summary: CombinedSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalitySummary {
    pub modality: String,
    pub mse: f32,
    pub psnr_db: f32,
    #[serde(default)]
    pub ssim: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CombinedSummary {
    pub mse: f32,
    pub psnr_db: f32,
    #[serde(default)]
    pub ssim: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompressionMetadata {
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub generated_by: String,
    #[serde(default)]
    pub caveats: Vec<String>,
    #[serde(default)]
    pub notes: BTreeMap<String, String>,
}

pub fn render_summary_tables_markdown(result: &CompressionBenchmarkResult) -> String {
    let mut markdown = String::new();
    markdown.push_str(&format!("# Compression summary: {}\n\n", result.bench_name));
    markdown.push_str(&format!("Split: `{}`\n\n", result.split));

    markdown.push_str("## AE operating points\n\n");
    markdown
        .push_str("| Point | Encoding mode | Rate (bpp) | Combined PSNR (dB) | Combined MSE |\n");
    markdown.push_str("| --- | --- | ---: | ---: | ---: |\n");
    append_ae_operating_point_row(
        &mut markdown,
        "raw-f32",
        &result.ae_operating_points.raw_f32,
    );
    append_ae_operating_point_row(
        &mut markdown,
        "quantized",
        &result.ae_operating_points.quantized,
    );
    markdown.push('\n');

    markdown.push_str("## AE modality summary\n\n");
    markdown.push_str("| Point | Modality | PSNR (dB) | MSE | SSIM |\n");
    markdown.push_str("| --- | --- | ---: | ---: | ---: |\n");
    append_ae_modality_rows(
        &mut markdown,
        "raw-f32",
        &result.ae_operating_points.raw_f32,
    );
    append_ae_modality_rows(
        &mut markdown,
        "quantized",
        &result.ae_operating_points.quantized,
    );
    markdown.push('\n');

    markdown.push_str("## Codec RD points\n\n");
    if result.codec_rd_curves.is_empty() {
        markdown.push_str("_No codec RD points available._\n\n");
    } else {
        for curve in &result.codec_rd_curves {
            markdown.push_str(&format!(
                "### {} ({})\n\n",
                curve.codec,
                curve.encoding_mode.as_str()
            ));
            markdown
                .push_str("| Quality control | Rate (bpp) | Combined PSNR (dB) | Combined MSE |\n");
            markdown.push_str("| --- | ---: | ---: | ---: |\n");
            for point in &curve.rd_points {
                markdown.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    point.quality_control,
                    fmt_f32(point.rate_bpp),
                    fmt_f32(point.combined_summary.psnr_db),
                    fmt_f32(point.combined_summary.mse)
                ));
            }
            if let Some(lossless_anchor) = &curve.lossless_anchor {
                markdown.push_str(&format!(
                    "| lossless-anchor | {} | {} | {} |\n",
                    fmt_f32(lossless_anchor.rate_bpp),
                    fmt_f32(lossless_anchor.combined_summary.psnr_db),
                    fmt_f32(lossless_anchor.combined_summary.mse)
                ));
            }
            if !curve.caveats.is_empty() {
                markdown.push_str("\n_Curve caveats:_\n");
                for caveat in &curve.caveats {
                    markdown.push_str(&format!("- {caveat}\n"));
                }
            }
            markdown.push('\n');
        }
    }

    markdown.push_str("## Matched-quality summary\n\n");
    if result.matched_quality.is_empty() {
        markdown.push_str("_No matched-quality summary available._\n\n");
    } else {
        markdown.push_str("| Target | Method | Encoding mode | Rate (bpp) | Combined PSNR (dB) | Combined MSE |\n");
        markdown.push_str("| --- | --- | --- | ---: | ---: | ---: |\n");
        for summary in &result.matched_quality {
            let target = format!("{}={:.2}", summary.quality_metric, summary.target_value);
            append_matched_quality_row(&mut markdown, &target, "ae/raw-f32", &summary.ae_raw_f32);
            append_matched_quality_row(
                &mut markdown,
                &target,
                "ae/quantized",
                &summary.ae_quantized,
            );
            for codec_point in &summary.codec_points {
                append_matched_quality_row(
                    &mut markdown,
                    &target,
                    &format!(
                        "codec/{} ({})",
                        codec_point.codec, codec_point.quality_control
                    ),
                    &codec_point.point,
                );
            }
            if !summary.notes.is_empty() {
                markdown.push_str(&format!(
                    "| {} | notes | - | - | - | {} |\n",
                    target,
                    summary.notes.join("; ")
                ));
            }
        }
        markdown.push('\n');
    }

    if !result.metadata.caveats.is_empty() {
        markdown.push_str("## Caveats\n\n");
        for caveat in &result.metadata.caveats {
            markdown.push_str(&format!("- {caveat}\n"));
        }
        markdown.push('\n');
    }

    if !result.metadata.notes.is_empty() {
        markdown.push_str("## Notes\n\n");
        markdown.push_str("| Key | Value |\n");
        markdown.push_str("| --- | --- |\n");
        for (key, value) in &result.metadata.notes {
            markdown.push_str(&format!("| {key} | {value} |\n"));
        }
    }

    markdown
}

fn append_ae_operating_point_row(
    markdown: &mut String,
    point_name: &str,
    point: &AeOperatingPoint,
) {
    markdown.push_str(&format!(
        "| {} | {} | {} | {} | {} |\n",
        point_name,
        point.encoding_mode.as_str(),
        fmt_f32(point.rate_bpp),
        fmt_f32(point.combined_summary.psnr_db),
        fmt_f32(point.combined_summary.mse)
    ));
}

fn append_ae_modality_rows(markdown: &mut String, point_name: &str, point: &AeOperatingPoint) {
    if point.modality_summary.is_empty() {
        markdown.push_str(&format!("| {point_name} | - | - | - | - |\n"));
        return;
    }
    for modality in &point.modality_summary {
        markdown.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            point_name,
            modality.modality,
            fmt_f32(modality.psnr_db),
            fmt_f32(modality.mse),
            modality
                .ssim
                .map(fmt_f32)
                .unwrap_or_else(|| "-".to_string())
        ));
    }
}

fn append_matched_quality_row(
    markdown: &mut String,
    target: &str,
    method: &str,
    point: &MatchedQualityPoint,
) {
    markdown.push_str(&format!(
        "| {} | {} | {} | {} | {} | {} |\n",
        target,
        method,
        point.encoding_mode.as_str(),
        fmt_f32(point.rate_bpp),
        fmt_f32(point.combined_summary.psnr_db),
        fmt_f32(point.combined_summary.mse)
    ));
}

fn fmt_f32(value: f32) -> String {
    format!("{value:.4}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compression_result_serializes_and_deserializes() {
        let expected = sample_result();
        let encoded = serde_json::to_string_pretty(&expected)
            .expect("compression benchmark result should serialize to JSON");

        assert!(encoded.contains("\"raw_f32\""));
        assert!(encoded.contains("\"codec_rd_curves\""));
        assert!(encoded.contains("\"matched_quality\""));

        let decoded: CompressionBenchmarkResult = serde_json::from_str(&encoded)
            .expect("compression benchmark result should deserialize");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn encoding_mode_uses_kebab_case_serialization() {
        let encoded =
            serde_json::to_string(&EncodingMode::IntraSequence).expect("encoding mode serializes");
        assert_eq!(encoded, "\"intra-sequence\"");

        let decoded: EncodingMode = serde_json::from_str(&encoded).expect("encoding mode parses");
        assert_eq!(decoded, EncodingMode::IntraSequence);
    }

    #[test]
    fn deserialization_populates_defaults_for_optional_sections() {
        let raw = json!({
            "bench_name": "tiny",
            "split": "val",
            "ae_operating_points": {
                "raw_f32": {
                    "encoding_mode": "independent",
                    "rate_bpp": 1.0,
                    "combined_summary": { "mse": 0.01, "psnr_db": 20.0 }
                },
                "quantized": {
                    "encoding_mode": "intra-sequence",
                    "rate_bpp": 0.5,
                    "combined_summary": { "mse": 0.02, "psnr_db": 18.0 }
                }
            }
        });

        let decoded: CompressionBenchmarkResult =
            serde_json::from_value(raw).expect("compression benchmark result should deserialize");

        assert!(decoded.codec_rd_curves.is_empty());
        assert!(decoded.matched_quality.is_empty());
        assert_eq!(decoded.metadata, CompressionMetadata::default());
        assert!(
            decoded
                .ae_operating_points
                .raw_f32
                .modality_summary
                .is_empty()
        );
        assert!(
            decoded
                .ae_operating_points
                .quantized
                .modality_summary
                .is_empty()
        );
    }

    #[test]
    fn summary_tables_render_to_markdown() {
        let markdown = render_summary_tables_markdown(&sample_result());

        assert!(markdown.contains("## AE operating points"));
        assert!(markdown.contains("| raw-f32 | independent | 0.8200 | 31.2000 | 0.0045 |"));
        assert!(markdown.contains("| quantized | intra-sequence | 0.4400 | 30.6000 | 0.0055 |"));
        assert!(markdown.contains("### av1 (independent)"));
        assert!(markdown.contains("| crf=28 | 0.6400 | 31.0000 | 0.0048 |"));
        assert!(markdown.contains("| lossless-anchor | 1.1800 | 38.2000 | 0.0002 |"));
        assert!(markdown.contains("## Matched-quality summary"));
        assert!(
            markdown.contains(
                "| psnr-db=31.00 | ae/raw-f32 | independent | 0.8200 | 31.2000 | 0.0045 |"
            )
        );
        assert!(markdown.contains("## Caveats"));
        assert!(markdown.contains("- Validation split uses same recording segments as train."));
        assert!(markdown.contains("| encoder_preset | slow |"));
    }

    #[test]
    fn summary_tables_render_placeholders_for_empty_sections() {
        let markdown = render_summary_tables_markdown(&minimal_result());

        assert!(markdown.contains("| raw-f32 | - | - | - | - |"));
        assert!(markdown.contains("| quantized | - | - | - | - |"));
        assert!(markdown.contains("_No codec RD points available._"));
        assert!(markdown.contains("_No matched-quality summary available._"));
        assert!(!markdown.contains("## Notes"));
    }

    fn minimal_result() -> CompressionBenchmarkResult {
        CompressionBenchmarkResult {
            bench_name: "tiny".to_string(),
            split: "val".to_string(),
            ae_operating_points: AeOperatingPoints {
                raw_f32: AeOperatingPoint {
                    encoding_mode: EncodingMode::Independent,
                    rate_bpp: 1.0,
                    modality_summary: Vec::new(),
                    combined_summary: CombinedSummary {
                        mse: 0.01,
                        psnr_db: 20.0,
                        ssim: None,
                    },
                    notes: Vec::new(),
                },
                quantized: AeOperatingPoint {
                    encoding_mode: EncodingMode::IntraSequence,
                    rate_bpp: 0.5,
                    modality_summary: Vec::new(),
                    combined_summary: CombinedSummary {
                        mse: 0.02,
                        psnr_db: 18.0,
                        ssim: None,
                    },
                    notes: Vec::new(),
                },
            },
            codec_rd_curves: Vec::new(),
            matched_quality: Vec::new(),
            metadata: CompressionMetadata::default(),
        }
    }

    fn sample_result() -> CompressionBenchmarkResult {
        let rgb = ModalitySummary {
            modality: "rgb".to_string(),
            mse: 0.0039,
            psnr_db: 31.9,
            ssim: Some(0.91),
        };
        let range = ModalitySummary {
            modality: "range".to_string(),
            mse: 0.0051,
            psnr_db: 30.6,
            ssim: Some(0.83),
        };
        let combined_raw = CombinedSummary {
            mse: 0.0045,
            psnr_db: 31.2,
            ssim: Some(0.87),
        };
        let combined_quantized = CombinedSummary {
            mse: 0.0055,
            psnr_db: 30.6,
            ssim: Some(0.84),
        };

        CompressionBenchmarkResult {
            bench_name: "waymo-compression".to_string(),
            split: "val".to_string(),
            ae_operating_points: AeOperatingPoints {
                raw_f32: AeOperatingPoint {
                    encoding_mode: EncodingMode::Independent,
                    rate_bpp: 0.82,
                    modality_summary: vec![rgb.clone(), range.clone()],
                    combined_summary: combined_raw.clone(),
                    notes: vec!["raw latent vectors (f32)".to_string()],
                },
                quantized: AeOperatingPoint {
                    encoding_mode: EncodingMode::IntraSequence,
                    rate_bpp: 0.44,
                    modality_summary: vec![rgb.clone(), range.clone()],
                    combined_summary: combined_quantized.clone(),
                    notes: vec!["8-bit latent quantization".to_string()],
                },
            },
            codec_rd_curves: vec![CodecRdCurve {
                codec: "av1".to_string(),
                encoding_mode: EncodingMode::Independent,
                rd_points: vec![
                    CodecRdPoint {
                        quality_control: "crf=28".to_string(),
                        rate_bpp: 0.64,
                        modality_summary: vec![rgb.clone(), range.clone()],
                        combined_summary: CombinedSummary {
                            mse: 0.0048,
                            psnr_db: 31.0,
                            ssim: Some(0.86),
                        },
                    },
                    CodecRdPoint {
                        quality_control: "crf=34".to_string(),
                        rate_bpp: 0.39,
                        modality_summary: vec![rgb.clone(), range.clone()],
                        combined_summary: CombinedSummary {
                            mse: 0.0068,
                            psnr_db: 29.3,
                            ssim: Some(0.79),
                        },
                    },
                ],
                lossless_anchor: Some(LosslessAnchor {
                    rate_bpp: 1.18,
                    modality_summary: vec![rgb.clone(), range.clone()],
                    combined_summary: CombinedSummary {
                        mse: 0.0002,
                        psnr_db: 38.2,
                        ssim: Some(0.99),
                    },
                    notes: vec!["ffv1 reference".to_string()],
                }),
                caveats: vec!["4:2:0 chroma subsampling can penalize RGB edges.".to_string()],
            }],
            matched_quality: vec![MatchedQualitySummary {
                quality_metric: "psnr-db".to_string(),
                target_value: 31.0,
                ae_raw_f32: MatchedQualityPoint {
                    encoding_mode: EncodingMode::Independent,
                    rate_bpp: 0.82,
                    modality_summary: vec![rgb.clone(), range.clone()],
                    combined_summary: combined_raw,
                },
                ae_quantized: MatchedQualityPoint {
                    encoding_mode: EncodingMode::IntraSequence,
                    rate_bpp: 0.44,
                    modality_summary: vec![rgb.clone(), range.clone()],
                    combined_summary: combined_quantized,
                },
                codec_points: vec![CodecMatchedQualityPoint {
                    codec: "av1".to_string(),
                    encoding_mode: EncodingMode::Independent,
                    quality_control: "crf=28".to_string(),
                    point: MatchedQualityPoint {
                        encoding_mode: EncodingMode::Independent,
                        rate_bpp: 0.64,
                        modality_summary: vec![rgb, range],
                        combined_summary: CombinedSummary {
                            mse: 0.0048,
                            psnr_db: 31.0,
                            ssim: Some(0.86),
                        },
                    },
                }],
                notes: vec!["codec point selected by nearest combined PSNR".to_string()],
            }],
            metadata: CompressionMetadata {
                run_id: "cmp-001".to_string(),
                generated_by: "unit-test".to_string(),
                caveats: vec![
                    "Validation split uses same recording segments as train.".to_string(),
                ],
                notes: [("encoder_preset".to_string(), "slow".to_string())]
                    .into_iter()
                    .collect(),
            },
        }
    }
}
