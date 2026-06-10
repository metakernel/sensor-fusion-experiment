use std::collections::BTreeSet;
use std::fmt;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Command;

pub type Result<T> = std::result::Result<T, FfmpegError>;

const H264_ENCODERS: &[&str] = &[
    "libx264",
    "h264_nvenc",
    "h264_qsv",
    "h264_amf",
    "h264_videotoolbox",
    "h264",
];
const H265_ENCODERS: &[&str] = &[
    "libx265",
    "hevc_nvenc",
    "hevc_qsv",
    "hevc_amf",
    "hevc_videotoolbox",
    "hevc",
];
const AV1_ENCODERS: &[&str] = &[
    "libaom-av1",
    "svtav1",
    "rav1e",
    "av1_nvenc",
    "av1_qsv",
    "av1_amf",
    "av1",
];
const PNG_ENCODERS: &[&str] = &["png"];
const FFV1_ENCODERS: &[&str] = &["ffv1"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
    Png,
    Ffv1,
}

impl VideoCodec {
    pub fn ffmpeg_codec_name(self) -> &'static str {
        codec_name(self)
    }

    pub fn encoder_candidates(self) -> &'static [&'static str] {
        codec_encoder_candidates(self)
    }

    pub fn default_pix_fmt_family(self) -> PixFmtFamily {
        default_pix_fmt_family_for_codec(self)
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(codec_name(*self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixFmtFamily {
    Rgb,
    Yuv,
    Gray,
}

impl fmt::Display for PixFmtFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixFmtFamily::Rgb => f.write_str("rgb"),
            PixFmtFamily::Yuv => f.write_str("yuv"),
            PixFmtFamily::Gray => f.write_str("gray"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixFmtHooks {
    pub rgb: String,
    pub yuv: String,
    pub gray: String,
}

impl Default for PixFmtHooks {
    fn default() -> Self {
        Self {
            rgb: default_pix_fmt_for_family(PixFmtFamily::Rgb).to_string(),
            yuv: default_pix_fmt_for_family(PixFmtFamily::Yuv).to_string(),
            gray: default_pix_fmt_for_family(PixFmtFamily::Gray).to_string(),
        }
    }
}

impl PixFmtHooks {
    pub fn for_family(&self, family: PixFmtFamily) -> &str {
        match family {
            PixFmtFamily::Rgb => &self.rgb,
            PixFmtFamily::Yuv => &self.yuv,
            PixFmtFamily::Gray => &self.gray,
        }
    }

    pub fn resolve(&self, family: PixFmtFamily, explicit: Option<&str>) -> String {
        explicit
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.for_family(family).to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeRequest {
    pub input: PathBuf,
    pub output: PathBuf,
    pub codec: VideoCodec,
    pub encoder: Option<String>,
    pub pix_fmt: Option<String>,
    pub pix_fmt_family: PixFmtFamily,
    pub overwrite: bool,
    pub pre_input_args: Vec<String>,
    pub output_args: Vec<String>,
}

impl EncodeRequest {
    pub fn new(input: impl Into<PathBuf>, output: impl Into<PathBuf>, codec: VideoCodec) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            codec,
            encoder: None,
            pix_fmt: None,
            pix_fmt_family: codec.default_pix_fmt_family(),
            overwrite: true,
            pre_input_args: Vec::new(),
            output_args: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeRequest {
    pub input: PathBuf,
    pub output: PathBuf,
    pub codec_hint: Option<VideoCodec>,
    pub pix_fmt: Option<String>,
    pub pix_fmt_family: PixFmtFamily,
    pub overwrite: bool,
    pub pre_input_args: Vec<String>,
    pub output_args: Vec<String>,
}

impl DecodeRequest {
    pub fn new(
        input: impl Into<PathBuf>,
        output: impl Into<PathBuf>,
        pix_fmt_family: PixFmtFamily,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            codec_hint: None,
            pix_fmt: None,
            pix_fmt_family,
            overwrite: true,
            pre_input_args: Vec::new(),
            output_args: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeExecution {
    pub encoder: String,
    pub pix_fmt: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeExecution {
    pub pix_fmt: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecProbeRequest {
    pub codec: VideoCodec,
    pub requested_encoder: Option<String>,
}

impl CodecProbeRequest {
    pub fn new(codec: VideoCodec) -> Self {
        Self {
            codec,
            requested_encoder: None,
        }
    }

    pub fn with_requested_encoder(mut self, encoder: impl Into<String>) -> Self {
        self.requested_encoder = Some(encoder.into());
        self
    }
}

impl From<VideoCodec> for CodecProbeRequest {
    fn from(codec: VideoCodec) -> Self {
        Self::new(codec)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecSkipReason {
    FfmpegBinaryMissing {
        program: String,
    },
    FfmpegProbeFailed {
        detail: String,
    },
    RequestedEncoderUnavailable {
        encoder: String,
        available: Vec<String>,
    },
    CodecEncoderUnavailable {
        candidates: Vec<String>,
    },
}

impl fmt::Display for CodecSkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecSkipReason::FfmpegBinaryMissing { program } => {
                write!(f, "ffmpeg binary `{program}` was not found")
            }
            CodecSkipReason::FfmpegProbeFailed { detail } => {
                write!(f, "ffmpeg encoder probe failed: {detail}")
            }
            CodecSkipReason::RequestedEncoderUnavailable { encoder, available } => {
                if available.is_empty() {
                    write!(
                        f,
                        "requested encoder `{encoder}` is unavailable and no alternatives were detected"
                    )
                } else {
                    write!(
                        f,
                        "requested encoder `{encoder}` is unavailable; available alternatives: {available:?}"
                    )
                }
            }
            CodecSkipReason::CodecEncoderUnavailable { candidates } => {
                write!(
                    f,
                    "no available encoder matched expected candidates: {candidates:?}"
                )
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecProbeOutcome {
    pub codec: VideoCodec,
    pub requested_encoder: Option<String>,
    pub selected_encoder: Option<String>,
    pub skipped: bool,
    pub skip_reason: Option<CodecSkipReason>,
    pub note: String,
}

impl CodecProbeOutcome {
    pub fn runnable(&self) -> bool {
        !self.skipped
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecProbeReport {
    pub ffmpeg_available: bool,
    pub outcomes: Vec<CodecProbeOutcome>,
    pub notes: Vec<String>,
}

impl CodecProbeReport {
    pub fn runnable_codecs(&self) -> impl Iterator<Item = &CodecProbeOutcome> {
        self.outcomes.iter().filter(|outcome| outcome.runnable())
    }

    pub fn skipped_codecs(&self) -> impl Iterator<Item = &CodecProbeOutcome> {
        self.outcomes.iter().filter(|outcome| outcome.skipped)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncoderInventory {
    encoders: BTreeSet<String>,
}

impl EncoderInventory {
    pub fn from_ffmpeg_encoders_output(output: &str) -> Self {
        let mut encoders = BTreeSet::new();
        for line in output.lines() {
            if let Some(name) = parse_video_encoder_name(line) {
                encoders.insert(name.to_string());
            }
        }
        Self { encoders }
    }

    pub fn contains(&self, encoder: &str) -> bool {
        self.encoders.contains(encoder)
    }

    pub fn best_encoder_for_codec(&self, codec: VideoCodec) -> Option<&'static str> {
        codec_encoder_candidates(codec)
            .iter()
            .copied()
            .find(|candidate| self.contains(candidate))
    }

    pub fn available_for_codec(&self, codec: VideoCodec) -> Vec<&'static str> {
        codec_encoder_candidates(codec)
            .iter()
            .copied()
            .filter(|candidate| self.contains(candidate))
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.encoders.iter().map(String::as_str)
    }
}

pub trait FfmpegExecutor {
    fn probe_encoders(&self) -> Result<EncoderInventory>;
    fn encode(&self, request: &EncodeRequest) -> Result<EncodeExecution>;
    fn decode(&self, request: &DecodeRequest) -> Result<DecodeExecution>;
}

pub fn probe_codec_support(
    executor: &dyn FfmpegExecutor,
    requests: &[CodecProbeRequest],
) -> CodecProbeReport {
    match executor.probe_encoders() {
        Ok(inventory) => {
            let mut notes = Vec::new();
            let outcomes = requests
                .iter()
                .map(|request| {
                    let outcome = evaluate_codec_probe_request(request, &inventory);
                    if outcome.skipped {
                        notes.push(outcome.note.clone());
                    }
                    outcome
                })
                .collect();

            CodecProbeReport {
                ffmpeg_available: true,
                outcomes,
                notes,
            }
        }
        Err(error) => {
            let shared_reason = match &error {
                FfmpegError::Spawn { program, source } if source.kind() == ErrorKind::NotFound => {
                    CodecSkipReason::FfmpegBinaryMissing {
                        program: program.clone(),
                    }
                }
                _ => CodecSkipReason::FfmpegProbeFailed {
                    detail: error.to_string(),
                },
            };
            let mut notes = vec![format!(
                "skipping ffmpeg-dependent codecs because {shared_reason}"
            )];
            let outcomes: Vec<CodecProbeOutcome> = requests
                .iter()
                .map(|request| {
                    let note = format!("skipping codec `{}`: {shared_reason}", request.codec);
                    notes.push(note.clone());
                    CodecProbeOutcome {
                        codec: request.codec,
                        requested_encoder: request.requested_encoder.clone(),
                        selected_encoder: None,
                        skipped: true,
                        skip_reason: Some(shared_reason.clone()),
                        note,
                    }
                })
                .collect();

            CodecProbeReport {
                ffmpeg_available: false,
                outcomes,
                notes,
            }
        }
    }
}

fn evaluate_codec_probe_request(
    request: &CodecProbeRequest,
    inventory: &EncoderInventory,
) -> CodecProbeOutcome {
    if let Some(explicit_encoder) = request.requested_encoder.as_deref() {
        if inventory.contains(explicit_encoder) {
            return CodecProbeOutcome {
                codec: request.codec,
                requested_encoder: request.requested_encoder.clone(),
                selected_encoder: Some(explicit_encoder.to_string()),
                skipped: false,
                skip_reason: None,
                note: format!(
                    "codec `{}` will use requested encoder `{explicit_encoder}`",
                    request.codec
                ),
            };
        }

        let available = inventory
            .available_for_codec(request.codec)
            .into_iter()
            .map(ToString::to_string)
            .collect();
        let reason = CodecSkipReason::RequestedEncoderUnavailable {
            encoder: explicit_encoder.to_string(),
            available,
        };
        return CodecProbeOutcome {
            codec: request.codec,
            requested_encoder: request.requested_encoder.clone(),
            selected_encoder: None,
            skipped: true,
            note: format!("skipping codec `{}`: {reason}", request.codec),
            skip_reason: Some(reason),
        };
    }

    if let Some(selected_encoder) = inventory.best_encoder_for_codec(request.codec) {
        return CodecProbeOutcome {
            codec: request.codec,
            requested_encoder: None,
            selected_encoder: Some(selected_encoder.to_string()),
            skipped: false,
            skip_reason: None,
            note: format!(
                "codec `{}` will use detected encoder `{selected_encoder}`",
                request.codec
            ),
        };
    }

    let reason = CodecSkipReason::CodecEncoderUnavailable {
        candidates: codec_encoder_candidates(request.codec)
            .iter()
            .map(ToString::to_string)
            .collect(),
    };
    CodecProbeOutcome {
        codec: request.codec,
        requested_encoder: None,
        selected_encoder: None,
        skipped: true,
        note: format!("skipping codec `{}`: {reason}", request.codec),
        skip_reason: Some(reason),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemFfmpegRunner {
    ffmpeg_bin: PathBuf,
    pix_fmt_hooks: PixFmtHooks,
}

impl Default for SystemFfmpegRunner {
    fn default() -> Self {
        Self::new("ffmpeg")
    }
}

impl SystemFfmpegRunner {
    pub fn new(ffmpeg_bin: impl Into<PathBuf>) -> Self {
        Self {
            ffmpeg_bin: ffmpeg_bin.into(),
            pix_fmt_hooks: PixFmtHooks::default(),
        }
    }

    pub fn with_pix_fmt_hooks(mut self, pix_fmt_hooks: PixFmtHooks) -> Self {
        self.pix_fmt_hooks = pix_fmt_hooks;
        self
    }

    pub fn pix_fmt_hooks(&self) -> &PixFmtHooks {
        &self.pix_fmt_hooks
    }

    pub fn probe_codec_support(&self, requests: &[CodecProbeRequest]) -> CodecProbeReport {
        probe_codec_support(self, requests)
    }

    fn run_checked(&self, args: &[String]) -> Result<std::process::Output> {
        let output = Command::new(&self.ffmpeg_bin)
            .args(args)
            .output()
            .map_err(|source| FfmpegError::Spawn {
                program: self.ffmpeg_bin.to_string_lossy().into_owned(),
                source,
            })?;

        if output.status.success() {
            return Ok(output);
        }

        Err(FfmpegError::CommandFailed {
            program: self.ffmpeg_bin.to_string_lossy().into_owned(),
            args: args.to_vec(),
            status: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    fn resolve_encoder(
        &self,
        request: &EncodeRequest,
        inventory: &EncoderInventory,
    ) -> Result<String> {
        if let Some(explicit) = request.encoder.as_deref() {
            if inventory.contains(explicit) {
                return Ok(explicit.to_string());
            }
            return Err(FfmpegError::RequestedEncoderUnavailable {
                encoder: explicit.to_string(),
            });
        }

        if let Some(selected) = inventory.best_encoder_for_codec(request.codec) {
            return Ok(selected.to_string());
        }

        Err(FfmpegError::NoEncoderAvailable {
            codec: request.codec,
            candidates: codec_encoder_candidates(request.codec)
                .iter()
                .map(ToString::to_string)
                .collect(),
        })
    }
}

impl FfmpegExecutor for SystemFfmpegRunner {
    fn probe_encoders(&self) -> Result<EncoderInventory> {
        let args = build_probe_encoders_args();
        let output = self.run_checked(&args)?;

        let mut listing = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.stderr.is_empty() {
            if !listing.is_empty() && !listing.ends_with('\n') {
                listing.push('\n');
            }
            listing.push_str(&String::from_utf8_lossy(&output.stderr));
        }

        Ok(EncoderInventory::from_ffmpeg_encoders_output(&listing))
    }

    fn encode(&self, request: &EncodeRequest) -> Result<EncodeExecution> {
        let inventory = self.probe_encoders()?;
        let encoder = self.resolve_encoder(request, &inventory)?;
        let pix_fmt = self
            .pix_fmt_hooks
            .resolve(request.pix_fmt_family, request.pix_fmt.as_deref());
        let args = build_encode_args(request, &encoder, &pix_fmt);
        self.run_checked(&args)?;

        Ok(EncodeExecution {
            encoder,
            pix_fmt,
            args,
        })
    }

    fn decode(&self, request: &DecodeRequest) -> Result<DecodeExecution> {
        let pix_fmt = self
            .pix_fmt_hooks
            .resolve(request.pix_fmt_family, request.pix_fmt.as_deref());
        let args = build_decode_args(request, &pix_fmt);
        self.run_checked(&args)?;

        Ok(DecodeExecution { pix_fmt, args })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FfmpegError {
    #[error("failed to launch `{program}`: {source}")]
    Spawn {
        program: String,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "`{program}` failed with status {status:?} (args: {args:?})\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )]
    CommandFailed {
        program: String,
        args: Vec<String>,
        status: Option<i32>,
        stdout: String,
        stderr: String,
    },
    #[error("requested ffmpeg encoder `{encoder}` is unavailable")]
    RequestedEncoderUnavailable { encoder: String },
    #[error("no encoder available for codec `{codec}`; candidates: {candidates:?}")]
    NoEncoderAvailable {
        codec: VideoCodec,
        candidates: Vec<String>,
    },
}

pub fn codec_name(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "hevc",
        VideoCodec::Av1 => "av1",
        VideoCodec::Png => "png",
        VideoCodec::Ffv1 => "ffv1",
    }
}

pub fn codec_encoder_candidates(codec: VideoCodec) -> &'static [&'static str] {
    match codec {
        VideoCodec::H264 => H264_ENCODERS,
        VideoCodec::H265 => H265_ENCODERS,
        VideoCodec::Av1 => AV1_ENCODERS,
        VideoCodec::Png => PNG_ENCODERS,
        VideoCodec::Ffv1 => FFV1_ENCODERS,
    }
}

pub fn default_pix_fmt_family_for_codec(codec: VideoCodec) -> PixFmtFamily {
    match codec {
        VideoCodec::Png => PixFmtFamily::Rgb,
        VideoCodec::H264 | VideoCodec::H265 | VideoCodec::Av1 | VideoCodec::Ffv1 => {
            PixFmtFamily::Yuv
        }
    }
}

pub fn default_pix_fmt_for_family(family: PixFmtFamily) -> &'static str {
    match family {
        PixFmtFamily::Rgb => "rgb24",
        PixFmtFamily::Yuv => "yuv420p",
        PixFmtFamily::Gray => "gray",
    }
}

pub fn build_probe_encoders_args() -> Vec<String> {
    vec!["-hide_banner".to_string(), "-encoders".to_string()]
}

pub fn build_encode_args(request: &EncodeRequest, encoder: &str, pix_fmt: &str) -> Vec<String> {
    let mut args = base_ffmpeg_args(request.overwrite);
    args.extend(request.pre_input_args.iter().cloned());
    args.push("-i".to_string());
    args.push(request.input.to_string_lossy().into_owned());
    args.push("-c:v".to_string());
    args.push(encoder.to_string());
    args.push("-pix_fmt".to_string());
    args.push(pix_fmt.to_string());
    args.extend(request.output_args.iter().cloned());
    args.push(request.output.to_string_lossy().into_owned());
    args
}

pub fn build_decode_args(request: &DecodeRequest, pix_fmt: &str) -> Vec<String> {
    let mut args = base_ffmpeg_args(request.overwrite);
    args.extend(request.pre_input_args.iter().cloned());
    if let Some(codec) = request.codec_hint {
        args.push("-c:v".to_string());
        args.push(codec_name(codec).to_string());
    }
    args.push("-i".to_string());
    args.push(request.input.to_string_lossy().into_owned());
    args.push("-pix_fmt".to_string());
    args.push(pix_fmt.to_string());
    args.extend(request.output_args.iter().cloned());
    args.push(request.output.to_string_lossy().into_owned());
    args
}

fn base_ffmpeg_args(overwrite: bool) -> Vec<String> {
    vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        if overwrite { "-y" } else { "-n" }.to_string(),
    ]
}

fn parse_video_encoder_name(line: &str) -> Option<&str> {
    let mut parts = line.split_whitespace();
    let flags = parts.next()?;
    if flags.len() != 6 || !flags.starts_with('V') {
        return None;
    }
    if !flags.chars().all(|ch| ch == '.' || ch.is_ascii_uppercase()) {
        return None;
    }

    let name = parts.next()?;
    if name == "=" || name.starts_with('=') || name.chars().all(|ch| ch == '-') {
        return None;
    }

    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    struct InventoryExecutor {
        inventory: EncoderInventory,
    }

    impl FfmpegExecutor for InventoryExecutor {
        fn probe_encoders(&self) -> Result<EncoderInventory> {
            Ok(self.inventory.clone())
        }

        fn encode(&self, _request: &EncodeRequest) -> Result<EncodeExecution> {
            panic!("encode is not used in these tests")
        }

        fn decode(&self, _request: &DecodeRequest) -> Result<DecodeExecution> {
            panic!("decode is not used in these tests")
        }
    }

    struct MissingBinaryExecutor;

    impl FfmpegExecutor for MissingBinaryExecutor {
        fn probe_encoders(&self) -> Result<EncoderInventory> {
            Err(FfmpegError::Spawn {
                program: "ffmpeg".to_string(),
                source: std::io::Error::new(ErrorKind::NotFound, "not found"),
            })
        }

        fn encode(&self, _request: &EncodeRequest) -> Result<EncodeExecution> {
            panic!("encode is not used in these tests")
        }

        fn decode(&self, _request: &DecodeRequest) -> Result<DecodeExecution> {
            panic!("decode is not used in these tests")
        }
    }

    struct FailedProbeExecutor;

    impl FfmpegExecutor for FailedProbeExecutor {
        fn probe_encoders(&self) -> Result<EncoderInventory> {
            Err(FfmpegError::CommandFailed {
                program: "ffmpeg".to_string(),
                args: build_probe_encoders_args(),
                status: Some(1),
                stdout: String::new(),
                stderr: "unknown option".to_string(),
            })
        }

        fn encode(&self, _request: &EncodeRequest) -> Result<EncodeExecution> {
            panic!("encode is not used in these tests")
        }

        fn decode(&self, _request: &DecodeRequest) -> Result<DecodeExecution> {
            panic!("decode is not used in these tests")
        }
    }

    #[test]
    fn parses_video_encoders_from_ffmpeg_output() {
        let output = r#"
Encoders:
 V..... = Video
 A..... = Audio
 ------
 V....D libx264          libx264 H.264 / AVC
 V..... h264_nvenc       NVIDIA NVENC H.264 encoder
 A..... aac              AAC (Advanced Audio Coding)
 V..... ffv1             FFmpeg video codec #1
"#;

        let inventory = EncoderInventory::from_ffmpeg_encoders_output(output);
        assert!(inventory.contains("libx264"));
        assert!(inventory.contains("h264_nvenc"));
        assert!(inventory.contains("ffv1"));
        assert!(!inventory.contains("aac"));
    }

    #[test]
    fn parser_rejects_non_video_or_malformed_encoder_lines() {
        assert_eq!(parse_video_encoder_name("A..... aac AAC"), None);
        assert_eq!(parse_video_encoder_name("V..... = Video"), None);
        assert_eq!(parse_video_encoder_name("V....- libx264 H.264"), None);
        assert_eq!(parse_video_encoder_name("v..... libx264 H.264"), None);
        assert_eq!(
            parse_video_encoder_name("V..... libx265 H.265 / HEVC"),
            Some("libx265")
        );
    }

    #[test]
    fn chooses_best_encoder_by_codec_priority() {
        let inventory = EncoderInventory::from_ffmpeg_encoders_output(
            "V..... h264_nvenc\nV..... libx264\nV..... hevc_nvenc\n",
        );

        assert_eq!(
            inventory.best_encoder_for_codec(VideoCodec::H264),
            Some("libx264")
        );
        assert_eq!(
            inventory.available_for_codec(VideoCodec::H265),
            vec!["hevc_nvenc"]
        );
        assert_eq!(inventory.best_encoder_for_codec(VideoCodec::Av1), None);
    }

    #[test]
    fn probe_codec_support_marks_missing_codec_encoders_as_skipped() {
        let executor = InventoryExecutor {
            inventory: EncoderInventory::from_ffmpeg_encoders_output(
                "V..... libx264\nV..... ffv1\n",
            ),
        };
        let requests = vec![VideoCodec::H264.into(), VideoCodec::Av1.into()];

        let report = probe_codec_support(&executor, &requests);

        assert!(report.ffmpeg_available);
        assert_eq!(report.runnable_codecs().count(), 1);
        assert_eq!(report.skipped_codecs().count(), 1);

        let h264 = report
            .outcomes
            .iter()
            .find(|outcome| outcome.codec == VideoCodec::H264)
            .expect("h264 outcome");
        assert_eq!(h264.selected_encoder.as_deref(), Some("libx264"));
        assert!(!h264.skipped);

        let av1 = report
            .outcomes
            .iter()
            .find(|outcome| outcome.codec == VideoCodec::Av1)
            .expect("av1 outcome");
        assert!(av1.skipped);
        assert!(matches!(
            av1.skip_reason.as_ref(),
            Some(CodecSkipReason::CodecEncoderUnavailable { .. })
        ));
        assert!(av1.note.contains("skipping codec `av1`"));
    }

    #[test]
    fn probe_codec_support_skips_when_requested_encoder_is_unavailable() {
        let executor = InventoryExecutor {
            inventory: EncoderInventory::from_ffmpeg_encoders_output("V..... libx264\n"),
        };
        let requests =
            vec![CodecProbeRequest::new(VideoCodec::H264).with_requested_encoder("h264_nvenc")];

        let report = probe_codec_support(&executor, &requests);
        let outcome = report.outcomes.first().expect("codec outcome");

        assert!(outcome.skipped);
        assert_eq!(outcome.selected_encoder, None);
        assert!(matches!(
            outcome.skip_reason.as_ref(),
            Some(CodecSkipReason::RequestedEncoderUnavailable { encoder, available })
                if encoder == "h264_nvenc" && available == &vec!["libx264".to_string()]
        ));
        assert!(
            outcome
                .note
                .contains("requested encoder `h264_nvenc` is unavailable")
        );
    }

    #[test]
    fn probe_codec_support_skips_all_codecs_when_ffmpeg_is_missing() {
        let requests = vec![VideoCodec::H264.into(), VideoCodec::H265.into()];

        let report = probe_codec_support(&MissingBinaryExecutor, &requests);

        assert!(!report.ffmpeg_available);
        assert_eq!(report.skipped_codecs().count(), 2);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("ffmpeg binary `ffmpeg` was not found"))
        );
        assert!(report.outcomes.iter().all(|outcome| matches!(
            outcome.skip_reason.as_ref(),
            Some(CodecSkipReason::FfmpegBinaryMissing { program }) if program == "ffmpeg"
        )));
    }

    #[test]
    fn probe_codec_support_skips_all_codecs_when_probe_fails() {
        let requests = vec![VideoCodec::Av1.into()];

        let report = probe_codec_support(&FailedProbeExecutor, &requests);
        let outcome = report.outcomes.first().expect("codec outcome");

        assert!(!report.ffmpeg_available);
        assert!(outcome.skipped);
        assert!(matches!(
            outcome.skip_reason.as_ref(),
            Some(CodecSkipReason::FfmpegProbeFailed { detail })
                if detail.contains("unknown option")
        ));
    }

    #[test]
    fn resolves_pix_fmt_from_hooks_and_overrides() {
        let hooks = PixFmtHooks {
            rgb: "rgb48le".to_string(),
            ..PixFmtHooks::default()
        };

        assert_eq!(hooks.resolve(PixFmtFamily::Rgb, None), "rgb48le");
        assert_eq!(hooks.resolve(PixFmtFamily::Rgb, Some("   ")), "rgb48le");
        assert_eq!(hooks.resolve(PixFmtFamily::Yuv, None), "yuv420p");
        assert_eq!(
            hooks.resolve(PixFmtFamily::Gray, Some(" gray10le ")),
            "gray10le"
        );
    }

    #[test]
    fn builds_encode_args_with_encoder_and_pix_fmt() {
        let mut request = EncodeRequest::new("input.y4m", "output.mp4", VideoCodec::H264);
        request.pre_input_args = strings(&["-framerate", "30"]);
        request.output_args = strings(&["-crf", "18"]);

        let args = build_encode_args(&request, "libx264", "yuv420p");
        assert_eq!(
            args,
            strings(&[
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "30",
                "-i",
                "input.y4m",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "output.mp4",
            ])
        );
    }

    #[test]
    fn builds_decode_args_with_codec_hint_and_pix_fmt() {
        let mut request = DecodeRequest::new("input.mp4", "frame.rgb", PixFmtFamily::Rgb);
        request.overwrite = false;
        request.codec_hint = Some(VideoCodec::H264);
        request.output_args = strings(&["-f", "rawvideo"]);

        let args = build_decode_args(&request, "rgb24");
        assert_eq!(
            args,
            strings(&[
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-c:v",
                "h264",
                "-i",
                "input.mp4",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "frame.rgb",
            ])
        );
    }

    #[test]
    fn builds_decode_args_without_codec_hint() {
        let mut request = DecodeRequest::new("input.av1", "frame.rgb", PixFmtFamily::Rgb);
        request.output_args = strings(&["-frames:v", "1"]);

        let args = build_decode_args(&request, "rgb24");
        assert_eq!(
            args,
            strings(&[
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "input.av1",
                "-pix_fmt",
                "rgb24",
                "-frames:v",
                "1",
                "frame.rgb",
            ])
        );
    }

    #[test]
    fn maps_expected_codecs_and_defaults() {
        assert_eq!(codec_name(VideoCodec::H265), "hevc");
        assert_eq!(codec_encoder_candidates(VideoCodec::Png), &["png"]);
        assert!(codec_encoder_candidates(VideoCodec::Av1).contains(&"libaom-av1"));
        assert_eq!(
            default_pix_fmt_family_for_codec(VideoCodec::Png),
            PixFmtFamily::Rgb
        );
        assert_eq!(default_pix_fmt_for_family(PixFmtFamily::Gray), "gray");
    }
}
