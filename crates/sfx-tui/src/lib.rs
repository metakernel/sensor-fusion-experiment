use std::{
    fs,
    io::stdout,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use sfx_core::manifest::{ProcessedSampleManifest, Split, TensorShape, read_manifest};

pub const CRATE_NAME: &str = "sfx-tui";

pub fn crate_name() -> &'static str {
    CRATE_NAME
}

pub fn run_tui(dataset_path: Option<PathBuf>) -> Result<()> {
    enable_raw_mode().context("enabling raw mode")?;

    let mut out = stdout();
    out.execute(EnterAlternateScreen)
        .context("entering alternate screen")?;

    let backend = CrosstermBackend::new(out);
    let mut terminal = match Terminal::new(backend) {
        Ok(terminal) => terminal,
        Err(error) => {
            let mut restore_out = stdout();
            let _ = disable_raw_mode();
            let _ = restore_out.execute(LeaveAlternateScreen);
            return Err(error).context("creating terminal");
        }
    };

    let result = run_tui_app(&mut terminal, dataset_path);

    disable_raw_mode().context("disabling raw mode")?;
    terminal
        .backend_mut()
        .execute(LeaveAlternateScreen)
        .context("leaving alternate screen")?;
    terminal
        .show_cursor()
        .context("restoring terminal cursor")?;

    result
}

fn run_tui_app(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    dataset_path: Option<PathBuf>,
) -> Result<()> {
    let mut app = App::load(dataset_path)?;

    loop {
        terminal.draw(|frame| ui(frame, &mut app))?;

        if event::poll(Duration::from_millis(100)).context("polling terminal events")?
            && let Event::Key(key) = event::read().context("reading terminal event")?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Down | KeyCode::Char('j') => app.move_down(),
                KeyCode::Up | KeyCode::Char('k') => app.move_up(),
                KeyCode::Char('t') => app.set_filter(Some("train")),
                KeyCode::Char('v') => app.set_filter(Some("val")),
                KeyCode::Char('e') => app.set_filter(Some("test")),
                KeyCode::Char('a') => app.set_filter(None),
                _ => {}
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct SampleRow {
    id: String,
    split: String,
    rgb_shape: String,
    range_shape: String,
    rgb_bytes: u64,
    range_bytes: u64,
}

#[derive(Debug, Clone)]
struct App {
    samples: Vec<SampleRow>,
    filtered: Vec<usize>,
    selected: usize,
    split_filter: Option<String>,
    scroll_offset: usize,
    empty_message: String,
}

impl App {
    fn load(dataset_path: Option<PathBuf>) -> Result<Self> {
        let workspace_root = find_workspace_root(std::env::current_dir()?)?;
        let manifest_path = workspace_root.join(".xtask/manifests/processed_samples.json");

        if !manifest_path.exists() {
            return Ok(Self::empty(
                "No processed samples found. Run `cargo xtask dataset prepare` first.",
            ));
        }

        let manifest: ProcessedSampleManifest =
            read_manifest(&manifest_path).map_err(anyhow::Error::from)?;
        manifest.validate().map_err(anyhow::Error::from)?;

        let processed_dir = infer_processed_dir(dataset_path, &workspace_root, &manifest);
        let rgb_shape = shape_label(manifest.rgb_shape.as_ref());
        let range_shape = shape_label(manifest.range_shape.as_ref());

        let samples = manifest
            .samples
            .iter()
            .map(|sample| SampleRow {
                id: sample.meta.id.0.clone(),
                split: split_label(&sample.meta.split).to_string(),
                rgb_shape: rgb_shape.clone(),
                range_shape: range_shape.clone(),
                rgb_bytes: file_size(&processed_dir.join(&sample.meta.rgb_path)),
                range_bytes: file_size(&processed_dir.join(&sample.meta.range_path)),
            })
            .collect();

        Ok(Self::new(
            samples,
            "No samples match the current split filter.".to_string(),
        ))
    }

    fn new(samples: Vec<SampleRow>, empty_message: String) -> Self {
        let mut app = Self {
            samples,
            filtered: Vec::new(),
            selected: 0,
            split_filter: None,
            scroll_offset: 0,
            empty_message,
        };
        app.refresh_filtered();
        app
    }

    fn empty(message: impl Into<String>) -> Self {
        Self::new(Vec::new(), message.into())
    }

    fn move_down(&mut self) {
        if !self.filtered.is_empty() && self.selected + 1 < self.filtered.len() {
            self.selected += 1;
        }
    }

    fn move_up(&mut self) {
        if !self.filtered.is_empty() && self.selected > 0 {
            self.selected -= 1;
        }
    }

    fn set_filter(&mut self, split_filter: Option<&str>) {
        self.split_filter = split_filter.map(str::to_string);
        self.refresh_filtered();
    }

    fn refresh_filtered(&mut self) {
        self.filtered = self
            .samples
            .iter()
            .enumerate()
            .filter(|(_, sample)| match self.split_filter.as_deref() {
                Some(filter) => sample.split == filter,
                None => true,
            })
            .map(|(index, _)| index)
            .collect();

        if self.filtered.is_empty() {
            self.selected = 0;
            self.scroll_offset = 0;
        } else {
            self.selected = self.selected.min(self.filtered.len().saturating_sub(1));
            self.scroll_offset = self.scroll_offset.min(self.selected);
        }
    }

    fn ensure_visible(&mut self, viewport_rows: usize) {
        if self.filtered.is_empty() || viewport_rows == 0 {
            self.scroll_offset = 0;
            return;
        }

        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }

        let viewport_end = self.scroll_offset + viewport_rows;
        if self.selected >= viewport_end {
            self.scroll_offset = self.selected + 1 - viewport_rows;
        }
    }

    fn selected_sample(&self) -> Option<&SampleRow> {
        self.filtered
            .get(self.selected)
            .and_then(|index| self.samples.get(*index))
    }

    fn filter_label(&self) -> &str {
        self.split_filter.as_deref().unwrap_or("all")
    }
}

fn ui(frame: &mut Frame, app: &mut App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(frame.area());

    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(outer[0]);

    let list_rows = panels[0].height.saturating_sub(2) as usize;
    app.ensure_visible(list_rows.max(1));

    render_samples_list(frame, app, panels[0], list_rows.max(1));
    render_details(frame, app, panels[1]);
    render_status(frame, app, outer[1]);
}

fn render_samples_list(
    frame: &mut Frame,
    app: &App,
    area: ratatui::layout::Rect,
    list_rows: usize,
) {
    let (items, selected) = if app.filtered.is_empty() {
        (
            vec![
                ListItem::new(Line::from(app.empty_message.clone()))
                    .style(Style::default().fg(Color::DarkGray)),
            ],
            None,
        )
    } else {
        let start = app.scroll_offset.min(app.filtered.len().saturating_sub(1));
        let end = (start + list_rows).min(app.filtered.len());
        let items = app.filtered[start..end]
            .iter()
            .map(|sample_index| {
                let sample = &app.samples[*sample_index];
                ListItem::new(Line::from(vec![
                    Span::styled(&sample.id, Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(format!("  [{}]", sample.split)),
                ]))
            })
            .collect();

        (items, Some(app.selected.saturating_sub(start)))
    };

    let list = List::new(items)
        .block(Block::default().title("Samples").borders(Borders::ALL))
        .highlight_symbol("❯ ")
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    let mut state = ListState::default();
    state.select(selected);
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_details(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let text = match app.selected_sample() {
        Some(sample) => vec![
            Line::from(vec![
                Span::styled("Sample ID: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(sample.id.clone()),
            ]),
            Line::from(vec![
                Span::styled("Split: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(sample.split.clone()),
            ]),
            Line::from(vec![
                Span::styled("RGB shape: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(sample.rgb_shape.clone()),
            ]),
            Line::from(vec![
                Span::styled(
                    "Range shape: ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(sample.range_shape.clone()),
            ]),
            Line::from(vec![
                Span::styled(
                    "RGB tensor: ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format_bytes(sample.rgb_bytes)),
            ]),
            Line::from(vec![
                Span::styled(
                    "Range tensor: ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format_bytes(sample.range_bytes)),
            ]),
        ],
        None => vec![Line::from(
            "Select a sample to inspect its split, shapes, and tensor sizes.",
        )],
    };

    let details = Paragraph::new(text)
        .block(
            Block::default()
                .title("Sample Details")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(details, area);
}

fn render_status(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let status = Paragraph::new(Line::from(vec![
        Span::styled("Total: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!("{}", app.samples.len())),
        Span::raw("  |  "),
        Span::styled("Visible: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!("{}", app.filtered.len())),
        Span::raw("  |  "),
        Span::styled("Filter: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(app.filter_label().to_string()),
        Span::raw("  |  q quit  j/k or ↑/↓ navigate  t/v/e split  a all"),
    ]))
    .style(Style::default().bg(Color::Blue).fg(Color::White));

    frame.render_widget(status, area);
}

fn find_workspace_root(start: PathBuf) -> Result<PathBuf> {
    let mut current = start;

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let contents = fs::read_to_string(&cargo_toml)
                .with_context(|| format!("reading {}", cargo_toml.display()))?;
            if contents.contains("[workspace]") {
                return Ok(current);
            }
        }

        if !current.pop() {
            break;
        }
    }

    anyhow::bail!("could not find workspace root from current directory")
}

fn infer_processed_dir(
    dataset_path: Option<PathBuf>,
    workspace_root: &Path,
    manifest: &ProcessedSampleManifest,
) -> PathBuf {
    if let Some(path) = dataset_path {
        return path;
    }

    let candidates = [
        workspace_root.join("data/processed"),
        workspace_root.to_path_buf(),
    ];

    if let Some(first_sample) = manifest.samples.first() {
        for candidate in &candidates {
            if candidate.join(&first_sample.meta.rgb_path).exists()
                || candidate.join(&first_sample.meta.range_path).exists()
            {
                return candidate.clone();
            }
        }
    }

    candidates[0].clone()
}

fn split_label(split: &Split) -> &'static str {
    match split {
        Split::Train => "train",
        Split::Val => "val",
        Split::Test => "test",
    }
}

fn shape_label(shape: Option<&TensorShape>) -> String {
    shape
        .map(|shape| format!("{}×{}×{}", shape.channels, shape.height, shape.width))
        .unwrap_or_else(|| "n/a".to_string())
}

fn file_size(path: &Path) -> u64 {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;

    match bytes {
        0..=1023 => format!("{bytes} B"),
        1024..=1_048_575 => format!("{:.1} KiB", bytes as f64 / KIB),
        _ => format!("{:.1} MiB", bytes as f64 / MIB),
    }
}
