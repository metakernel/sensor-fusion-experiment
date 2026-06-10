const RATE_REL_EPSILON: f64 = 1e-12;
const PSNR_EPSILON: f64 = 1e-9;

pub type Result<T> = std::result::Result<T, RdInterpolationError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdPoint {
    pub rate: f64,
    pub psnr_db: f64,
}

impl RdPoint {
    pub const fn new(rate: f64, psnr_db: f64) -> Self {
        Self { rate, psnr_db }
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum RdInterpolationError {
    #[error("target psnr must be finite, got {target_psnr_db}")]
    InvalidTargetPsnr { target_psnr_db: f64 },
    #[error(
        "rd point #{index} must have finite psnr and positive finite rate, got rate={rate}, psnr={psnr_db}"
    )]
    InvalidPoint {
        index: usize,
        rate: f64,
        psnr_db: f64,
    },
    #[error("need at least two non-degenerate rd points after monotonic filtering, found {points}")]
    NotEnoughPoints { points: usize },
    #[error(
        "target psnr {target_psnr_db} dB is outside supported range [{min_psnr_db}, {max_psnr_db}] dB"
    )]
    OutOfRange {
        target_psnr_db: f64,
        min_psnr_db: f64,
        max_psnr_db: f64,
    },
}

pub fn monotonic_rd_points(points: &[RdPoint]) -> Result<Vec<RdPoint>> {
    let mut sorted = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if !point.rate.is_finite() || point.rate <= 0.0 || !point.psnr_db.is_finite() {
            return Err(RdInterpolationError::InvalidPoint {
                index,
                rate: point.rate,
                psnr_db: point.psnr_db,
            });
        }
        sorted.push(*point);
    }

    sorted.sort_by(|left, right| {
        left.rate
            .total_cmp(&right.rate)
            .then_with(|| right.psnr_db.total_cmp(&left.psnr_db))
    });

    let mut deduplicated_by_rate: Vec<RdPoint> = Vec::with_capacity(sorted.len());
    for point in sorted {
        if let Some(last) = deduplicated_by_rate.last_mut() {
            if nearly_equal_rate(last.rate, point.rate) {
                if point.psnr_db > last.psnr_db {
                    last.psnr_db = point.psnr_db;
                }
                continue;
            }
        }
        deduplicated_by_rate.push(point);
    }

    let mut monotonic = Vec::with_capacity(deduplicated_by_rate.len());
    let mut best_psnr = f64::NEG_INFINITY;
    for point in deduplicated_by_rate {
        if point.psnr_db > best_psnr + PSNR_EPSILON {
            best_psnr = point.psnr_db;
            monotonic.push(point);
        }
    }

    if monotonic.len() < 2 {
        return Err(RdInterpolationError::NotEnoughPoints {
            points: monotonic.len(),
        });
    }

    Ok(monotonic)
}

pub fn interpolate_log_rate_at_psnr(points: &[RdPoint], target_psnr_db: f64) -> Result<f64> {
    if !target_psnr_db.is_finite() {
        return Err(RdInterpolationError::InvalidTargetPsnr { target_psnr_db });
    }

    let monotonic = monotonic_rd_points(points)?;
    let min_psnr_db = monotonic.first().expect("checked len").psnr_db;
    let max_psnr_db = monotonic.last().expect("checked len").psnr_db;

    if target_psnr_db < min_psnr_db - PSNR_EPSILON || target_psnr_db > max_psnr_db + PSNR_EPSILON {
        return Err(RdInterpolationError::OutOfRange {
            target_psnr_db,
            min_psnr_db,
            max_psnr_db,
        });
    }
    if (target_psnr_db - min_psnr_db).abs() <= PSNR_EPSILON {
        return Ok(monotonic[0].rate.ln());
    }
    if (target_psnr_db - max_psnr_db).abs() <= PSNR_EPSILON {
        return Ok(monotonic[monotonic.len() - 1].rate.ln());
    }

    for window in monotonic.windows(2) {
        let lower = window[0];
        let upper = window[1];
        if target_psnr_db > upper.psnr_db + PSNR_EPSILON {
            continue;
        }
        let span = upper.psnr_db - lower.psnr_db;
        if span <= PSNR_EPSILON {
            continue;
        }
        let t = (target_psnr_db - lower.psnr_db) / span;
        let lower_log_rate = lower.rate.ln();
        let upper_log_rate = upper.rate.ln();
        return Ok(lower_log_rate + t * (upper_log_rate - lower_log_rate));
    }

    Err(RdInterpolationError::OutOfRange {
        target_psnr_db,
        min_psnr_db,
        max_psnr_db,
    })
}

pub fn interpolate_rate_at_psnr(points: &[RdPoint], target_psnr_db: f64) -> Result<f64> {
    interpolate_log_rate_at_psnr(points, target_psnr_db).map(f64::exp)
}

fn nearly_equal_rate(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= RATE_REL_EPSILON * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(rate: f64, psnr_db: f64) -> RdPoint {
        RdPoint::new(rate, psnr_db)
    }

    fn assert_close(left: f64, right: f64) {
        let tolerance = 1e-10 * right.abs().max(1.0);
        assert!(
            (left - right).abs() <= tolerance,
            "expected values within {tolerance}, left={left}, right={right}"
        );
    }

    #[test]
    fn interpolates_in_psnr_vs_log_rate_space() {
        let points = [point(1.0, 30.0), point(4.0, 40.0)];

        let interpolated_rate = interpolate_rate_at_psnr(&points, 35.0).expect("interpolate");
        let interpolated_log_rate =
            interpolate_log_rate_at_psnr(&points, 35.0).expect("interpolate");

        assert_close(interpolated_rate, 2.0);
        assert_close(interpolated_log_rate, 2.0_f64.ln());
    }

    #[test]
    fn filters_and_sorts_to_monotonic_frontier() {
        let points = [
            point(300.0, 34.0),
            point(100.0, 30.0),
            point(100.0, 31.0),
            point(200.0, 29.0),
            point(250.0, 34.0),
            point(400.0, 36.0),
        ];

        let filtered = monotonic_rd_points(&points).expect("monotonic");
        assert_eq!(
            filtered,
            vec![point(100.0, 31.0), point(250.0, 34.0), point(400.0, 36.0)]
        );

        let interpolated = interpolate_rate_at_psnr(&points, 35.0).expect("interpolate");
        assert_close(interpolated, (250.0_f64 * 400.0_f64).sqrt());
    }

    #[test]
    fn reports_out_of_range_without_extrapolation() {
        let points = [point(100.0, 30.0), point(200.0, 35.0)];

        let err = interpolate_rate_at_psnr(&points, 40.0).expect_err("out of range");
        assert!(matches!(
            err,
            RdInterpolationError::OutOfRange {
                target_psnr_db,
                min_psnr_db,
                max_psnr_db
            } if target_psnr_db == 40.0 && min_psnr_db == 30.0 && max_psnr_db == 35.0
        ));

        let err = interpolate_rate_at_psnr(&points, 25.0).expect_err("out of range");
        assert!(matches!(
            err,
            RdInterpolationError::OutOfRange {
                target_psnr_db,
                min_psnr_db,
                max_psnr_db
            } if target_psnr_db == 25.0 && min_psnr_db == 30.0 && max_psnr_db == 35.0
        ));
    }

    #[test]
    fn returns_error_for_degenerate_curves() {
        let points = [point(100.0, 30.0), point(100.0, 30.0), point(200.0, 29.0)];

        let err = monotonic_rd_points(&points).expect_err("degenerate curve");
        assert_eq!(err, RdInterpolationError::NotEnoughPoints { points: 1 });
    }

    #[test]
    fn returns_exact_rate_at_range_boundaries() {
        let points = [point(10.0, 20.0), point(40.0, 30.0)];

        assert_close(
            interpolate_rate_at_psnr(&points, 20.0).expect("min boundary"),
            10.0,
        );
        assert_close(
            interpolate_rate_at_psnr(&points, 30.0).expect("max boundary"),
            40.0,
        );
    }

    #[test]
    fn honors_psnr_epsilon_at_boundaries() {
        let points = [point(10.0, 20.0), point(40.0, 30.0)];

        assert_close(
            interpolate_rate_at_psnr(&points, 20.0 - PSNR_EPSILON * 0.5)
                .expect("within lower epsilon"),
            10.0,
        );
        assert_close(
            interpolate_rate_at_psnr(&points, 30.0 + PSNR_EPSILON * 0.5)
                .expect("within upper epsilon"),
            40.0,
        );

        let err = interpolate_rate_at_psnr(&points, 30.0 + PSNR_EPSILON * 2.0)
            .expect_err("outside upper epsilon");
        assert!(matches!(
            err,
            RdInterpolationError::OutOfRange {
                target_psnr_db,
                min_psnr_db,
                max_psnr_db
            } if target_psnr_db > max_psnr_db && min_psnr_db == 20.0 && max_psnr_db == 30.0
        ));
    }

    #[test]
    fn deduplicates_nearly_equal_rates_with_relative_tolerance() {
        let points = [
            point(100.0, 30.0),
            point(100.0 * (1.0 + RATE_REL_EPSILON * 0.5), 31.0),
            point(200.0, 35.0),
        ];

        let filtered = monotonic_rd_points(&points).expect("monotonic");
        assert_eq!(filtered, vec![point(100.0, 31.0), point(200.0, 35.0)]);
    }

    #[test]
    fn interpolates_across_flat_psnr_segments_after_filtering() {
        let points = [point(100.0, 30.0), point(200.0, 30.0), point(400.0, 40.0)];

        let filtered = monotonic_rd_points(&points).expect("monotonic");
        assert_eq!(filtered, vec![point(100.0, 30.0), point(400.0, 40.0)]);

        let interpolated = interpolate_rate_at_psnr(&points, 35.0).expect("interpolate");
        assert_close(interpolated, 200.0);
    }

    #[test]
    fn reports_invalid_point_and_target_inputs() {
        let invalid_point = [point(0.0, 20.0), point(10.0, 25.0)];
        let err = interpolate_rate_at_psnr(&invalid_point, 22.0).expect_err("invalid point");
        assert!(matches!(
            err,
            RdInterpolationError::InvalidPoint {
                index: 0,
                rate,
                psnr_db
            } if rate == 0.0 && psnr_db == 20.0
        ));

        let valid_points = [point(10.0, 20.0), point(20.0, 25.0)];
        let err = interpolate_rate_at_psnr(&valid_points, f64::NAN).expect_err("invalid target");
        assert!(matches!(
            err,
            RdInterpolationError::InvalidTargetPsnr { target_psnr_db }
            if target_psnr_db.is_nan()
        ));
    }
}
