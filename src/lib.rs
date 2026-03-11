// Copyright © 2026 H S Helson Go
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#![no_std]

use num_traits::float::FloatCore;

/// Calculates the Gram Polynomial (s=0) or its sth derivative over 2m+1 points, at the point i,
/// for the k'th order (0-based)
///
/// # Arguments
/// * `i` - The point at which to evaluate the polynomial
/// * `m` - Controls the number of points 2m + 1 in the set
/// * `k` - The order of the polynomial (0-based)
/// * `s` - The order of the derivative (0-based)
///
/// # Returns
/// The value of the Gram Polynomial or its derivative at the point `i`
fn gram_poly<T: FloatCore>(i: T, m: i32, k: i32, s: i32) -> T {
    if k > 0 {
        let ms = T::from(m).unwrap();
        let ks = T::from(k).unwrap();
        let ss = T::from(s).unwrap();
        let two = T::from(2).unwrap();
        let four = two + two;
        (four * ks - two) / (ks * (two * ms - ks + T::one()))
            * (i * gram_poly(i, m, k - 1, s) + ss * gram_poly(i, m, k - 1, s - 1))
            - ((ks - T::one()) * (two * ms + ks)) / (ks * (two * ms - ks + T::one()))
                * gram_poly(i, m, k - 2, s)
    } else if (k == 0) && (s == 0) {
        T::one()
    } else {
        T::zero()
    }
}

/// Calculates the generalized factorial (a)(a-1)(a-2)...(a-b+1)
///
/// # Arguments
/// * `a` - The starting point of the factorial
/// * `b` - The number of terms to multiply
///
/// # Returns
/// The value of the generalized factorial
fn gen_fact(a: i32, b: i32) -> i32 {
    let mut gf = 1;
    for j in (a - b + 1)..=a {
        gf *= j;
    }
    gf
}

/// Calculates the weight of the ith data point for the t'th least-square point of the s'th
/// derivative, over 2m+1 points, order n
///
/// # Arguments
/// * `i` - The ith data point
/// * `t` - The t'th least-square point (0-based)
/// * `m` - Controls the number of points 2m + 1 in the set
/// * `n` - The order of the polynomial (0-based)
/// * `s` - The order of the derivative (0-based)
///
/// # Returns
/// The weight of the ith data point for the t'th least-square point of the s'th derivative
fn weight<T: FloatCore>(i: T, t: i32, m: i32, n: i32, s: i32) -> T {
    let ts = T::from(t).unwrap();
    (0..=n).fold(T::zero(), |sum, k| {
        let ks = T::from(k).unwrap();
        sum + (T::from(2).unwrap() * ks + T::one()) * T::from(gen_fact(2 * m, k)).unwrap()
            / T::from(gen_fact(2 * m + k + 1, k + 1)).unwrap()
            * gram_poly(i, m, k, 0)
            * gram_poly(ts, m, k, s)
    })
}

#[cfg(not(feature = "std"))]
mod weight_array {
    use core::ops::{Index, IndexMut};
    use num_traits::float::FloatCore;

    const MAX_NUM_POINTS: usize = 20;

    /// Size constrained array to hold weights, size is determined at initialization time, but cannot
    /// exceed MAX_NUM_POINTS; API
    #[derive(Debug, Clone)]
    pub struct WeightArray<T: FloatCore> {
        weights: [T; MAX_NUM_POINTS],
        size: usize,
    }

    impl<T: FloatCore> WeightArray<T> {
        pub fn new(size: usize) -> Self {
            assert!(
                size <= MAX_NUM_POINTS,
                "Size ({size}) must be less than or equal to MAX_NUM_POINTS ({MAX_NUM_POINTS})"
            );
            Self {
                weights: [T::zero(); MAX_NUM_POINTS],
                size,
            }
        }

        pub fn len(&self) -> usize {
            self.size
        }

        pub fn iter(&self) -> impl Iterator<Item = &T> {
            self.weights.iter().take(self.size)
        }
    }

    // Implement iteration and indexing, but not push or pop
    impl<T: FloatCore> Index<usize> for WeightArray<T> {
        type Output = T;

        fn index(&self, index: usize) -> &Self::Output {
            &self.weights[index]
        }
    }

    impl<T: FloatCore> IndexMut<usize> for WeightArray<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.weights[index]
        }
    }

    pub fn make_weight_array<T: FloatCore>(size: usize) -> WeightArray<T> {
        WeightArray::new(size)
    }
}

#[cfg(feature = "std")]
mod weight_array {
    extern crate std;
    use std::vec::Vec;

    use num_traits::Zero;

    pub type WeightArray<T> = Vec<T>;
    pub fn make_weight_array<T: Zero + Copy>(size: usize) -> WeightArray<T> {
        std::vec![T::zero(); size]
    }
}
use weight_array::{make_weight_array, WeightArray};

fn generate_weights<T: FloatCore>(m: i32, t: i32, n: i32, s: i32) -> WeightArray<T> {
    let mut weights = make_weight_array(2 * m as usize + 1);
    for i in 0..=(2 * m) {
        // Cast i and m individually to avoid underflow when i < m
        weights[i as usize] = weight(T::from(i).unwrap() - T::from(m).unwrap(), t, m, n, s);
    }
    weights
}

#[derive(Debug, Clone, Copy)]
pub enum Error {
    NegativeWindowSize,
    EvenWindowSize,
    InputLengthMismatch,
    DataPointOutOfRange,
    OrderTooHigh,
    DerivationOrderTooHigh,
    NonPositiveTimeStep,
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    m: i32,
    t: i32,
    n: i32,
    s: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            m: 3,
            t: 0,
            n: 2,
            s: 0,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ConfigBuilder {
    window_size: Option<i32>,
    data_point: Option<i32>,
    order: Option<i32>,
    derivation_order: Option<i32>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the window size (2m + 1) for the filter. Must be a positive odd integer.
    pub fn window_size(mut self, window_size: i32) -> Self {
        self.window_size = Some(window_size);
        self
    }

    /// Sets the data point (t) for the filter. Must be a non-negative integer less than or equal
    /// to the number of data points (m).
    pub fn data_point(mut self, data_point: i32) -> Self {
        self.data_point = Some(data_point);
        self
    }

    /// Sets the order (n) of the polynomial for the filter. Must be a non-negative integer less
    /// than 3, above which the recursive Gram polynomial calculation may become unstable.
    pub fn order(mut self, order: i32) -> Self {
        self.order = Some(order);
        self
    }

    /// Sets the derivation order (s) for the filter. Must be a non-negative integer less than or
    /// equal to the polynomial order (n).
    pub fn derivation_order(mut self, derivation_order: i32) -> Self {
        self.derivation_order = Some(derivation_order);
        self
    }

    /// Validates the provided parameters and constructs a Config object if all parameters are valid.
    ///
    /// # Errors
    /// Returns an Error if any of the parameters are invalid, such as:
    /// - Negative or even window size
    /// - Data point out of range compared to window size
    /// - Order higher than 3, above which the recursive Gram polynomial calculation may become unstable
    /// - Derivation order higher than the polynomial order
    pub fn build(self) -> Result<Config, Error> {
        let mut candidate = Config::default();

        // 1. Handle Window Size and 'm'
        if let Some(ws) = self.window_size {
            if ws < 0 {
                return Err(Error::NegativeWindowSize);
            }
            if ws % 2 == 0 {
                return Err(Error::EvenWindowSize);
            }
            candidate.m = ws / 2;
        }

        // 2. Handle Data Point 't'
        // t is an offset relative to center. |t| <= m.
        if let Some(t) = self.data_point {
            // Comply with reference: t must be in range [-m, m]
            if t < -candidate.m || t > candidate.m {
                return Err(Error::DataPointOutOfRange);
            }
            candidate.t = t;
        } // 3. Handle Order 'n'

        if let Some(n) = self.order {
            // Mathematical constraint: polynomial order < number of points
            if n < 0 || n > 2 * candidate.m {
                return Err(Error::OrderTooHigh);
            }
            candidate.n = n;
        }

        // 4. Handle Derivation Order 's'
        if let Some(s) = self.derivation_order {
            if s < 0 || s > candidate.n {
                return Err(Error::DerivationOrderTooHigh);
            }
            candidate.s = s;
        }

        Ok(candidate)
    }
}

#[derive(Debug, Clone)]
pub struct SavitzkyGolayFilter<T: FloatCore> {
    weights: WeightArray<T>,
    config: Config,
}

impl<T: FloatCore> SavitzkyGolayFilter<T> {
    pub fn new(config: Config) -> Self {
        let weights = generate_weights(config.m, config.t, config.n, config.s);
        Self { weights, config }
    }

    /// Applies the filter to the input data `v` with a specified time step `dt`. The input data
    /// must have the same length as the number of weights (2m + 1). The output is the filtered
    /// value at the specified data point, which may be a smoothed value or a derivative depending
    /// on the configuration.
    ///
    /// # Arguments
    /// * `v` - A slice of input data values, must have length equal to the number of weights (2m + 1)
    /// * `dt` - The time step between data points, used for scaling derivatives. For smoothing (s=0), this can be set to 1.0.
    ///
    /// # Returns
    /// The filtered value at the specified data point, which may be a smoothed value or a
    /// derivative depending on the configuration. Returns an error if the input data length does
    /// not match the number of weights.
    pub fn filter(&self, v: &[T], dt: T) -> Result<T, Error> {
        if v.len() != self.weights.len() {
            return Err(Error::InputLengthMismatch);
        }

        if dt <= T::zero() {
            return Err(Error::NonPositiveTimeStep);
        }

        Ok(self
            .weights
            .iter()
            .zip(v)
            .fold(T::zero(), |sum, (w, x)| sum + *w * *x)
            / dt.powi(self.config.s))
    }

    pub fn window_size(&self) -> usize {
        self.weights.len()
    }

    pub fn data_point(&self) -> i32 {
        self.config.t
    }

    pub fn order(&self) -> i32 {
        self.config.n
    }

    pub fn derivation_order(&self) -> i32 {
        self.config.s
    }
}

impl Default for SavitzkyGolayFilter<f64> {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_invalid_config() {
        // Negative window size
        let config_neg_window = ConfigBuilder::new().window_size(-5).build();
        assert!(matches!(config_neg_window, Err(Error::NegativeWindowSize)));

        // Even window size
        let config_even_window = ConfigBuilder::new().window_size(6).build();
        assert!(matches!(config_even_window, Err(Error::EvenWindowSize)));

        // Data point out of range
        let config_data_point_out_of_range = ConfigBuilder::new()
            .window_size(7)
            .data_point(4) // t must be in range [-m, m], here m=3
            .build();
        assert!(matches!(
            config_data_point_out_of_range,
            Err(Error::DataPointOutOfRange)
        ));

        // Order too high
        let config_order_too_high = ConfigBuilder::new()
            .window_size(7)
            .order(7) // n must be < number of points (2m + 1), here 2*3+1=7
            .build();
        assert!(matches!(config_order_too_high, Err(Error::OrderTooHigh)));

        // Derivation order too high
        let config_derivation_order_too_high = ConfigBuilder::new()
            .window_size(7)
            .order(2)
            .derivation_order(3) // s must be <= n, here n=2
            .build();
        assert!(matches!(
            config_derivation_order_too_high,
            Err(Error::DerivationOrderTooHigh)
        ));
    }

    #[test]
    fn test_gorry_tables() {
        // Convolution weights for quadratic initial-point smoothing:
        // m=3 (window 7), t=-3 (oldest point), n=2, s=0
        let sg7_gram = [32.0, 15.0, 3.0, -4.0, -6.0, -3.0, 5.0];

        let config = ConfigBuilder::new()
            .window_size(7)
            .data_point(-3) // t = -m
            .order(2)
            .derivation_order(0)
            .build()
            .unwrap();

        let filter = SavitzkyGolayFilter::<f64>::new(config);

        // The Gorry paper weights are often scaled for integer representation.
        // For m=3, n=2, s=0, the common denominator is 42.
        for (i, &expected) in sg7_gram.iter().enumerate() {
            let computed = filter.weights[i] * 42.0;
            assert_relative_eq!(computed, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gorry_derivative() {
        // Convolution weights for quadratic initial-point first derivative:
        // m=3 (window 7), t=-3, n=2, s=1
        let sg7_deriv_gram = [-13.0, -2.0, 5.0, 8.0, 7.0, 2.0, -7.0];

        let config = ConfigBuilder::new()
            .window_size(7)
            .data_point(-3)
            .order(2)
            .derivation_order(1)
            .build()
            .unwrap();

        let filter = SavitzkyGolayFilter::<f64>::new(config);

        // Common denominator for m=3, n=2, s=1 is 28.
        for (i, &expected) in sg7_deriv_gram.iter().enumerate() {
            let computed = filter.weights[i] * 28.0;
            assert_relative_eq!(computed, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_identity() {
        // Filtering a constant signal should return the constant.
        let config = ConfigBuilder::new()
            .window_size(7)
            .data_point(0) // Center
            .order(2)
            .derivation_order(0)
            .build()
            .unwrap();

        let filter = SavitzkyGolayFilter::<f64>::new(config);
        let data = [1.0; 7];
        let res = filter.filter(&data, 1.0).unwrap();
        assert_relative_eq!(res, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_real_time_filter() {
        let config = ConfigBuilder::new()
            .window_size(7)
            .data_point(3) // t = m (Real-time latest point)
            .order(2)
            .derivation_order(0)
            .build()
            .unwrap();

        let filter = SavitzkyGolayFilter::<f64>::new(config);
        let data = [0.1, 0.7, 0.9, 0.7, 0.8, 0.5, -0.3];
        let result = filter.filter(&data, 1.0).unwrap();
        let result_ref = -0.22619047619047616; // Reference value from C++ test
        assert_relative_eq!(result, result_ref, epsilon = 1e-10);
    }

    #[test]
    fn test_real_time_derivative() {
        let t = 3; // Latest point
        let n = 2;

        // 1st order derivative
        let config_s1 = ConfigBuilder::new()
            .window_size(7)
            .data_point(t)
            .order(n)
            .derivation_order(1)
            .build()
            .unwrap();

        let filter_s1 = SavitzkyGolayFilter::<f64>::new(config_s1);
        let filter_s1_dt = SavitzkyGolayFilter::<f64>::new(config_s1);

        // Linear data: y = 0.1 * x. Slopes should be 0.1.
        let data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

        // With dt = 1.0
        let res1 = filter_s1.filter(&data, 1.0).unwrap();
        assert_relative_eq!(res1, 0.1, epsilon = 1e-10);

        // With dt = 0.005
        let dt = 0.005;
        let res1_dt = filter_s1_dt.filter(&data, dt).unwrap();
        assert_relative_eq!(res1_dt, 0.1 / dt, epsilon = 1e-10);

        // 2nd order derivative on linear data should be ~0.
        let config_s2 = ConfigBuilder::new()
            .window_size(7)
            .data_point(t)
            .order(n)
            .derivation_order(2)
            .build()
            .unwrap();
        let filter_s2 = SavitzkyGolayFilter::<f64>::new(config_s2);
        let res2 = filter_s2.filter(&data, 1.0).unwrap();
        assert!(res2.abs() < 1e-6);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_polynomial_derivative() {
        use std::vec::Vec;
        // f(x) = 10x^3 + 2x^2 - 3x - 4
        let a = 10.0;
        let b = 2.0;
        let c = -3.0;
        let d = -4.0;
        let dt = 0.42;

        let m = 50;
        let n = 3;
        let t = 0; // Evaluate at center

        let config_s1 = ConfigBuilder::new()
            .window_size(2 * m + 1)
            .data_point(t)
            .order(n)
            .derivation_order(1)
            .build()
            .unwrap();
        let config_s2 = ConfigBuilder::new()
            .window_size(2 * m + 1)
            .data_point(t)
            .order(n)
            .derivation_order(2)
            .build()
            .unwrap();

        let filter_s1 = SavitzkyGolayFilter::<f64>::new(config_s1);
        let filter_s2 = SavitzkyGolayFilter::<f64>::new(config_s2);

        let mut data = Vec::new();
        for x_idx in 0..(2 * m + 1) {
            let x = x_idx as f64;
            data.push(a * x.powi(3) + b * x.powi(2) + c * x + d);
        }

        // Expected 1st derivative at center (x = m): f'(x) = 30x^2 + 4x - 3
        let x_eval = m as f64;
        let expected_s1 = (3.0 * a * x_eval.powi(2) + 2.0 * b * x_eval + c) / dt;
        let res_s1 = filter_s1.filter(&data, dt).unwrap();

        // Expected 2nd derivative at center: f''(x) = 60x + 4
        let expected_s2 = (6.0 * a * x_eval + 2.0 * b) / dt.powi(2);
        let res_s2 = filter_s2.filter(&data, dt).unwrap();

        assert_relative_eq!(res_s1, expected_s1, epsilon = 1e-10);
        assert_relative_eq!(res_s2, expected_s2, epsilon = 1e-10);
    }

    #[test]
    fn test_wrong_window_size() {
        let config = ConfigBuilder::new().window_size(7).build().unwrap();
        let filter = SavitzkyGolayFilter::<f64>::new(config);

        let short_data = [0.0; 5];
        let result = filter.filter(&short_data, 1.0);

        assert!(result.is_err());
    }
}
