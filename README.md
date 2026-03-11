# Gram Savitzky Golay for Rust

Rust implementation of Savitzky-Golay filtering based on Gram polynomials,
ported from: https://github.com/arntanguy/gram_savitzky_golay.

## Features

- `no_std` by default.
  - In a no-std environment, the crate limits the number of data points to 20
    and stores them in a fixed-size array.
  - In a std environment, the crate can handle an arbitrary number of data
    points using a `Vec`.
- Generic over numeric types, supporting any type that implements the
  `num-traits` crate's `FloatCore` trait, including `f32`, `f64`.
- Very dependency-light: only `num-traits` for numeric traits
- Single file implementation in `src/lib.rs`.
  - Core logic in under 400 lines of code.

## Example

```rust
fn main() -> Result<(), gram_savitzky_golay::Error> {
    // Window size is 2*m+1
    let m = 3;
    let window_size = 2 * m + 1;

    // Polynomial Order
    // At most 3, otherwise the recursive Gram polynomial evaluation will be unstable
    let n = 2;

    // Initial Point Smoothing (i.e. evaluate polynomial at first point in the window)
    // Points are defined in range [-m;m]
    let t = m;

    // Derivation order? 0: no derivation, 1: first derivative, 2: second derivative
    // Must be less than or equal to n, since we can't take the derivative of a polynomial of order
    // n more than n times
    let d = 0;

    let cfg = gram_savitzky_golay::ConfigBuilder::new()
        .window_size(window_size)
        .order(n)
        .data_point(t)
        .derivation_order(d)
        .build()?;

    let filter = gram_savitzky_golay::SavitskyGolayFilter::new(cfg);

    let data = [0.1, 0.7, 0.9, 0.7, 0.8, 0.5, -0.3];
    let result = filter.filter(&data, 1.0)?;
    println!("Input: {data:?} Output: {result}");

    let d = 1;

    let cfg = gram_savitzky_golay::ConfigBuilder::new()
        .window_size(window_size)
        .order(n)
        .data_point(t)
        .derivation_order(d)
        .build()?;

    let first_derivative_filter = gram_savitzky_golay::SavitskyGolayFilter::new(cfg);
    let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    let derivative_result = first_derivative_filter.filter(&values, 1.0)?;
    println!("Input: {values:?} First Derivative: {derivative_result}");

    Ok(())
}
```
