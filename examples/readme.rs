fn main() -> Result<(), gram_savitzky_golay::Error> {
    // Window size is 2*m+1
    let m = 3;
    let window_size = 2 * m + 1;

    // Polynomial Order
    // At most 3, otherwise the recursive Gram polynomial evaluation will be unstable
    let n = 2;

    // Initial Point Smoothing (ie evaluate polynomial at first point in the window)
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

    let filter = gram_savitzky_golay::SavitzkyGolayFilter::new(cfg);

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

    let first_derivative_filter = gram_savitzky_golay::SavitzkyGolayFilter::new(cfg);
    let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    let derivative_result = first_derivative_filter.filter(&values, 1.0)?;
    println!("Input: {values:?} First Derivative: {derivative_result}");

    Ok(())
}
