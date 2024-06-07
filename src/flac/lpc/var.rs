use std::f64;

pub struct VarPredictor;

impl VarPredictor {
    /// Get the autocorrelation of a vector of samples
    ///
    /// The function computes the autocorrelations of the provided vector of
    /// data from `R[0]` until `R[max_lag]`. For example, if `max_lag` is 2, then
    /// the output contains three elements corresponding to R[0] until R[3],
    /// respectively
    /// https://github.com/xiph/flac/blob/cfe3afca9b3f27f0877203570705e072f0981b2e/src/libFLAC/lpc.c#L110
    /// ^ Reference Implementation
    pub fn get_autocorrelation(samples: &Vec <i64>, max_lag: u8) -> Vec <f64> {
        // Initialize autocorrelation vector
        let mut autoc = vec![0.0; max_lag as usize + 1];

        // Iterate over each lag
        for l in 0..=max_lag {
            let mut d = 0.0;
            // Compute autocorrelation for current lag
            for i in l..samples.len() as u8{
                d += samples[i as usize] as f64 * samples[(i - l) as usize] as f64;
            }
            // Store autocorrelation for current lag
            autoc[l as usize] = d;
        }

        autoc
    }

    /// Get the predictor coefficients
    /// 
    /// `autoc` contains the autocorrelation vector where `autoc[i]` corresponds to
    /// the autocorrelation value of lag `i - 1`. `predictor_order` should be
    /// less than `autoc.len()`. The coefficients are computed using the Levinson-Durbin
    /// algorithm.
    /// https://github.com/xiph/flac/blob/cfe3afca9b3f27f0877203570705e072f0981b2e/src/libFLAC/lpc.c#L176
    /// ^ Reference Implementation
    pub fn get_predictor_coeffs(autoc: &Vec<f64>, predictor_order: u8) -> Vec<f64> {
        let mut lpc: Vec<f64> = vec![0.0; predictor_order as usize];
        let mut err = autoc[0];
    
        for i in 0..predictor_order {
            // Sum up this iteration's reflection coefficient.
            let mut r = -autoc[(i + 1) as usize];
            for j in 0..i {
                r -= lpc[j as usize] * autoc[(i - j) as usize];
            }
            r /= err;
    
            // Update LPC coefficients and total error.
            lpc[i as usize] = r;
    
            for j in 0..(i >> 1) {
                let tmp = lpc[j as usize];
                lpc[j as usize] += r * lpc[(i - 1 - j) as usize];
                lpc[(i - 1 - j) as usize] += r * tmp;
            }
            if i & 1 == 1 {
                lpc[(i >> 1) as usize] += lpc[(i >> 1) as usize] * r;
            }
    
            err *= 1.0 - r * r;
        }
    
        // negate FIR filter coeff to get predictor coeff
        for coeff in lpc.iter_mut() {
            *coeff = -*coeff;
        }
    
        lpc
    }

    /// Get a the list of LPC coefficients until some provided predictor order inclusive.
    /// 
    /// For the return value `lpc_list`, `lpc_list[i]` contains a `Vec` of coefficients
    /// for predictor order `i + 1`. The Levinson-Durbin algorithm is used to progressively
    /// compute the LPC coefficients across multiple predictor orders.
    fn build_predictor_coeffs(autoc: &Vec <f64>, max_predictor_order: u8) -> Vec <Vec <f64>> {
        todo!()
    }

    /// Quantize the predictor coefficients and find their shift factor
    /// 
    /// The shift factor `S` is computed from the maximum absolute value of a coefficient
    /// `L_max`. This value is computed as `precision - lg(L_max)` or to
    /// the maximum shift value of 1 << 5 = 31, whichever is smaller. Note that it is
    /// possible for this shift factor to be negative. In that case, the shift value
    /// will still be used in quantizing the coefficients but its effective value
    /// will be zero.
    /// 
    /// Quantization involves converting the provided floating-point coefficients
    /// into integers. Each of the values are rounded up or down depending on
    /// some accummulated rounding error `\epsilon`. Initially, this error is zero.
    /// For each coefficient `L_i`, the coefficient is multiplied (for positive shift)
    /// or divided (for negative shift) by `1 << abs(S)` to get the raw value `L_i_r + \epsilon`.
    /// Then, `L_i_r + \epsilon` is rounded away from zero to get the quantized coefficient.
    /// The new rounding error `\epsilon = L_i_r + \epsilon - round(L_i_r)` is then updated for the
    /// next coefficient.
    pub fn quantize_coeffs(lpc_coefs: &Vec<f64>, mut precision: u8) -> (Vec<i64>, u8) {
        let l_max = lpc_coefs.iter().cloned().fold(f64::MIN, f64::max).abs();
    
        let lg_l_max = l_max.log2();
        let s = if precision as f64 - lg_l_max < 0.0 {
            0
        } else {
            (precision as f64 - lg_l_max).floor() as i8
        }.min(31).max(-31) as u8;
    
        let shift_value = if s > 0 { 1 << s } else { 1 };
        let mut quantized_coefs = Vec::with_capacity(lpc_coefs.len());
        let mut epsilon = 0.0;
    
        for &coef in lpc_coefs {
            let raw_value = if s > 0 {
                coef * (shift_value as f64) + epsilon
            } else {
                coef / (shift_value as f64) + epsilon
            };
            let quantized_value = raw_value.round();
            epsilon = raw_value - quantized_value;
            quantized_coefs.push(quantized_value as i64);
        }
    
        (quantized_coefs, s)
    }

    /// Compute the residuals from a given linear predictor
    /// 
    /// The resulting vector `residual[i]` corresponds to the `i + predictor_order`th
    /// signal. The first `predictor_order` values of the residual are the "warm-up"
    /// samples, or the unencoded samples, equivalent to `&samples[..predictor_order]`.
    /// 
    /// The residuals are computed with the `samples` reversed. For some `i`th residual,
    /// `residual[i] = data[i] - (sum(dot(qlp_coefs, samples[i..(i - predictor_order)])) >> qlp_shift)`.
    pub fn get_residuals(samples: &Vec <i64>, qlp_coefs: &Vec <i64>, predictor_order: u8, qlp_shift: u8) -> Vec <i64> {
        let mut residuals: Vec <i64> = vec![0; samples.len()];
        
        let mut residual_shift = 0;
        let mut rev_qlp_coefs: Vec<_> = qlp_coefs.iter().rev().cloned().collect();
        
        for i in (predictor_order as usize)..samples.len() as usize {
            let mut dot_result = 0;
            for i in 0..(rev_qlp_coefs.len() as i64) {
                dot_result += &rev_qlp_coefs[i as usize] * &samples[residual_shift..(qlp_coefs.len() + residual_shift)].to_vec()[i as usize];
            }
            
            residuals[i] = samples[i] - (dot_result >> qlp_shift);
            residual_shift += 1;
        }
        
        let residual = residuals[(predictor_order as usize)..].to_vec();
        residual
    }

    /// compute the quantized LPC coefficients, precision, and shift for the given
    /// predictor order
    pub fn get_predictor_coeffs_from_samples(samples: &Vec <i64>, predictor_order: u8, bps: u8, block_size: u64) -> (Vec <i64>, u8, u8) {
        todo!()
    }

    /// Get the quantized LPC coefficients, precision, and shift for the best predictor order
    /// for the given sample
    /// 
    /// This function selects the best predictor order by finding the order that yields the
    /// absolute minimum sum of residuals. Note that the maximmum predictor order is 32.
    pub fn get_best_lpc(samples: &Vec <i64>, bps: u8, block_size: u64) -> (Vec <i64>, u8, u8) {
        todo!()
    }

    /// Get the best coefficient precision
    /// 
    /// FLAC uses the bit depth and block size to determine the best coefficient
    /// precision. By default, the precision is 14 bits but can be one of the
    /// following depending on several parameters:
    /// 
    /// | Bit depth | Block size |     Best precision      |
    /// |-----------|------------|-------------------------|
    /// |   < 16    |     any    | max(1, 2 + bit_depth/2) |
    /// |     16    |     192    |           7             |
    /// |     16    |     384    |           8             |
    /// |     16    |     576    |           9             |
    /// |     16    |    1152    |          10             |
    /// |     16    |    2304    |          11             |
    /// |     16    |    4608    |          12             |
    /// |     16    |     any    |          13             |
    /// |   > 16    |     384    |          12             |
    /// |   > 16    |    1152    |          13             |
    /// |   > 16    |     any    |          14             |
    pub fn get_best_precision(bps: u8, block_size: u64) -> u8 {
        if bps < 16 {
            (2 + bps / 2).max(1)
        } else if bps == 16 {
            match block_size {
                192 => 7,
                384 => 8,
                576 => 9,
                1152 => 10,
                2304 => 11,
                4608 => 12,
                _ => 13,
            }
        } else {
            match block_size {
                384 => 12,
                1152 => 13,
                _ => 14,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_ietf_02() {
        let in_vec = vec![
            0, 79, 111, 78,
            8, -61, -90, -68,
            -13, 42, 67, 53,
            13, -27, -46, -38,
            -12, 14, 24, 19,
            6, -4, -5, 0,
        ];

        let out_vec_ans = vec![
            3, -1, -13, -10,
            -6, 2, 8, 8,
            6, 0, -3, -5,
            -4, -1, 1, 1,
            4, 2, 2, 2,
            0,
        ];

        let out_vec = VarPredictor::get_residuals(&in_vec, &vec![7, -6, 2], 3, 2);

        assert_eq!(out_vec_ans, out_vec);
    }

    // https://docs.rs/assert_float_eq/latest/assert_float_eq/macro.assert_f64_near.html
    // ^ Reference Implementation
    macro_rules! assert_f64_near {
        // Explicit steps.
        ($a:expr, $b:expr, $n:expr) => {{
            let (a, b, n) = ($a, $b, $n);
            let r = (a - b).abs() < 10f64.powi(-n);
            assert!(r, "{} is not approximately equal to {} with precision {}", a, b, n);
        }};
        // No explicit steps, use default.
        ($a:expr, $b:expr) => (assert_f64_near!($a, $b, 4));
    }

    #[test]
    fn test_get_predictor_coeffs() {
        // Test input data (autocorrelation coefficients)
        let autoc = vec![24710.0, 18051.0, 7050.0, 632.0];
        let predictor_order = 3;

        // Expected result (adjusted)
        let expected_result = vec![1.27123, -0.85145, 0.28488];

        // Call the function
        let predictor_coeffs = VarPredictor::get_predictor_coeffs(&autoc, predictor_order);

        // Compare with expected result
        assert_eq!(predictor_coeffs.len(), expected_result.len());
        for i in 0..predictor_coeffs.len() {
            assert_f64_near!(predictor_coeffs[i], expected_result[i], 5);
        }
    }
}