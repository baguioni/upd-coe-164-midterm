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
        let mut err: f64 = autoc[0];
    
        for i in 0..predictor_order as usize {
            // Sum up this iteration's reflection coefficient.
            let mut r = autoc[(i + 1)];
            for j in 0..i {
                r -= lpc[j] * autoc[(i - j)];
            }
            r /= err;
            
            let mut lpc_clone = lpc.clone();
            
            for j in 0..i {
                lpc_clone[j] -= r * lpc[i - j - 1];
            }
            
            lpc_clone[i] = r;
    
            lpc = lpc_clone;
            err *= 1.0 - r * r;
        }

        lpc
    }

    /// Get a the list of LPC coefficients until some provided predictor order inclusive.
    /// 
    /// For the return value `lpc_list`, `lpc_list[i]` contains a `Vec` of coefficients
    /// for predictor order `i + 1`. The Levinson-Durbin algorithm is used to progressively
    /// compute the LPC coefficients across multiple predictor orders.
    fn build_predictor_coeffs(autoc: &Vec <f64>, max_predictor_order: u8) -> Vec <Vec <f64>> {
        let mut lpc_list : Vec<Vec<f64>> = Vec::new();
        for i in 0..=max_predictor_order {
            let lpc_element = VarPredictor::get_predictor_coeffs(autoc, i);
            lpc_list.push(lpc_element);
        }
        
        lpc_list.retain(|v| !v.is_empty());

        lpc_list
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
        use std::f64;
    
        // Step 1: Compute L_max
        let l_max = lpc_coefs.iter().cloned().fold(f64::MIN, f64::max).abs();
    
        // Step 2: Determine shift factor S
        let lg_l_max = l_max.log2();
        let s = if precision as f64 - lg_l_max < 0.0 {
            0
        } else {
            (precision as f64 - lg_l_max).floor() as i8
        }.min(31).max(-31) as u8;
    
        // Step 3: Quantize the coefficients
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
        let rev_qlp_coefs: Vec<_> = qlp_coefs.iter().rev().cloned().collect();
        
        
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
        let autocorrelation: Vec<f64> = VarPredictor::get_autocorrelation(samples, predictor_order);
        let predictor_coefs: Vec<f64> = VarPredictor::get_predictor_coeffs(&autocorrelation, predictor_order);
        let best_precision: u8 = VarPredictor::get_best_precision(bps, block_size);
        let quantized_coefs: (Vec<i64>, u8) = VarPredictor::quantize_coeffs(&predictor_coefs, best_precision);
        
        (quantized_coefs.0, best_precision, quantized_coefs.1)
    }

    /// Get the quantized LPC coefficients, precision, and shift for the best predictor order
    /// for the given sample
    /// 
    /// This function selects the best predictor order by finding the order that yields the
    /// absolute minimum sum of residuals. Note that the maximmum predictor order is 32.
    pub fn get_best_lpc(samples: &Vec<i64>, bps: u8, block_size: u64) -> (Vec<i64>, u8, u8) {
        let max_predictor_order = 32;
        let mut best_order = 0;
        let mut best_residual_sum = f64::INFINITY;
        let mut best_quantized_coefs = Vec::new();
        let mut best_precision = 0;
        let mut best_shift = 0;

        for order in 1..=max_predictor_order {
            let autocorrelation = VarPredictor::get_autocorrelation(samples, order);
            let predictor_coefs = VarPredictor::get_predictor_coeffs(&autocorrelation, order);
            let precision = VarPredictor::get_best_precision(bps, block_size);
            let (quantized_coefs, shift) = VarPredictor::quantize_coeffs(&predictor_coefs, precision);

            let residuals = VarPredictor::get_residuals(samples, &quantized_coefs, order, shift);
            let residual_sum: f64 = residuals.iter().map(|&x| (x as f64).abs()).sum();

            if residual_sum < best_residual_sum {
                best_residual_sum = residual_sum;
                best_order = order;
                best_quantized_coefs = quantized_coefs;
                best_precision = precision;
                best_shift = shift;
            }
        }

        (best_quantized_coefs, best_precision, best_shift)
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

    #[test]
    fn sample_autocorrelation() {
        let in_vec = vec![32, 84, 192, 78, 61];
        let out_vec = VarPredictor::get_autocorrelation(&in_vec, 4);
        assert_eq!(out_vec, vec![54749.0, 38550.0, 24408.0, 7620.0, 1952.0]);
    }
    
    #[test]
    fn sample_get_predictor_coefs() {
        let in_vec = vec![32.0, 84.0, 192.0, 78.0, 61.0];
        let out_vec = VarPredictor::get_predictor_coeffs(&vec![54749.0, 38550.0, 24408.0, 7620.0, 1952.0], 4);
        assert_eq!(out_vec, vec![0.7903804482322525, 0.09395406942446108, -0.39168694991013925, 0.15955728829739096]);
    }

    #[test]
    fn sample_build_predictor_coefs() {
        let in_vec = vec![32, 84, 192, 78, 61];
        let autoc = VarPredictor::get_autocorrelation(&in_vec, 4);
        let out_vec = VarPredictor::build_predictor_coeffs(&autoc, 4);
        assert_eq!(out_vec, vec![vec![0.7041224497251092], vec![0.7739075411204036, -0.09910931177175031], 
            vec![0.7468988870582981, 0.11179116448535134, -0.2725137888587779], 
            vec![0.7903804482322525, 0.09395406942446108, -0.39168694991013925, 0.15955728829739096]]);
    }

    #[test]
    fn sample_quantize_coefs() {
        let lpc = vec![0.7903804482322525, 0.09395406942446108, -0.39168694991013925, 0.15955728829739096];
        let out = VarPredictor::quantize_coeffs(&lpc, 4);
        assert_eq!(out, (vec![13, 1, -6, 2], 4));
    }

    #[test]
    fn sample_get_best_precision() {
        let out = VarPredictor::get_best_precision(23, 9879);
        assert_eq!(out, 14);
    }
}