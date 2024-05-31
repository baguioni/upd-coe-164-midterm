pub struct FixedPredictor;

impl FixedPredictor {
    /// Get order that yields the least sum of residuals
    /// 
    /// The predictor orders are from 0 to 4 inclusive and is retrieved
    /// by finding the predictor that yields the *minimum* absolute
    /// sum of residuals for the given `data` and derived predictor.
    pub fn best_predictor_order(data: &Vec<i64>) -> Option<u8> {
        let res_0 = FixedPredictor::get_residuals(data, 0);
        let res_1 = FixedPredictor::get_residuals(data, 1);
        let res_2 = FixedPredictor::get_residuals(data, 2);
        let res_3 = FixedPredictor::get_residuals(data, 3);
        let res_4 = FixedPredictor::get_residuals(data, 4);
    
        let sum_0 = FixedPredictor::calculate_sum(res_0);
        let sum_1 = FixedPredictor::calculate_sum(res_1);
        let sum_2 = FixedPredictor::calculate_sum(res_2);
        let sum_3 = FixedPredictor::calculate_sum(res_3);
        let sum_4 = FixedPredictor::calculate_sum(res_4);
    
        let min_predictor_order = vec![sum_0, sum_1, sum_2, sum_3, sum_4]
            .into_iter()
            .enumerate()
            .filter_map(|(index, sum)| sum.map(|value| (index, value)))
            .min_by_key(|&(_, sum)| sum)
            .map(|(index, _)| index as u8);  // Convert usize to u8 here
    
        min_predictor_order
    }
    

    /// Get residuals of a fixed predictor order 
    /// 
    /// The predictor orders are from 0 to 4 inclusive and corresponds
    /// to one of the five "fixed" predictor orders written in the FLAC
    /// specification. The predictor orders are defined as follows:
    /// 
    /// 0: r[i] = 0
    /// 1: r[i] = data[i - 1]
    /// 2: r[i] = 2 * data[i - 1] - data[i - 2]
    /// 3: r[i] = 3 * data[i - 1] - 3 * data[i - 2] + data[i - 3]
    /// 4: r[i] = 4 * data[i - 1] - 6 * data[i - 2] + 4 data[i - 3] - data[i - 4]
    /// 
    /// This function returns a vector with each element containing data[i] - r[i].
    /// 
    /// # Errors
    /// `None` is returned if an error occurs in the function. This includes whether
    /// the predictor order provided is not within 0 and 4 inclusive and whether the
    /// size of `data` is less than the predictor order.
    /// https://github.com/xiph/flac/blob/master/src/libFLAC/fixed.c#L470
    /// ^ Reference Implementation
    pub fn get_residuals(data: &Vec <i64>, predictor_order: u8) -> Option <Vec <i64>> {
        let data_len = data.len();

        if data_len == 0 || predictor_order > 4 {
            return None;
        }

        let mut residual = vec![0; data_len];
    
        match predictor_order {
            0 => {
                residual.copy_from_slice(&data);
            }
            1 => {
                for i in 1..data_len {
                    residual[i] = data[i] - data[i - 1];
                }
            }
            2 => {
                if data_len < 2 {
                    return None;
                }
                for i in 2..data_len {
                    residual[i] = data[i] - 2 * data[i - 1] + data[i - 2];
                }
            }
            3 => {
                if data_len < 3 {
                    return None;
                }
                for i in 3..data_len {
                    residual[i] = data[i] - 3 * data[i - 1] + 3 * data[i - 2] - data[i - 3];
                }
            }
            4 => {
                if data_len < 4 {
                    return None;
                }
                for i in 4..data_len {
                    residual[i] = data[i] - 4 * data[i - 1] + 6 * data[i - 2] - 4 * data[i - 3] + data[i - 4];
                }
            }
            _ => {
                return None;
            }
        }
    
        let index: usize = predictor_order.into();
        Some(residual[index..].to_vec())
    }

    fn calculate_sum(vector: Option<Vec<i64>>) -> Option<i64> {
        match vector {
            Some(vec) => {
                let sum: i64 = vec.iter().sum();
                Some(sum)
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_ietf_02a() {
        let in_vec = vec![
            4302, 7496, 6199, 7427,
            6484, 7436, 6740, 7508,
            6984, 7583, 7182, -5990,
            -6306, -6032, -6299, -6165,
        ];

        let out_vec_ans = vec![
            3194, -1297, 1228,
            -943, 952, -696, 768,
            -524, 599, -401, -13172,
            -316, 274, -267, 134,
        ];

        let ans = FixedPredictor::get_residuals(&in_vec, 1);

        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), out_vec_ans);
    }
}