use std::ops::{Shl, Shr};
// use crate::flac::bitstream;

/// Represents a Rice encoder
///
/// This encoder is expected to encode `num_samples` residuals from a predictor of
/// order `predictor_order`. Note that Rice encoding in FLAC is only available
/// for LPC and FIXED audio subframes.
pub struct RiceEncoderOptions {
    num_samples: u64,
    predictor_order: u8,
}

/// Represents a Rice-encoded stream
///
/// Rice encoding is _not necessarily_ byte-aligned. The `extra_bits_len`
/// value denotes the number of LSBits in the last byte of the `stream`
/// that are _not_ part of the encoding.
#[derive(Debug)]
pub struct RiceEncodedStream {
    pub stream: Vec <u8>,
    pub param: u8,
    pub extra_bits_len: u8,
}

impl RiceEncoderOptions {
    /// Create a builder to the Rice encoder
    pub fn new(num_samples: u64, predictor_order: u8) -> Self {
        Self {
            num_samples,
            predictor_order,
        }
    }

    /// Get the minimum partition order
    /// 
    /// The default minimum partition order is zero
    fn min_rice_partition_order() -> u8 {
        0
    }

    /// Get the maximum partition order
    /// 
    /// The maximum partition order is computed as the lowest power of two
    /// that makes up the block size, or the index of the least significant
    /// 1 bit in the block size. Note that odd-sized block sizes can only
    /// have a partition order of 0 as the number of partitions should be
    /// a power of two.
    /// https://github.com/xiph/flac/blob/master/src/libFLAC/format.c#L540``
    /// ^ Reference Implementation
    fn max_rice_partition_order(mut block_size: u64) -> u8 {
        // The maximum Rice partition order permitted by the format.
        const FLAC_MAX_RICE_PARTITION_ORDER: u8 = 15; 

        let mut max_rice_partition_order = 0;
        
        // returns the index of the least significant 1 bit
        while block_size & 1 == 0 {
            max_rice_partition_order += 1;
            block_size >>= 1;
        }
        
        // add 1 to get location
        max_rice_partition_order += 1;
        max_rice_partition_order.min(FLAC_MAX_RICE_PARTITION_ORDER)
    }

    /// Compute the best partition order and best Rice parameters for each partition
    /// 
    /// The best partition order is computed based on the order that yields the minimum
    /// total number of bits of the resulting Rice encoding.
    fn best_partition_and_params(&self, residuals: &Vec <i64>) -> (Vec <u8>, u8) {
        let mut best_params = Vec::new();
        let mut best_partition_order = 0;

        let max_order = Self::max_rice_partition_order(self.num_samples);
        let mut min_bits = u64::MAX;

        for partition_order in Self::min_rice_partition_order()..=max_order {
            if let Some((params, bits)) = self.best_parameters(partition_order, residuals) {
                if bits < min_bits {
                    min_bits = bits;
                    best_partition_order = partition_order;
                    best_params = params;
                }
            }
        }

        (best_params, best_partition_order)
    }
    

    /// Compute the best Rice parameters for some partition of the residuals
    /// 
    /// The best Rice parameter `M` can be approximated using the following:
    /// 
    /// `M = log2(abs_r_mean - 1) - log2(n_partition_samples) + 1`.
    /// 
    /// Note that in practice, the sum of the absolute value of the residuals
    /// is used instead of the absolute residual mean `abs_r_mean`. In addition,
    /// Most implementations will bound `M` to be represented by at most 18 bits.
    /// 
    /// Note that only partition order 0 is allowed for odd-length residuals
    /// as the number of partitions should be a power of two.
    /// 
    /// # Errors
    /// Returns `None` if a best parameter cannot be found for any partition. This
    /// arises usually if the predictor order is larger than the amount of residuals
    /// in a partition.
    fn best_parameters(&self, partition_order: u8, residuals: &Vec<i64>) -> Option<(Vec<u8>, u64)> {
        if partition_order > residuals.len() as u8 {
            return None;
        }

        use std::f64::consts::LN_2;

        fn compute_s(Q: i64, N: i64) -> f64 {
            (LN_2 * (Q as f64) / (N as f64)).log2() as f64
        }
    
        fn compute_f(r_low: i64, N: i64, Q: i64) -> i64 {
            (r_low + 2) * N + Q
        }
    
    
        let result = Self::calculate_rice_encodings(residuals);
        
    
        let N = residuals.len() as i64;
        let s = compute_s(result[0] as i64, N);
        
        let mut r_low = s.floor() as i64;
        let mut r_high = s.ceil() as i64;
    
        let mut f_low = compute_f(r_low, N, result[r_low as usize] as i64);
        let mut f_high = compute_f(r_high, N, result[r_high as usize] as i64);
        
        let mut t = 1;
        
        if f_low < f_high {
            let mut temp = r_low;
            r_low = r_high;
            r_high = temp;
            
            temp = f_low;
            f_low = f_high;
            f_high = temp;
            
            t = -1;
        }
    
        while f_low > f_high {
            r_low = r_high;
            f_low = f_high;
            r_high += t;
            f_high = compute_f(r_high, N, result[r_high as usize] as i64);
        }
        
        Some((result, r_low as u64))
    }

    // Helper function for Best Rice Parameter (Donado)
    fn calculate_rice_encodings(residuals: &Vec<i64>) -> Vec<u8> {
        fn unary_rice_encoding(n: i64) -> Vec<i64> {
            if n == 0 {
                return vec![0];
            }
        
            let mut encodings = Vec::new();
            let log2_n = (n as f64).log2().floor() as i64;
        
            for r in 0..=log2_n {
                encodings.push(n / (i64::pow(2, r as u32)));
            }
        
            encodings
        }
    
        let mut rice_encodings = Vec::new();
    
        for &residual in residuals {
            rice_encodings.push(unary_rice_encoding(residual));
        }
    
        let max_length = rice_encodings.iter().map(|v| v.len()).max().unwrap_or(0);
    
        let mut result = vec![0; max_length];
    
        for encoding in rice_encodings {
            for (i, &value) in encoding.iter().enumerate() {
                result[i] += value;
            }
        }
    
        result.iter().map(|&x| x as u8).collect()
    }


    /// Find the exact total number of bits needed to represent a Rice-encoded
    /// partition of samples
    /// 
    /// A residual `r` can be represented using 1 bit for the unary stop mark,
    /// `rice_param` bits for the truncated binary part of the rice encoding, and
    /// `zigzag(r) >> rice_param` bits for the unary tally marks.
    fn bits_in_partition_exact(rice_param: u8, n_partition_samples: u64, residuals: &Vec<i64>) -> u64 {
        let mut total_bits = 0;

        for &residual in residuals {
            let zigzag_encoded = Self::zigzag(residual);
            let unary_tally_bits = zigzag_encoded >> rice_param;
            total_bits += 1 + rice_param as u64 + unary_tally_bits;
        }

        total_bits
    }

    /// Find the total number of bits occupied by this encoding
    /// 
    /// Rice encoding uses `q + 1` bits for the unary-encoded quotient `q` and
    /// `rice_param` bits for the binary remainder
    fn bits_in_partition_sums(rice_param: u8, n_partition_samples: u64, abs_residual_sum: u64) -> u64 {
        // Compute the total number of bits required for the partition
        let mut total_bits = 0;

        // Iterate over each residual in the partition
        for i in 0..n_partition_samples {
            // Average residual value
            let average_residual = abs_residual_sum / n_partition_samples;
            
            // Compute quotient and remainder
            let q = average_residual >> rice_param;
            let r = average_residual & ((1 << rice_param) - 1);

            // Compute the number of bits for this residual
            total_bits += (q + 1) + rice_param as u64;
        }

        total_bits
    }

    /// Encode residuals into Rice encoding
    /// 
    /// To encode a residual into its Rice encoding, it should be first processed
    /// using zigzag encoding so that all of the residuals become nonnegative numbers.
    /// Then, the Rice encoding of each residual is computed.
    /// 
    /// Note that the contents are _not_ ensured to be byte-aligned. Hence, this method returns
    /// the Rice-encoded byte vector containing the number of extra unused bits at the last element.
    pub fn encode(rice_param: u8, residuals: &Vec<i64>) -> RiceEncodedStream {
        let mut bit_buffer: u64 = 0;
        let mut bit_count: u8 = 0;
        let mut stream = Vec::new();
    
        for &residual in residuals {
            let zigzag_res = Self::zigzag(residual);
            let (encoded_bits, num_bits) = Self::encode_residual(zigzag_res, rice_param);
            bit_buffer |= (encoded_bits as u64) << bit_count;
            bit_count += num_bits;
    
            while bit_count >= 8 {
                stream.push((bit_buffer & 0xFF) as u8);
                bit_buffer >>= 8;
                bit_count -= 8;
            }
        }
    
        if bit_count > 0 {
            stream.push(bit_buffer as u8);
        }
    
        RiceEncodedStream {
            stream: stream,
            param: rice_param,
            extra_bits_len: bit_count,
        }
    }
    
    pub fn encode_residual(residual: u64, rice_param: u8) -> (u64, u8) {
        fn min_bits_required(num: u8) -> u64 {
            if num == 0 {
                return 1;
            }
    
            let mut count = 0;
            let mut n = num as u64;
    
            while n != 0 {
                count += 1;
                n >>= 1;
            }
    
            count
        }
    
        // Calculate K, U, and B
        let K: u64 = min_bits_required(rice_param);
        let u = residual >> K;
        let B = residual & ((1 << K) - 1);
    
        let unary_code_len = u + 1; // unary representation length
        let U = unary_code_len as u64;
        let result = (U << K) | B;
        let total_bits = unary_code_len + K; // total bits
    
        (result, total_bits as u8)
    }



    /// Encode residuals into a partitioned Rice-encoded stream
    /// 
    /// This method computes the Rice encoding of a stream of residuals by first partitioning
    /// the residual into groups. Each group is then found its best Rice parameter and
    /// each residual in the group is then encoded using the parameter.
    /// 
    /// The method returns each Rice-encoded group in chronological order and the partition order,
    /// respectively. The number of elemenets in the vector of Rice-encoded groups should be less than
    /// or equal to `2^partition order`.
    /// 
    /// Note that each of the contents are _not_ ensured to be byte-aligned. Hence, this method
    /// returns the Rice-encoded byte stream and the number of extra unused bits at the last byte
    /// of the stream, respectively.
    pub fn encode_by_partition(&self, residuals: &Vec<i64>) -> (Vec<RiceEncodedStream>, u8) {
        let (best_params, best_partition_order) = self.best_partition_and_params(residuals);
        let num_partitions = 1 << best_partition_order;

        let mut partitions = vec![Vec::new(); num_partitions as usize];
        let partition_size = self.num_samples as usize / num_partitions as usize;

        for (i, &residual) in residuals.iter().enumerate() {
            let partition_index = i / partition_size;
            partitions[partition_index].push(residual);
        }

        let mut encoded_partitions = Vec::new();

        for (partition, &param) in partitions.iter().zip(&best_params) {
            let encoded_stream = Self::encode(param, partition);
            encoded_partitions.push(encoded_stream);
        }

        (encoded_partitions, best_partition_order)
    }

    /// Convert an integer into its zigzag encoding. With this encoding, all
    /// positive numbers are even and all negative numbers are odd.
    /// https://docs.rs/residua-zigzag/latest/zigzag/#zigzag-encoding
    /// ^ Reference implementation
    pub fn zigzag(num: i64) -> u64 {
        ((num << 1) ^ (num >> 63)) as u64
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_sample_ietf_02() {
        let in_vec = vec![
            3194, -1297, 1228, -943,
            952, -696, 768, -524,
            599, -401, -13172, -316,
            274, -267, 134,
        ];

        let out_vec_ans = vec![
            0x11, 0xe8, 0xa2, 0x14,
            0xcc, 0x7a, 0xef, 0xb8,
            0x6b, 0x7f, 0x00, 0x60,
            0xbe, 0x57, 0x59, 0x08,
            0x00, 0x77, 0x3d, 0x3b,
            0xd1, 0x25, 0x0a, 0xc8,
            0x60,
        ];

        let rice_enc_stream = RiceEncoderOptions::encode(11, &in_vec);

        assert_eq!(rice_enc_stream.stream, out_vec_ans);
        assert_eq!(rice_enc_stream.extra_bits_len, 3);
    }

    #[test]
    fn encode_sample_ietf_03() {
        let in_vec = vec![
            3, -1, -13,
        ];

        let out_vec_ans = vec![
            0xe9, 0x12,
        ];

        let rice_enc_stream = RiceEncoderOptions::encode(3, &in_vec);

        assert_eq!(rice_enc_stream.stream, out_vec_ans);
        assert_eq!(rice_enc_stream.extra_bits_len, 1);
    }

    #[test]
    fn max_rice_partition_order_test() {
        let input_block_size = 192;
        let expected_max_partition_order = 7;

        let result = RiceEncoderOptions::max_rice_partition_order(input_block_size);
        assert_eq!(result, expected_max_partition_order);
    }
}