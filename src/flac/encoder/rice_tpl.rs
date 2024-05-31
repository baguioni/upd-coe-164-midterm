use std::ops::{Shl, Shr};
use crate::flac::bitstream;

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
        todo!()
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
        todo!()
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
    fn best_parameters(&self, partition_order: u8, residuals: &Vec <i64>) -> Option <(Vec <u8>, u64)> {
        todo!()
    }

    /// Find the exact total number of bits needed to represent a Rice-encoded
    /// partition of samples
    /// 
    /// A residual `r` can be represented using 1 bit for the unary stop mark,
    /// `rice_param` bits for the truncated binary part of the rice encoding, and
    /// `zigzag(r) >> rice_param` bits for the unary tally marks.
    fn bits_in_partition_exact(rice_param: u8, n_partition_samples: u64, residuals: &Vec<i64>) -> u64 {
        todo!()
    }

    /// Find the total number of bits occupied by this encoding
    /// 
    /// Rice encoding uses `q + 1` bits for the unary-encoded quotient `q` and
    /// `rice_param` bits for the binary remainder
    fn bits_in_partition_sums(rice_param: u8, n_partition_samples: u64, abs_residual_sum: u64) -> u64 {
        todo!()
    }

    /// Encode residuals into Rice encoding
    /// 
    /// To encode a residual into its Rice encoding, it should be first processed
    /// using zigzag encoding so that all of the residuals become nonnegative numbers.
    /// Then, the Rice encoding of each residual is computed.
    /// 
    /// Note that the contents are _not_ ensured to be byte-aligned. Hence, this method returns
    /// the Rice-encoded byte vector containing the number of extra unused bits at the last element.
    pub fn encode(rice_param: u8, residuals: &Vec <i64>) -> RiceEncodedStream {
        let zigzag_residuals = residuals.iter().map(|&num| RiceEncoderOptions::zigzag(num)).collect();

        for residual in zigzag_residuals {
            let res = RiceEncoderOptions::encode_residual(residual, rice_param);
        }
    }

    pub fn encode_residual(residual: u64, rice_param: u8) -> u64 {
        // Calculate K, U, and B
        let K: u64 = RiceEncoderOptions::min_bits_required(rice_param);
        let mut u = residual >> K;
        let B = residual & ((rice_param - 1) as u64);
    
        let mut unary_code = String::new();
        for _ in 0..=u {
            unary_code.push('1');
        }
        unary_code.push('0');
        
        let U = RiceEncoderOptions::unary_to_u64(&unary_code);
        // Concatenate unary and binary codes
        let result = (U << K) | B;
    
        result
    }

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
    
    fn unary_to_u64(unary_code: &str) -> u64 {
        let mut count = 0;
        let mut max_count = 0;
    
        for c in unary_code.chars() {
            if c == '1' {
                count += 1;
            } else {
                max_count = max_count.max(count);
                count = 0;
            }
        }
    
        max_count
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
    pub fn encode_by_partition(&self, residuals: &Vec <i64>)  -> (Vec <RiceEncodedStream>, u8) {
        todo!()
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
        assert_eq!(result, expected_max_partition_order, input_block_size);
    }
}