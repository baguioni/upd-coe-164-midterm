/// Represents a kind of CRC encoding
/// 
/// This struct is used to configure the type of CRC encoding to use.
/// For example, if the generator polynomial for a CRC8 encoding is:
/// 
/// `x^8 + x^2 + x^1 + 1`
/// 
/// Then, the value of `poly` should be 0b0000_0111 (note the missing
/// MSB `1` bit) and `poly_len` should be `u8`.
pub struct CrcOptions <T> {
    poly: T,
    poly_len: T,
}


impl <T> CrcOptions <T> {
    /// Create a builder to the CRC encoder
    pub fn new(poly: T, poly_len: T) -> Self {
        Self {
            poly,
            poly_len
        }
    }
}

// http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html#ch41
// ^ Reference implementation
impl CrcOptions <u8> {
    /// Encode data using CRC8 encoding
    /// 
    /// This method is available only if `CrcOptions` is of type `u8`.
    pub fn build_crc8(&self, data: &Vec <u8>) -> u8 {
        let mut crc: u8 = 0;

        for &curr_byte in data {
            crc ^= curr_byte;

            for _ in 0..self.poly_len {
                if (crc & 0x80) != 0 {
                    crc = (crc << 1) ^ self.poly;
                } else {
                    crc <<= 1;
                }
            }
        }

        crc & ((1 << (self.poly_len as usize)) - 1) as u8 // Mask CRC to poly_len bits
    }
}

impl CrcOptions <u16> {
    /// Encode data using CRC16 encoding
    /// 
    /// This method is available only if `CrcOptions` is of type `u16`.
    pub fn build_crc16(&self, data: &Vec <u16>) -> u16 {
        let mut crc = 0u16;
    
        for &b in data {
            crc ^= u16::from(b) << (self.poly_len - 8);
    
            for _ in 0..8 {
                if crc & (1 << (self.poly_len - 1)) != 0 {
                    crc = (crc << 1) ^ self.poly;
                } else {
                    crc <<= 1;
                }
            }
        }
    
        crc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_crc8_01() {
        let in_vec = vec![
            0x10,
        ];
        let ans = CrcOptions::new(0b0000_0111u8, 8)
            .build_crc8(&in_vec);

        assert_eq!(ans, 0x70);
    }
    
    #[test]
    fn sample_crc8_02() {
        let in_vec = vec![
            0x3a, 0x9f, 0x7c,
        ];
        let ans = CrcOptions::new(0b0010_0101u8, 8)
            .build_crc8(&in_vec);

        assert_eq!(ans, 0xba);
    }

    #[test]
    fn sample_crc8_ietf_01() {
        let in_vec = vec![
            0xff, 0xf8, 0x69, 0x18,
            0x00, 0x00,
        ];
        let ans = CrcOptions::new(0b0000_0111u8, 8)
            .build_crc8(&in_vec);

        assert_eq!(ans, 0xbf);
    }
    
    #[test]
    fn sample_crc8_ietf_02() {
        let in_vec = vec![
            0x15, 0xa2, 0xf3, 0xe4,
            0x75, 0xc6, 0x36, 0x55, 
            0x2e,
        ];
        let ans = CrcOptions::new(0b0000_1000u8, 8)
            .build_crc8(&in_vec);

        assert_eq!(ans, 0xa8);
    }

    #[test]
    fn sample_crc16_01() {
        let in_vec = vec![
            0x10, 0x00,
        ];
        let ans = CrcOptions::new(0b1000_0000_0000_0101u16, 16)
            .build_crc16(&in_vec);

        assert_eq!(ans, 0xe003);
    }
    
    #[test]
    fn sample_crc16_02() {
        let in_vec = vec![
            0x10, 0x12, 0x0f, 0x1d,
        ];
        let ans = CrcOptions::new(0b0101_1010_1010_0101u16, 16)
            .build_crc16(&in_vec);

        assert_eq!(ans, 0x1fee);
    }

    #[test]
    fn sample_crc16_ietf_01() {
        let in_vec = vec![
            0xff, 0xf8, 0x69, 0x18,
            0x00, 0x00, 0xbf, 0x03,
            0x58, 0xfd, 0x03, 0x12,
            0x8b,
        ];
        let ans = CrcOptions::new(0b1000_0000_0000_0101u16, 16)
            .build_crc16(&in_vec);

        assert_eq!(ans, 0xaa9a);
    }
    
    #[test]
    fn sample_crc16_ietf_02() {
        let in_vec = vec![
           0xa5, 0xb8, 0xc9, 0xd1,
           0x2e, 0x4f, 0x18, 0x15,
           0x23, 0x01, 0x17, 0x8a, 
           0x9b, 0x4c, 0x6d, 0x7e, 
           0x5f,
        ];
        let ans = CrcOptions::new(0b1111_0000_0011_1100u16, 16)
            .build_crc16(&in_vec);

        assert_eq!(ans, 0xfe88);
    }
}