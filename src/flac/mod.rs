pub mod encoder;
pub mod lpc;
pub mod bitstream;

use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::windows::fs::FileExt;
use std::path::Path;

use byteorder::{BigEndian, WriteBytesExt};
use md5::{Md5, Digest};

use crate::wav::PCMWaveInfo;

use encoder::crc::CrcOptions;

use self::bitstream::BitstreamWriter;
use self::encoder::rice::{RiceEncodedStream, RiceEncoderOptions};
use self::encoder::utf8::Utf8Encoder;
use self::lpc::fixed::FixedPredictor;
use self::lpc::var::VarPredictor;

#[derive(Debug)]
pub enum FlacWriterError {
    InvalidFormatError,
    DataAlignmentError,
    WriteError,
    ReadError,
}

pub struct FlacFrame {
    is_variable_blocksize: bool,
    block_size: u16,
    sample_rate: FlacFrameHeaderValueOption <u64>,
    num_channels: u8,
    bit_depth: FlacFrameHeaderValueOption <u8>,
    frame_index: u64,
    subframes: Vec <FlacSubframe>,
}

pub struct FlacSubframe {
    subframe_type: FlacSubframeType,
    bit_depth: u8,
}

pub enum FlacFrameHeaderValueOption <T> {
    Streaminfo(T),
    InFrame(T),
}

pub enum FlacSubframeType {
    None,
    Constant {value: i64},
    Fixed {order: u8},
    Lpc {
        order: u8,
        precision: u8,
        shift: u8, 
        qlp_coefs: Option <Vec <i64>>,
    },
    Verbatim,
}

pub struct FlacWriter;

impl From <io::Error> for FlacWriterError {
    fn from(_: io::Error) -> Self {
        FlacWriterError::WriteError
    }
}

impl fmt::Display for FlacWriterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error: {:?}", self)
    }
}

impl FlacWriter {
    /// Create a FLAC file from a PCM Wave file
    pub fn write_from_wave(wav: PCMWaveInfo, file_path: &str) -> Result <(), FlacWriterError> {
        todo!()
    }

    /// Determine the best block size given a predictor order
    /// 
    /// By default, the block size is 4096 unless the maximum
    /// LPC order is 0, which defaults to a block size of 1152.
    fn best_block_size(max_lpc_order: u64) -> u16 {
        if max_lpc_order == 0 {
            1152
        } else {
            4096
        }
    }
}

impl FlacFrame {
    /// Create a new FLAC audio frame
    /// 
    /// A FLAC audio frame contains a slice of an audio file. Each frame can
    /// be compressed preferably using most optimal compression scheme detected.
    pub fn new(block_size: u16, sample_rate: FlacFrameHeaderValueOption <u64>, num_channels: u8, bit_depth: FlacFrameHeaderValueOption <u8>, frame_index: u64) -> FlacFrame {
        todo!()
    }

    /// Convert this audio frame into a vector of bytes
    /// 
    /// An audio frame is ensured to be byte-aligned (i.e. necessary "0" padding
    /// bits have been appended).
    pub fn build_bytes(&mut self, sample_block: &Vec <Vec <i64>>) -> Vec <u8> {
        todo!()
    }

    /// Determine the block size type of this audio frame
    /// 
    /// FLAC encodes the block size depending on the value of the raw
    /// block size as shown on the table below. Most common block sizes
    /// have fixed types for easier processing, but any block sizes from
    /// 0 until 65535 inclusive are supported.
    /// 
    /// |      Block size     | Type (bin)       |
    /// |---------------------|------------------|
    /// |                 192 |             0001 |
    /// |                 576 |             0010 |
    /// |                1152 |             0011 |
    /// |                2304 |             0100 |
    /// |                4608 |             0101 |
    /// |                 256 |             1000 |
    /// |                 512 |             1001 |
    /// |                1024 |             1010 |
    /// |                2048 |             1011 |
    /// |                4096 |             1100 |
    /// |                8192 |             1101 |
    /// |               16384 |             1110 |
    /// |               32768 |             1111 |
    /// | 576 * (2 ^ (n - 2)) |     n \in [2, 5] |
    /// | 256 * (2 ^ (n - 8)) |    n \in [8, 15] |
    /// |               < 256 |             0110 |
    /// |             < 65536 |             0111 |
    fn block_size_type(&self) -> u8 {
        match self.block_size {
            192 => 0b0001,
            576 => 0b0010,
            1152 => 0b0011,
            2304 => 0b0100,
            4608 => 0b0101,
            256 => 0b1000,
            512 => 0b1001,
            1024 => 0b1010,
            2048 => 0b1011,
            4096 => 0b1100,
            8192 => 0b1101,
            16384 => 0b1110,
            32768 => 0b1111,
            n if (2..=5).contains(&(n / 576)) && (n % 576 == 0) => ((n / 576) as u8 - 2) | 0b0010,
            n if (8..=15).contains(&(n / 256)) && (n % 256 == 0) => ((n / 256) as u8 - 8) | 0b1000,
            n if n < 256 => 0b0110,
            n if n < 65536 => 0b0111,
            _ => 0b0111,
        }
    }

    /// Determine the sample rate type of this audio frame
    /// 
    /// FLAC encodes the sample rate depending on its value as shown on the table below.
    /// Most common sample rates have fixed types for easier processing, but any
    /// sample rate from 0 until 655350 Hz inclusive are supported. It is also possible
    /// to get the sample rate from the mandatory STREAMINFO header.
    /// 
    /// |  Sample rate (kHz)  | Type (bin)       |
    /// |---------------------|------------------|
    /// |     from STREAMINFO |             0000 |
    /// |               88.2  |             0001 |
    /// |              176.4  |             0010 |
    /// |              192    |             0011 |
    /// |                8    |             0100 |
    /// |               16    |             0101 |
    /// |               22.05 |             0110 |
    /// |               24    |             0111 |
    /// |               32    |             1000 |
    /// |               44.1  |             1001 |
    /// |               48    |             1010 |
    /// |               96    |             1011 |
    /// |            < 256    |             1100 |
    /// |          < 65535    |             1101 |
    /// |         < 655350    |             1110 |
    /// |            invalid  |             1111 |
    fn sample_rate_type(&self) -> u8 {
        match self.sample_rate {
            FlacFrameHeaderValueOption::Streaminfo(_) => 0b0000,
            FlacFrameHeaderValueOption::InFrame(sample_rate) => match sample_rate {
                88200 => 0b0001,
                176400 => 0b0010,
                192000 => 0b0011,
                8000 => 0b0100,
                16000 => 0b0101,
                22050 => 0b0110,
                24000 => 0b0111,
                32000 => 0b1000,
                44100 => 0b1001,
                48000 => 0b1010,
                96000 => 0b1011,
                n if n < 256000 => 0b1100,
                n if n < 655350 => 0b1101,
                _ => 0b1110, // Assume invalid for rates >= 655350
            },
        }
    }

    /// Determine the bit depth type of this audio frame
    /// 
    /// FLAC encodes the bit depth depending on its value as shown on the table below.
    /// Most common bit depths have fixed types for easier processing, but any
    /// bit depth from 4 until 32 bits inclusive are supported. It is also possible
    /// to get the bit depth from the mandatory STREAMINFO header.
    /// 
    /// |  Bit depth (bits)  | Type (bin) |
    /// |--------------------|------------|
    /// |   from STREAMINFO  |        000 |
    /// |                 8  |        001 |
    /// |                12  |        010 |
    /// |           invalid  |        011 |
    /// |                16  |        100 |
    /// |                20  |        101 |
    /// |                24  |        110 |
    /// |                32  |        111 |
    fn bit_depth_type(&self) -> u8 {
        match self.bit_depth {
            FlacFrameHeaderValueOption::Streaminfo(_) => 0b000,
            FlacFrameHeaderValueOption::InFrame(bit_depth) => match bit_depth {
                8 => 0b001,
                12 => 0b010,
                16 => 0b100,
                20 => 0b101,
                24 => 0b110,
                32 => 0b111,
                _ => 0b011, // Invalid bit depth
            },
        }
    }

    /// Build the header bytes of this audio frame
    /// 
    /// An audio frame header is ensured to be byte-aligned (i.e. necessary "0" padding
    /// bits have been appended).
    fn build_header_bytes(&self) -> Vec <u8> {
        todo!()
    }
}

impl FlacSubframe {
    /// Create a new VERBATIM audio frame
    pub fn new_verbatim(bit_depth: u8) -> Self {
        todo!()
    }

    /// Create a new CONSTANT audio frame
    pub fn new_constant(bit_depth: u8, sample_value: i64) -> Self {
        todo!()
    }

    /// Create a new FIXED audio frame that autodetects the best
    /// predictor order for a given block of samples
    pub fn new_fixed(bit_depth: u8, samples: &Vec <i64>) -> Option <Self> {
        todo!()
    }

    /// Create a new FIXED audio frame from some predictor order
    pub fn new_fixed_by_order(bit_depth: u8, order: u8) -> Self {
        todo!()
    }

    /// Create a new LPC audio frame that autodetects the best
    /// predictor order for a given block of samples
    pub fn new_variable(bit_depth: u8, block_size: u64, samples: &Vec <i64>) -> Self {
        todo!()
    }

    /// Create a new LPC audio frame from some predictor order
    pub fn new_variable_by_order(bit_depth: u8, block_size: u64, order: u8, samples: &Vec <i64>) -> Self {
        todo!()
    }

    /// Get the residuals of a given block of samples and encode them into a Rice-encoded
    /// byte stream.
    /// 
    /// Note that the contents are _not_ ensured to be byte-aligned. Hence, this method returns
    /// the Rice-encoded byte stream and the number of extra unused bits at the last byte
    /// of the stream, respectively.
    fn get_encoded_residuals(&self, samples: &Vec <i64>) -> Option <(Vec <RiceEncodedStream>, u8)> {
        todo!()
    }

    /// Compute the number of wasted bits in a block of samples
    /// 
    /// The "wasted bits" as defined in FLAC are the maximum number of
    /// LSBits whose values are 0 for all samples in a block. Instead of encoding samples
    /// as is, each sample can be shifted by the number of wasted bits to the right first
    /// before being encoded through one of the four subframe types.
    fn get_wasted_shift(&self, samples: &Vec <i64>) -> u64 {
        todo!()
    }

    /// Build the header bytes of this audio subframe
    /// 
    /// An audio subframe header is _not_ ensured to be byte-aligned. Hence,
    /// this method returns the header bytes and the number of extra unused
    /// bits at the last byte of the stream, respectively.
    fn build_header_bytes(&self, samples: &Vec <i64>) -> (Vec <u8>, u8) {
        todo!()
    }

    /// Convert this audio subframe into a vector of bytes. This includes the
    /// header and the contents defined by one of the four subframe types.
    /// 
    /// An audio subframe is _not_ ensured to be byte-aligned. Hence,
    /// this method returns the bytes and the number of extra unused
    /// bits at the last byte of the stream, respectively.
    pub fn build_bytes(&self, samples: &Vec <i64>) -> (Vec <u8>, u8) {
        todo!()
    }
}

impl <T> FlacFrameHeaderValueOption <T>  {
    pub fn value(&self) -> &T {
        todo!()
    }
}