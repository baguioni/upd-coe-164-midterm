use core::fmt;
use std::fs::File;
use std::path::Path;
use std::error;
use std::io::{self, Read, Seek, SeekFrom, BufReader};
use byteorder::{ByteOrder, LittleEndian};

// ===== [IMPORTANT] WAV COMPLIANT VERSION NOTES =====
// THIS VERSION OF WAV ONLY SUPPORTS RIFF, RIFX SUPPORT TO BE ADDED LATER
// ===== !!!!!!!!! =====

// DELTA VERSION, POLISHED FOR PRODUCTION

/// Represents a PCM WAV file
pub struct PCMWaveInfo {
    pub riff_header: RiffChunk,
    pub fmt_header: PCMWaveFormatChunk,
    pub data_chunks: Vec<PCMWaveDataChunk>,
}

/// Represents a RIFF chunk from a WAV file
pub struct RiffChunk {
    pub file_size: u32,
    pub is_big_endian: bool,
}

/// Represents a format chunk from a WAV file
#[derive(Clone, Copy)]
pub struct PCMWaveFormatChunk {
    pub num_channels: u16,
    pub samp_rate: u32,
    pub bps: u16,
}

/// Represents a data chunk from a WAV file
pub struct PCMWaveDataChunk {
    pub size_bytes: u32,
    pub format: PCMWaveFormatChunk,
    pub data_buf: BufReader<File>,
}

/// Represents an iterator to a data chunk from a WAV file
pub struct PCMWaveDataChunkWindow {
    chunk_size: usize,
    data_chunk: PCMWaveDataChunk,
}

/// Represents a WAV reader
pub struct WaveReader;

/// Represents an error in the WAV reader
#[derive(Debug)]
pub enum WaveReaderError {
    NotRiffError,
    NotWaveError,
    NotPCMError,
    ChunkTypeError,
    DataAlignmentError,
    ReadError,
}

impl WaveReader {
    /// Open a PCM WAV file
    pub fn open_pcm(file_path: &str) -> Result<PCMWaveInfo, WaveReaderError> {
        let mut wav_file = File::open(file_path).map_err(|_| WaveReaderError::ReadError)?;

        let riff_header = Self::read_riff_chunk(&mut wav_file)?;
        wav_file.seek(SeekFrom::Start(12)).map_err(|_| WaveReaderError::ReadError)?;
        let fmt_header = Self::read_fmt_chunk(&mut wav_file)?;
        let mut data_chunks = Vec::new();

        wav_file.seek(SeekFrom::Start(36)).map_err(|_| WaveReaderError::ReadError)?;
        let data_chunk = Self::read_data_chunk(
            wav_file.seek(SeekFrom::Current(0)).map_err(|_| WaveReaderError::ReadError)?, 
            &fmt_header, 
            wav_file
        )?;
        data_chunks.push(data_chunk);

        Ok(PCMWaveInfo {
            riff_header,
            fmt_header,
            data_chunks,
        })
    }

    /// Read the RIFF header from a PCM WAV file
    fn read_riff_chunk(fh: &mut File) -> Result <RiffChunk, WaveReaderError> {
        let wav_file = fh; // as per the specfications; fh is an argument to this function


        let mut riff_chunk = RiffChunk{ // instantiate a RiffChunk object to be returned (when no errors encountered)
            file_size: 0,
            is_big_endian: false,
        };


        // === [START] BUFFER DATA ===
        let mut riff_buff_fx = [0u8; 4];
        let mut riff_buff_file_size = [0u8; 4];
        let mut riff_buff_magic_wave = [0u8; 4];

        wav_file.read_exact(&mut riff_buff_fx); // reads the first four bytes then appends the read data to riff_buff_fx
        wav_file.read_exact(&mut riff_buff_file_size); // while we may have used just the vanilla "read" function previously, in this case we want to read exactly as per the last position of the pointer
        wav_file.read_exact(&mut riff_buff_magic_wave);
        // === [END] OF BUFFER DATA SECTION ===


        // === [START] CONVERT STRINGS TO MAGIC STRINGS FOR COMPARISON ===
        let riff = "RIFF".to_string();
        let rifx = "RIFX".to_string();
        let wave = "WAVE".to_string();
        // === [END] Converting strings to magic strings ===


        // === [START] CHECK IF RIFF OR RIFX AND POPULATE RESPECTIVE ARRAYS ===
        let riff_wav_magic_fx = String::from_utf8_lossy(&riff_buff_fx);
        let mut riff_wav_magic_wave = 0u32;

        if riff_wav_magic_fx == riff{
            riff_chunk.file_size = u32::from_le_bytes(riff_buff_file_size);
            let riff_wav_magic_wave = u32::from_le_bytes(riff_buff_magic_wave);
        }
        else if riff_wav_magic_fx == rifx{
            riff_chunk.file_size = u32::from_be_bytes(riff_buff_file_size);
            riff_chunk.is_big_endian = true;
            let riff_wav_magic_wave = u32::from_be_bytes(riff_buff_magic_wave);
        }
        else{
            return Err(WaveReaderError::NotRiffError)
        }
        // === [END] Checking if RIFF or RIFX ===


        // === [START] CHECK FOR "WAVE" ===
        let wave_bytes: [u8; 4] = [0x57, 0x41, 0x56, 0x45]; // ASCII representation of "WAVE"
        if riff_buff_magic_wave == wave_bytes {
            return Ok(riff_chunk);
        } else {
            return Err(WaveReaderError::NotWaveError);
        }
        // === [END] CHECKING FOR "wAVE" ===

        
        // === [DEBUG ONLY] PRINT EACH SUBCHUNK ===
        println!("[1] RIFF/RIFX: {riff_wav_magic_fx}");
        //println!("[2] File size: {riff_wav_file_size:#10} bytes");
        println!("[3] RIFF file format: {riff_wav_magic_wave} - (magic string is WAVE; should be 0x57415645)");
        // === [END OF DEBUG] ===
    }

    /// Read the format chunk from a PCM WAV file
    fn read_fmt_chunk(fh: &mut File) -> Result<PCMWaveFormatChunk, WaveReaderError> {
        let mut fmt_buff = [0u8; 24];
        fh.read_exact(&mut fmt_buff).map_err(|_| WaveReaderError::ReadError)?;

        if &fmt_buff[0..4] != b"fmt " {
            return Err(WaveReaderError::ChunkTypeError);
        }

        let num_channels = LittleEndian::read_u16(&fmt_buff[10..12]);
        let samp_rate = LittleEndian::read_u32(&fmt_buff[12..16]);
        let bps = LittleEndian::read_u16(&fmt_buff[22..24]);

        Ok(PCMWaveFormatChunk {
            num_channels,
            samp_rate,
            bps,
        })
    }

    /// Read the data chunk from a PCM WAV file
    fn read_data_chunk(start_pos: u64, fmt_info: &PCMWaveFormatChunk, mut fh: File) -> Result<PCMWaveDataChunk, WaveReaderError> {
        let mut data_buff = [0u8; 8];
        fh.seek(SeekFrom::Start(start_pos)).map_err(|_| WaveReaderError::ReadError)?;
        fh.read_exact(&mut data_buff).map_err(|_| WaveReaderError::ReadError)?;

        if &data_buff[0..4] != b"data" {
            return Err(WaveReaderError::ChunkTypeError);
        }

        let audio_size = LittleEndian::read_u32(&data_buff[4..8]);
        if audio_size % fmt_info.block_align() as u32 != 0 {
            return Err(WaveReaderError::DataAlignmentError);
        }

        let audio_stream = BufReader::new(fh);
        Ok(PCMWaveDataChunk {
            size_bytes: audio_size,
            format: *fmt_info,
            data_buf: audio_stream,
        })
    }
}

impl error::Error for WaveReaderError {}

impl fmt::Display for WaveReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaveReaderError::NotRiffError => write!(f, "Not a RIFF File"),
            WaveReaderError::NotWaveError => write!(f, "Not a WAV File"),
            WaveReaderError::NotPCMError => write!(f, "Not a PCM File"),
            WaveReaderError::ChunkTypeError => write!(f, "Unrecognized Chunk Type"),
            WaveReaderError::DataAlignmentError => write!(f, "Data size mismatch"),
            WaveReaderError::ReadError => write!(f, "I/O Error"),
        }
    }
}

impl From<io::Error> for WaveReaderError {
    fn from(_: io::Error) -> Self {
        WaveReaderError::ReadError
    }
}

impl fmt::Display for PCMWaveInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WAVE File {} bytes, {}-bit {} channels, {} Hz, {} data chunks", 
            self.riff_header.file_size,
            self.fmt_header.bps,
            self.fmt_header.num_channels,
            self.fmt_header.samp_rate,
            self.data_chunks.len()
        )
    }
}

impl PCMWaveFormatChunk {
    /// Get or calculate the byte rate of this PCM WAV file
    fn byte_rate(&self) -> u32 {
        (self.samp_rate * self.num_channels as u32 * self.bps as u32) / 8
    }

    /// Get or calculate the block alignment of this PCM WAV file
    fn block_align(&self) -> u16 {
        (self.num_channels * (self.bps / 8)) as u16
    }
}

impl Iterator for PCMWaveDataChunk {
    type Item = Vec<i64>;

    fn next(&mut self) -> Option<Self::Item> {
        let num_channels = self.format.num_channels as usize;
        let mut chunks = Vec::new();

        for _ in 0..num_channels {
            let bps = self.format.bps as usize / 8;
            let mut buffer = vec![0; bps];

            // EOF or Error
            if self.data_buf.read_exact(&mut buffer).is_err() {
                return None;
            }

            chunks.push(match bps {
                2 => i16::from_le_bytes([buffer[0], buffer[1]]) as i64,
                1 => i8::from_le_bytes([buffer[0]]) as i64,
                _ => return None,
            });
        }
        Some(chunks)
    }
}


impl Iterator for PCMWaveDataChunkWindow {
    type Item = Vec <Vec <i64>>;

    fn next(&mut self) -> Option <Self::Item> {
        let mut chunk_window = Vec::new();

        for _ in 0..self.chunk_size {
            if let Some(sample) = self.data_chunk.next() {
                chunk_window.push(sample);
            } else {
                break;
            }
        }

        if chunk_window.is_empty() {
            return None
        }
        Some(chunk_window)
    }
}

impl PCMWaveDataChunk {
    /// Consume a data chunk and get an iterator
    /// 
    /// This method is used to get a *single* inter-channel
    /// sample from a data chunk.
    pub fn chunks_byte_rate(self) -> PCMWaveDataChunkWindow {
        let grab_data_chunk = PCMWaveDataChunkWindow {
            chunk_size: self.format.byte_rate() as usize,
            data_chunk: self
        };
        return grab_data_chunk;
    }

    /// Consume a data chunk and get an iterator
    /// 
    /// This method is used to get a `chunk_size` amount of inter-channel
    /// samples. For example, if there are two channels and the chunk size is
    /// 44100 corresponding to a sample rate of 44100 Hz, then the iterator will
    /// return a `Vec` of size *at most* 44100 with each element as another `Vec`
    /// of size 2.
    pub fn chunks(self, chunk_size: usize) -> PCMWaveDataChunkWindow {
        PCMWaveDataChunkWindow {
            chunk_size,
            data_chunk: self,
        }
    }
}

// TODO: Add more tests here!
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    mod read_riff {
        use super::*;
        use std::io::Write;

        fn create_temp_file(file_name: &str, content: &[u8]) -> Result <(), io::Error> {
            let mut file = File::create(file_name)?;
            file.write_all(content)?;

            Ok(())
        }
        
        macro_rules! internal_tests {
            ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() -> Result <(), WaveReaderError> {
                    let (input, (will_panic, expected)) = $value;

                    let file_name = format!("midp_{}.wav.part", stringify!($name));
                    let result;
                    {
                        create_temp_file(&file_name, input)?;
                        let mut input_fh = File::open(&file_name)?;
                        result = WaveReader::read_riff_chunk(&mut input_fh);
                    }
                    std::fs::remove_file(&file_name)?;

                    if will_panic {
                        assert!(result.is_err());
                    }
                    else if let Ok(safe_result) = result {
                        assert_eq!(expected.file_size, safe_result.file_size);
                        assert_eq!(expected.is_big_endian, safe_result.is_big_endian);
                    }
                    else {
                        result?;
                    }

                    Ok(())
                }
            )*
            }
        }
        
        internal_tests! {
            it_valid_le_00: (
                &[0x52, 0x49, 0x46, 0x46, 0x0, 0x0, 0x0, 0x0, 0x57, 0x41, 0x56, 0x45],
                (
                    false,
                    RiffChunk {
                        file_size: 0,
                        is_big_endian: false,
                    },
                )),
            it_valid_le_01: (
                &[0x52, 0x49, 0x46, 0x46, 0x80, 0x0, 0x0, 0x0, 0x57, 0x41, 0x56, 0x45],
                (
                    false,
                    RiffChunk {
                        file_size: 128,
                        is_big_endian: false,
                    },
                )),
            it_valid_le_02: (
                &[0x52, 0x49, 0x46, 0x46, 0x1C, 0x40, 0x36, 0x0, 0x57, 0x41, 0x56, 0x45],
                (
                    false,
                    RiffChunk {
                        file_size: 3_555_356,
                        is_big_endian: false,
                    },
                )),
            it_valid_be_00: (
                &[0x52, 0x49, 0x46, 0x58, 0x0, 0x0, 0x0, 0x0, 0x57, 0x41, 0x56, 0x45],
                (
                    false,
                    RiffChunk {
                        file_size: 0,
                        is_big_endian: true,
                    },
                )),
            it_valid_be_01: (
                &[0x52, 0x49, 0x46, 0x58, 0x00, 0x0, 0x0, 0x80, 0x57, 0x41, 0x56, 0x45],
                (
                    false,
                    RiffChunk {
                        file_size: 128,
                        is_big_endian: true,
                    },
                )),
            it_valid_be_02: (
                &[0x52, 0x49, 0x46, 0x58, 0x00, 0x36, 0x40, 0x1C, 0x57, 0x41, 0x56, 0x45],
                (
                    false,
                    RiffChunk {
                        file_size: 3_555_356,
                        is_big_endian: true,
                    },
                )),
            it_bad_riff: (
                &[0x00, 0x49, 0x46, 0x46, 0x00, 0x36, 0x40, 0x1C, 0x57, 0x41, 0x56, 0x45],
                (
                    true,
                    RiffChunk {
                        file_size: 0,
                        is_big_endian: false,
                    },
                )),
            it_bad_wave: (
                &[0x52, 0x49, 0x46, 0x46, 0x00, 0x36, 0x40, 0x1C, 0x57, 0x41, 0x56, 0x00],
                (
                    true,
                    RiffChunk {
                        file_size: 0,
                        is_big_endian: false,
                    },
                )),
        }
    }

    #[cfg(test)]
    mod read_wav_fmt {
        use super::*;
        use std::io::Write;

        fn create_temp_file(file_name: &str, content: &[u8]) -> Result <(), io::Error> {
            let mut file = File::create(file_name)?;
            file.write_all(content)?;

            Ok(())
        }
        
        macro_rules! internal_tests {
            ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() -> Result <(), WaveReaderError> {
                    let (input, (will_panic, expected)) = $value;

                    let file_name = format!("midp_{}.wav.part", stringify!($name));
                    let result;
                    {
                        create_temp_file(&file_name, input)?;
                        let mut input_fh = File::open(&file_name)?;
                        result = WaveReader::read_fmt_chunk(&mut input_fh);
                    }
                    std::fs::remove_file(&file_name)?;

                    if will_panic {
                        assert!(result.is_err());
                    }
                    else if let Ok(safe_result) = result {
                        assert_eq!(expected.num_channels, safe_result.num_channels);
                        assert_eq!(expected.samp_rate, safe_result.samp_rate);
                        assert_eq!(expected.bps, safe_result.bps);
                    }
                    else {
                        result?;
                    }

                    Ok(())
                }
            )*
            }
        }
        
        internal_tests! {
            it_valid_00: (
                &[
                    0x66, 0x6d, 0x74, 0x20,
                    0x10, 0x0, 0x0, 0x0,
                    0x01, 0x0,
                    0x01, 0x0,
                    0x44, 0xac, 0x0, 0x0,
                    0x44, 0xac, 0x0, 0x0,
                    0x01, 0x00, 0x08, 0x0,
                ],
                (
                    false,
                    PCMWaveFormatChunk {
                        num_channels: 1,
                        samp_rate: 44100,
                        bps: 8,
                    },
                )),
            it_valid_01: (
                &[
                    0x66, 0x6d, 0x74, 0x20,
                    0x10, 0x0, 0x0, 0x0,
                    0x01, 0x0,
                    0x02, 0x0,
                    0x44, 0xac, 0x0, 0x0,
                    0x88, 0x58, 0x01, 0x0,
                    0x02, 0x00, 0x08, 0x0,
                ],
                (
                    false,
                    PCMWaveFormatChunk {
                        num_channels: 2,
                        samp_rate: 44100,
                        bps: 8,
                    },
                )),
            it_valid_02: (
                &[
                    0x66, 0x6d, 0x74, 0x20,
                    0x10, 0x0, 0x0, 0x0,
                    0x01, 0x0,
                    0x02, 0x0,
                    0x44, 0xac, 0x0, 0x0,
                    0x10, 0xb1, 0x02, 0x0,
                    0x04, 0x00, 0x10, 0x0,
                ],
                (
                    false,
                    PCMWaveFormatChunk {
                        num_channels: 2,
                        samp_rate: 44100,
                        bps: 16,
                    },
                )),
        }
    }

    mod read_data_fmt {
        // TODO
    }
}
