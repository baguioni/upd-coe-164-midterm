use core::fmt;
use std::fs::File;
use std::path::Path;
use std::error;
use std::io::{self, Read, Seek, SeekFrom, BufReader};
use byteorder::{ByteOrder, LittleEndian, LE};

// ===== [IMPORTANT] WAV COMPLIANT VERSION NOTES =====
// THIS VERSION OF WAV ONLY SUPPORTS RIFF, RIFX SUPPORT TO BE ADDED LATER
// ===== !!!!!!!!! =====

// CHARLIE VERSION, ALMOST READY FOR PRODUCTION

// ===== OUTSIDE OF TEMPLATE ADDITIONS ===
//#[derive()]
//enum Result {
//}
// ===== END OF ADDITIONS =====



/// Represents a PCM WAV file
pub struct PCMWaveInfo {
    pub riff_header: RiffChunk,
    pub fmt_header: PCMWaveFormatChunk,
    pub data_chunks: Vec <PCMWaveDataChunk>,
}



// === ADDED AS PER SPECIFICIATIONS ===
// impl PCMWaveInfo{
//     fn byte_rate(&self, byterate: u32){
//         println!("{}",self.fmt_header::fmthead_byterate);
//     }

//     fn block_align(&self){

//     }
// }


/// Represents a RIFF chnk from a WAV file
/// 
/// The RIFF chunk is the first 12 bytes of a WAV file.
pub struct RiffChunk {
    pub file_size: u32,
    pub is_big_endian: bool,
}

/// Represents a format chunk from a WAV file
/// 
/// A format chunk in a WAV file starts with a magic string
/// `fmt_` where `_` is a space (0x20 in hex) and then followed by
/// 20 bytes of metadata denoting information about the audio file
/// itself such as the sample and bit rates.
#[derive(Clone, Copy)]
pub struct PCMWaveFormatChunk {
    pub num_channels: u16,
    pub samp_rate: u32,
    pub bps: u16,
}

/// Represents a data chunk from a WAV file
/// 
/// A data chunk in a WAV file starts with a magic string `data` and then
/// followed by the number of samples that follow and then finally the
/// audio data samples themselves.
pub struct PCMWaveDataChunk {
    pub size_bytes: u32,
    pub format: PCMWaveFormatChunk,
    pub data_buf: io::BufReader<File>,
}

/// Represents an iterator to a data chunk from a WAV file
/// 
/// This struct is not instantiated by itself and is generated
/// by calling the methods `PCMWaveDataChunk::chunks_byte_rate()`
/// and `PCMWaveDataChunk::chunks()`.
pub struct PCMWaveDataChunkWindow {
    chunk_size: usize,
    data_chunk: PCMWaveDataChunk
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
    /// 
    /// The WAV file located at `file_path` will be represented as a `PCMWaveInfo`
    /// struct for further processing.
    /// 
    /// # Errors
    /// Returns a `WaveReaderError` with the appropriate error if something
    /// happens.
    pub fn open_pcm(file_path: &str) -> Result <PCMWaveInfo, WaveReaderError> {
        let mut wav_file = File::open(file_path).map_err(|_| WaveReaderError::ReadError)?;

        let riff_header = Self::read_riff_chunk(&mut wav_file)?;
        wav_file.seek(SeekFrom::Start(12)).map_err(|_| WaveReaderError::ReadError)?;
        let fmt_header = Self::read_fmt_chunk(&mut wav_file)?;
        let mut data_chunks = Vec::new();

        wav_file.seek(SeekFrom::Start(36)).map_err(|_| WaveReaderError::ReadError)?;
        let data_chunk = Self::read_data_chunk(
            wav_file
            .seek(SeekFrom::Current(0))
            .map_err(|_| WaveReaderError::ReadError)?, &fmt_header, wav_file)?;
        data_chunks.push(data_chunk);

        Ok(PCMWaveInfo{
                riff_header: riff_header,
                fmt_header: fmt_header,
                data_chunks: data_chunks
            }
        )
    }

    /// Read the RIFF header from a PCM WAV file
    /// 
    /// The RIFF header is the first twelve bytes of a PCM WAV
    /// file of the format `<RIFF_magic_str:4B><file_size:4B><RIFF_type_magic_str:4B>`.
    /// Note that the file handle `fh` should point to the very start of the file.
    /// 
    /// # Errors
    /// Returns a `WaveReaderError` with the appropriate error if something
    /// happens. This includes file read errors and format errors.
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
    /// 
    /// The format chunk usually appears immediately after the RIFF header and consists of 24 bytes of metadata.
    /// Note that the file handle `fh` should point to the start of a format chunk.
    /// 
    /// # Errors
    /// Returns a `WaveReaderError` with the appropriate error if something
    /// happens. This includes file read errors and format errors.
    fn read_fmt_chunk(fh: &mut File) -> Result <PCMWaveFormatChunk, WaveReaderError> {
        let wav_file = fh; // same as from read_riff_chunk

        // === [START] BUFFER DATA ===
        let mut fmthead_buff_fmt_magic = [0u8; 4];
        let mut fmthead_buff_fsubchunksize = [0u8; 4];
        let mut fmthead_buff_audiofmt = [0u8; 2];
        let mut fmthead_buff_numchs = [0u8; 2];
        let mut fmthead_buff_samplerate = [0u8; 4];
        let mut fmthead_buff_byterate = [0u8; 4];
        let mut fmthead_buff_blockalign = [0u8; 2];
        let mut fmthead_buff_bitdepth = [0u8; 2];
        
        wav_file.read_exact(&mut fmthead_buff_fmt_magic);
        wav_file.read_exact(&mut fmthead_buff_fsubchunksize);
        wav_file.read_exact(&mut fmthead_buff_audiofmt);
        wav_file.read_exact(&mut fmthead_buff_numchs);
        wav_file.read_exact(&mut fmthead_buff_samplerate);
        wav_file.read_exact(&mut fmthead_buff_byterate);
        wav_file.read_exact(&mut fmthead_buff_blockalign);
        wav_file.read_exact(&mut fmthead_buff_bitdepth);
        // === [END] OF BUFFER DATA SECTION ===


        // === [START] FMT Magic String ===
        let mut fmthead_fmt_magic = 0u32;
        for j in 0..4{
            fmthead_fmt_magic += fmthead_buff_fmt_magic[j] as u32;
            if j == 3{
                break
            }
            fmthead_fmt_magic <<= 8;
        }
        // === [END] of FMT Magic String ===


        // === [START] FSubchunkSize READER ===
        let fmthead_fsubchunksize = u32::from_le_bytes(fmthead_buff_fsubchunksize); // always little endian (?)
        // === [END] OF FSUBCHUNKSIZE READER ===


        // === [START] AudioFmt READER ===
        let mut fmthead_audiofmt = 0u32;
        for k in 0..2{
            fmthead_audiofmt += fmthead_buff_audiofmt[k] as u32;
            if k == 3{
                break
            }
            // Since little-endian always(?), no need for bitshift: fmthead_audiofmt <<= 8;
        }
        // === [END] OF AudioFmt READER ===


        // === [START] Byte Rate READER ===
        let fmthead_byterate = u32::from_le_bytes(fmthead_buff_byterate);
        // === [END] OF Byte Rate READER ===


        // === [START] Block Align READER ===
        let mut fmthead_blockalign = 0u32;
        for m in 0..2{
            fmthead_blockalign += fmthead_buff_blockalign[m] as u32;
        }
        // === [END] OF Block Align Reader ===


        // === [START] Numchs READER ===
        let mut fmthead_numchs = 0u16;
        for l in 0..2{
            fmthead_numchs += fmthead_buff_numchs[l] as u16;
            if l == 3{
                break
            }
            // Since little-endian always(?), no need for bitshift: fmthead_numchs <<= 8;
        }
        // === [END] OF NUMCHS READER ===

        // === [START] Sample Rate READER ===
        let fmthead_samplerate = u32::from_le_bytes(fmthead_buff_samplerate);
        // === [END] OF Sample Rate READER ===
        
        
        // === [START] BIT DEPTH READER ===
        let mut fmthead_bitdepth = 0u16;
        for n in 0..2{
            fmthead_bitdepth += fmthead_buff_bitdepth[n] as u16;
        }
        // === [END] OF BIT DEPTH READER ===


        let fmt_magic = "fmt ".to_string();
        let fmt_to_check = String::from_utf8_lossy(&fmthead_buff_fmt_magic);
        let mut fmt_chunk = PCMWaveFormatChunk{
            num_channels: fmthead_numchs,
            samp_rate: fmthead_samplerate,
            bps: fmthead_bitdepth,
        };

        if fmt_to_check == fmt_magic{
            return Ok(fmt_chunk);
        }
        else{
            return Err(WaveReaderError::ChunkTypeError);
        }

        
        // === [DEBUG ONLY] PRINTING THE CHUNKS ===
        println!("[4] fmt_ magic string: {fmthead_fmt_magic:#10x} - (magic string should be 0x666d7420)");
        println!("[5] FSubchunkSize: {fmthead_fsubchunksize:#10x} - (fixed value is 16 or 0x00000010)");
        println!("[6] AudioFmt: {fmthead_audiofmt:#06x} - (fixed value is 1 or 0x0001)");
        // println!("[7] Number of Channels: {fmthead_numchs:#1} - (1 for mono, 2 for stereo)");
        // println!("[8] Sample Rate: {fmthead_samplerate:#10} Hz");
        println!("[9] Byte Rate: {fmthead_byterate:#10}");
        println!("[10] Block Alignment: {fmthead_blockalign:#10} bytes");
        // println!("[11] Bit Depth: {fmthead_bitdepth:#10} bits-per-sample (size of the sample)");
        // === [END OF DEBUG] ===        
    }

    /// Read the data chunk from a PCM WAV file
    /// 
    /// The data chunk usually appears immediately after the format
    /// chunk and contains the samples of the audio itself. Note that
    /// a file can contain multiple data chunks, and it is possible that this
    /// method should be called more than once to completely read the file.
    /// Note that the file handle `fh` should point to the start of a data chunk.
    /// 
    /// # Errors
    /// Returns a `WaveReaderError` with the appropriate error if something
    /// happens. This includes file read errors and format errors.
    fn read_data_chunk(start_pos: u64, fmt_info: &PCMWaveFormatChunk, mut fh: File) -> Result <PCMWaveDataChunk, WaveReaderError> {
        //todo!();
        // What's left to do?

        
        // === [START] PARSE DETAILS FROM OTHER OBJECTS ===
        let bitdepth = fmt_info.bps;
        let numchs = fmt_info.samp_rate;
        // === [END] OF PARSING DETAILS FROM OTHER OBJECTS ===


        // === [START] BUFFER DATA ===
        let mut data_buff_magic = [0u8; 4];
        let mut data_buff_dsubchunksize = [0u8; 4];

        fh.read_exact(&mut data_buff_magic);
        fh.read_exact(&mut data_buff_dsubchunksize);
        // === [END] OF BUFFER DATA SECTION ===


        // === [START] DATA HEADER MAGIC STRING READRER ===
        let data_magic = u32::from_be_bytes(data_buff_magic);
        // === [END] OF DATA HEADER MAGIC STRING READER ===


        // === [START] DSubchunk Size Reader ===
        let data_dsubchunksize = u32::from_le_bytes(data_buff_dsubchunksize);
        // === [END] OF Dsubchunk Size Reader ===


        // === [START] Number of Samples (computed) ===
        // let number_of_samples = (data_dsubchunksize << 8)/(numchs * bitdepth);
        // === [END] of Number of Samples (computed) ===


        // === [START] (Experimental) Audio Data ===
        let mut data_buff_audio = [[0u8; 2]; 38956]; // fixed amount for now, but it should be a variable
        // === [END] Audio Data ===        

    
        // === [DEBUG ONLY] PRINT EACH SUBCHUNK ===
        println!("[12] data magic string: {data_magic:#10x} - (0x64617461)");
        println!("[13] DSubchunk Size: {data_dsubchunksize:#10}");
        // println!("[14] Number of Samples: {number_of_samples:#10}");
        println!("[15] No audio data representation yet");
        // === [END OF DEBUG] ===

        // == Implementation ==
        let mut data_header = [0u8; 8];
        fh.read_exact(&mut data_header).map_err(|_| WaveReaderError::ReadError)?;
        let data_tag = String::from_utf8_lossy(&data_header[0..4]).to_string();

        if data_tag.as_str() != "data" {
            return Err(WaveReaderError::ChunkTypeError);
        }

        let data_size = LittleEndian::read_u32(&data_header[4..8]);


        if data_size % fmt_info.block_align() as u32 != 0 {
            return Err(WaveReaderError::DataAlignmentError);
        }

        let data_buf = BufReader::new({
            fh.seek(SeekFrom::Start(start_pos + 8))?;
            fh
        });

        return Ok(PCMWaveDataChunk{
            size_bytes: data_size,
            format: *fmt_info,
            data_buf: data_buf,
        })
    }
}

impl error::Error for WaveReaderError {}

impl fmt::Display for WaveReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaveReaderError::NotRiffError => write!(f, "Not a RIFF File"),
            WaveReaderError::NotWaveError => write!(f,"Not a WAV File"),
            WaveReaderError::NotPCMError => write!(f, "Not a PCM File"),
            WaveReaderError::ChunkTypeError => write!(f, "Unrecognized Chunk Type"),
            WaveReaderError::DataAlignmentError => write!(f, "Data size mismatch"),
            WaveReaderError::ReadError => write!(f, "I/O Error"),
        }
    }
}

impl From <io::Error> for WaveReaderError {
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
        let samp_rate = self.samp_rate;
        let num_channels = self.num_channels as u32;
        let bps = self.bps as u32;

        (samp_rate * num_channels * bps) / 8
    }

    /// Get or calculate the block alignment of this PCM WAV file
    /// 
    /// The *block alignment* is the size of one *inter-channel* sample
    /// in bytes. An *inter-channel sample* is a sample with all of its
    /// channels collated together.
    fn block_align(&self) -> u16 {
        let num_channels = self.num_channels;
        let bytes_per_sample = self.bps / 8;

        (num_channels * bytes_per_sample) as u16
    }
}

impl Iterator for PCMWaveDataChunk {
    type Item = Vec <i64>;

    fn next(&mut self) -> Option <Self::Item> {
        let num_channels = self.format.num_channels as usize;
        let mut  chunks = Vec::new();

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
        PCMWaveDataChunkWindow {
            chunk_size: self.format.byte_rate() as usize,
            data_chunk: self
        }
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