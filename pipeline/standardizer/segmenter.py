"""
Audio Segmenter Module

Chunks audio into fixed-duration segments for consistent model input.
Tracks source_id across all chunks from same original audio.
"""

import numpy as np
from typing import List, Dict, Any
from pipeline.config import Config


def segment_audio(
    audio: np.ndarray,
    source_id: str,
    config: Config,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Split audio into fixed-duration chunks.
    
    Args:
        audio: Preprocessed audio array
        source_id: Identifier linking chunks to source (for speaker-disjoint splits)
        config: Pipeline configuration
        metadata: Additional metadata to copy to each chunk
        
    Returns:
        List of chunk dictionaries with audio and tracking info
    """
    sr = config.audio.target_sr
    chunk_samples = int(config.segmentation.chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - config.segmentation.overlap))
    
    chunks = []
    chunk_idx = 0
    
    # Slide window through audio
    start = 0
    while start < len(audio):
        end = start + chunk_samples
        
        # Extract chunk
        if end <= len(audio):
            chunk_audio = audio[start:end]
        else:
            # Last chunk - handle padding or skip
            remaining = audio[start:]
            
            if config.segmentation.pad_last_chunk:
                # Pad with zeros to reach target duration
                padding = np.zeros(chunk_samples - len(remaining))
                chunk_audio = np.concatenate([remaining, padding])
            else:
                # Skip short last chunk
                break
        
        # Build chunk record
        chunk = {
            "audio": chunk_audio,
            "source_id": source_id,
            "chunk_idx": chunk_idx,
            "start_sample": start,
            "end_sample": min(end, len(audio)),
            "duration": len(chunk_audio) / sr,
            "is_padded": end > len(audio),
        }
        
        # Copy over metadata from source
        if metadata:
            for key in ["speaker_id", "transcript", "gender", "accent", "locale"]:
                if key in metadata:
                    chunk[key] = metadata[key]
        
        chunks.append(chunk)
        chunk_idx += 1
        start += hop_samples
    
    return chunks


def estimate_num_chunks(audio_duration: float, config: Config) -> int:
    """
    Estimate number of chunks that will be produced.
    
    Useful for progress estimation and memory planning.
    """
    chunk_duration = config.segmentation.chunk_duration
    overlap = config.segmentation.overlap
    hop_duration = chunk_duration * (1 - overlap)
    
    if audio_duration <= chunk_duration:
        return 1
    
    return int(np.ceil((audio_duration - chunk_duration) / hop_duration)) + 1


def get_chunk_transcript(
    full_transcript: str,
    chunk_idx: int,
    total_chunks: int
) -> str:
    """
    Approximate transcript portion for a chunk.
    
    Note: This is a rough approximation. For accurate alignment,
    use forced alignment tools like Montreal Forced Aligner.
    """
    if not full_transcript or total_chunks <= 1:
        return full_transcript
    
    words = full_transcript.split()
    words_per_chunk = max(1, len(words) // total_chunks)
    
    start_word = chunk_idx * words_per_chunk
    end_word = min((chunk_idx + 1) * words_per_chunk, len(words))
    
    return " ".join(words[start_word:end_word])


if __name__ == "__main__":
    from pipeline.config import load_config
    
    cfg = load_config("../config.yaml")
    
    # Test with 10-second audio
    sr = cfg.audio.target_sr
    test_audio = np.random.randn(10 * sr) * 0.1  # 10 seconds of noise
    
    chunks = segment_audio(
        test_audio,
        source_id="test_001",
        config=cfg,
        metadata={"speaker_id": "spk_test", "transcript": "hello world"}
    )
    
    print(f"Source duration: {len(test_audio)/sr:.2f}s")
    print(f"Chunk duration: {cfg.segmentation.chunk_duration}s")
    print(f"Number of chunks: {len(chunks)}")
    
    for chunk in chunks:
        print(f"  Chunk {chunk['chunk_idx']}: {chunk['duration']:.2f}s, "
              f"padded={chunk['is_padded']}")
