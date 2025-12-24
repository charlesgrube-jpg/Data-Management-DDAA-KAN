---
description: Test individual pipeline components
---

# Test Pipeline Component

This workflow helps you test individual components of the pipeline in isolation.

## Components

### Test Preprocessor
```bash
python -c "from pipeline.standardizer.preprocessor import preprocess_audio; import numpy as np; audio = np.random.randn(16000); from pipeline.config import load_config; cfg = load_config('config.yaml'); result, meta = preprocess_audio(audio, 16000, cfg); print('Success:', result is not None, meta)"
```

### Test Segmenter
```bash
python -c "from pipeline.standardizer.segmenter import segment_audio; import numpy as np; from pipeline.config import load_config; cfg = load_config('config.yaml'); audio = np.random.randn(48000); chunks = segment_audio(audio, 'test_001', cfg, {}); print('Chunks created:', len(chunks))"
```

### Test TTS Synthesizer
```bash
python -c "from pipeline.synthesizer.gtts_synthesizer import synthesize_tts; audio = synthesize_tts('Hello world, this is a test', 16000); print('TTS success:', audio is not None, 'Shape:', audio.shape if audio is not None else None)"
```

### Test Effects
```bash
python -c "from pipeline.effects.effects import apply_effects, select_quality_tier; import numpy as np; from pipeline.config import load_config; cfg = load_config('config.yaml'); audio = np.random.randn(16000); tier = select_quality_tier(cfg); result = apply_effects(audio, 16000, tier, cfg); print('Effects applied:', tier, result.shape)"
```

## Interactive Testing

For more detailed testing, use Python interactive mode:
```bash
python -i
```

Then import and test components:
```python
from pipeline.config import load_config
config = load_config("config.yaml")
# Test your component here
```
