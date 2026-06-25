# Preset voices

Drop reference voice clips here (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`,
`.opus`) and they appear in the browser's **Preset voice…** dropdown.

Each clip is decoded in the browser, resampled to 24 kHz mono, capped to ~10 s,
and Mimi-encoded on the server to prime the assistant's voice. Use a few
seconds of clean single-speaker speech for best cloning.

No clips are bundled by default (the PersonaPlex `voices.tgz` files are
precomputed `.pt` embeddings, which are incompatible with this token-forcing
pipeline). The **Record** and **Upload audio** options work without presets.
