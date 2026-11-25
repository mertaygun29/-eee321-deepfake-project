# Test fixtures

Put a real photo with one clearly visible face at `sample_face.jpg` in this
folder to run the tests that need a real face. This file is intentionally
git-ignored (root `.gitignore` blocks `*.jpg`) — never commit real photos.
Tests that need it will skip with a clear message if it's missing.

For lip-sync tests: a few-seconds video with a visible talking face at
`sample_talking.mp4`, and a short speech/audio clip at
`sample_speech.wav`. Also gitignored — add them locally to run the
fixture-gated lip-sync tests. A quick way to make placeholder versions
(not realistic lip movement, but enough to exercise the pipeline) from
`sample_face.jpg` and a test tone:

```powershell
ffmpeg -loop 1 -i sample_face.jpg -c:v libx264 -t 3 -pix_fmt yuv420p -vf "fps=25,scale=320:-2" sample_talking.mp4
ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 sample_speech.wav
```
