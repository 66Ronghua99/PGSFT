# Third-party software and data

| Component | Pinned identity | License / use |
|---|---|---|
| Qwen3-4B-Thinking-2507 | `768f209d9ea81521153ed38c47d515654e938aea` | Apache-2.0; model downloaded separately |
| Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0; model downloaded separately |
| Nemotron-SFT-SWE-v3 | `3f73de64c1fe928a8f538fe45ccc10c228cc4c6a` | source dataset terms; downloaded separately |
| PSFT reference code | `zwhong714/PSFT@930e23980a723ecef5af138e6e32aa3798b1fd64` | Apache-2.0; objective behavior adapted, no source vendored |
| BFCL | `bfcl-eval==2026.3.23` | Apache-2.0; installed separately |
| LiveCodeBench | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` | MIT; installed separately |
| LiveCodeBench dataset | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` | dataset downloaded separately |
| SWE-bench | `swebench==4.1.0` | MIT; installed separately |
| SWE-bench Lite dataset | `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2` | dataset downloaded separately |

The checked-in Qwen Jinja files preserve the tokenizer formatting required by
the pinned model releases. The SWE editor/protocol were adapted from this
project's evaluation runner and use only Python standard-library code.
