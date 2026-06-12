# AudioDNA

A game sound-effect classifier and similarity search engine.

> **Status: work in progress** — Phase 1 (scaffolding) complete.
> The full README (architecture, results, screenshots) lands in Phase 8.

AudioDNA collects game sound effects from [Freesound.org](https://freesound.org)
across six categories (`impact`, `footsteps`, `ambience`, `ui`, `explosion`,
`weapon`), extracts audio features with librosa, trains a category classifier,
and serves similarity search ("find 5 sounds like this one") in a Streamlit app.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then paste your Freesound API key into .env
python check_setup.py         # verify everything works
```

Get a free Freesound API key at <https://freesound.org/apiv2/apply/>.
