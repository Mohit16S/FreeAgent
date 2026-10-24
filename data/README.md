# Local data

Run `python -m freeagent.cli ingest --competition 2 --season 27` to retrieve StatsBomb research files.
`raw/` preserves downloaded source bytes and manifest metadata. `processed/` contains
validated matches, appearance approximations, provider-ID maps, quality reports and
quarantined appearances. Both are excluded from source control under source-use limits.
Do not publish or redistribute these datasets; review `raw/LICENSE.pdf`.

For your own permitted CSV data, follow `docs/DATA_DICTIONARY.md`, place matches.csv
and appearances.csv in processed/, include provenance, and run `python -m freeagent.cli validate`.
Missing data produces clear errors; synthetic fixtures exist only in tests/.

On a fresh Streamlit deployment, use **Data & models → Download real data and train models**, or run `python -m freeagent.cli setup --evaluate`. Ignored local datasets are deliberately not required in Git.
