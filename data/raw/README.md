# Raw Catalog — Provenance

- **Source:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
  (public research dataset), category metadata under `raw/meta_categories/`.
- **Fetcher:** `pipelines/crawlers/fetch_amazon_subset.py` (streams a prefix of each
  metadata file, early-stop; keeps rows with positive price; de-duplicates by title).
- **Fetched:** 2026-11-17. **Size:** 500 products.
- **License/attribution:** dataset publicly released by the McAuley Lab for research;
  this small subset is redistributed with attribution to the source above.

## Category mapping (Amazon → project)
| Project | Amazon source | Note |
| :--- | :--- | :--- |
| book | meta_Books | |
| tech | meta_Electronics | |
| cosmetics | meta_Beauty_and_Personal_Care | |
| health | meta_Health_and_Household | |
| perfume | meta_All_Beauty | title keywords: perfume/fragrance/eau de/cologne/parfum |
| fashion | meta_Amazon_Fashion | |

## Honesty note
This catalog is **English** public data used for the live demo pipeline. The contextual
bridging contribution remains *Directional* (small N); the *Confirmatory* claims rely on
the separate public benchmark track (see `docs/benchmark_protocol.md`). Do not describe
this subset as Vietnamese market data.