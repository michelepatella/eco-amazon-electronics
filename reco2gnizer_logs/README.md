# `reco2gnizer_logs/`

Directory for all the pipeline execution logs involving the emission data enrichment process.

**Note**: This project uses DVC to manage the emission data enrichment process' log files. Make sure `dvc` is installed, then pull all logging artifacts from the remote bucket:
```bash
dvc pull
```

```text
.
├── amazon_reviews_23                       <- Logs for Amazon Reviews'23 products enrichment
│   └── item_metadata
│       ├── google_genai:gemini-2.5-flash
│       │   └── electronics
│       └── openai:o3-mini
│           └── electronics
├── amazon_reviews_23.dvc                   <- DVC tracker for Amazon Reviews'23 logs
├── ground_truths                           <- Logs for ground truth products enrichment
│   ├── google_genai:gemini-2.5-flash
│   │   └── electronics
│   └── openai:o3-mini
│       └── electronics
├── ground_truths.dvc                       <- DVC tracker for ground truth logs
└── README.md                               <- Structure overview (this file)
```
