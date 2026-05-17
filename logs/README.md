# `logs/`

Directory for all the system logs.

**Note**: This project uses DVC to manage log files. Make sure `dvc` is installed, then pull all logging artifacts from the remote bucket:
```bash
dvc pull
```

```text
.
├── pipeline                                        <- All pipeline execution logs
│   └── electronics
├── pipeline.dvc                                    <- Tracks pipeline execution logs with DVC
├── README.md                                       <- Structure overview (this file)
├── reco2gnizer                                     <- All agent execution logs
│   ├── amazon_reviews_23
│   │   └── item_metadata
│   │       ├── google_genai:gemini-2.5-flash
│   │       │   └── electronics
│   │       └── openai:o3-mini
│   │           └── electronics
│   └── ground_truths
│       ├── google_genai:gemini-2.5-flash
│       │   └── electronics
│       └── openai:o3-mini
│           └── electronics
└── reco2gnizer.dvc                                 <- Tracks agent execution logs with DVC
```
