# `models/`

```text
.
├── BPR
│   ├── BPR.pth  <- BPR model trained
│   └── predictions
│       ├── gemini_2_5_flash
│       │   ├── alpha_0_25.pth  <- BPR reranked predictions on Gemini 2.5 Flash PCF values (⍺=0.25)
│       │   ├── alpha_0_5.pth  <- BPR reranked predictions on Gemini 2.5 Flash PCF values (⍺=0.5)
│       │   ├── alpha_0_75.pth  <- BPR reranked predictions on Gemini 2.5 Flash PCF values (⍺=0.75)
│       │   └── alpha_1_0.pth  <- BPR reranked predictions on Gemini 2.5 Flash PCF values (⍺=1.0)
│       └── openai_o3_mini
│           ├── alpha_0_25.pth  <- BPR reranked predictions on OpenAI o3-mini PCF values (⍺=0.25)
│           ├── alpha_0_5.pth  <- BPR reranked predictions on OpenAI o3-mini PCF values (⍺=0.5)
│           ├── alpha_0_75.pth  <- BPR reranked predictions on OpenAI o3-mini PCF values (⍺=0.75)
│           └── alpha_1_0.pth  <- BPR reranked predictions on OpenAI o3-mini PCF values (⍺=1.0)
├── LightGCN
│   ├── LightGCN.pth  <- LightGCN model trained
│   └── predictions
│       ├── gemini_2_5_flash
│       │   ├── alpha_0_25.pth  <- LGCN reranked predictions on Gemini 2.5 Flash PCF values (⍺=0.25)
│       │   ├── alpha_0_5.pth  <- LGCN reranked predictions on Gemini 2.5 Flash PCF values (⍺=0.5)
│       │   ├── alpha_0_75.pth  <- LGCN reranked predictions on Gemini 2.5 Flash PCF values (⍺=0.75)
│       │   └── alpha_1_0.pth  <- LGCN reranked predictions on Gemini 2.5 Flash PCF values (⍺=1.0)
│       └── openai_o3_mini
│           ├── alpha_0_25.pth  <- LGCN reranked predictions on OpenAI o3-mini PCF values (⍺=0.25)
│           ├── alpha_0_5.pth  <- LGCN reranked predictions on OpenAI o3-mini PCF values (⍺=0.5)
│           ├── alpha_0_75.pth  <- LGCN reranked predictions on OpenAI o3-mini PCF values (⍺=0.75)
│           └── alpha_1_0.pth  <- LGCN reranked predictions on OpenAI o3-mini PCF values (⍺=1.0)
└── README.md  <- This file 
```

**Note**: The repository doesn't include model files due to their size.