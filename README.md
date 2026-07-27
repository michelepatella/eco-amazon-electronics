<a id="readme-top"></a>

<br/>
<div align="center">
  <h3 align="center">🌱</h3>
  <p align="center">
    Multi-objective recommendation pipeline balancing<br>item relevance with environmental impact for<br>sustainable recommendations.
  </p>
  <p align="center">
  </p>

  <a href="https://github.com/michelepatella/eco-amazon-electronics">GitHub</a> •
  <a href="https://dagshub.com/michelepatella/eco-amazon-electronics">DagsHub</a>
</div>

<br>

<div align="center">
  
[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![RecBole](https://img.shields.io/badge/RecBole-E35A3C?style=for-the-badge&logo=https://recbole.io/docs/_images/logo.png)](https://recbole.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ray](https://img.shields.io/badge/Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white)](https://www.ray.io/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%238CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)

</div>

---



## Pipeline
```text
                  ┌─────────────────────────────────────────────────────────┐
                  │                    Amazon Reviews'23                    │
                  │                                                         │
                  │  ┌---------------------┐       ┌---------------------┐  │
                  │  │    User Reviews     │       │    Item Metadata    │  │
                  │  └----------┬----------┘       └----------┬----------┘  │
                  └─────────────┼─────────────────────────────┼─────────────┘
                                │                             │
                     ╔══════════▼══════════╗       ╔══════════▼══════════╗
                     ║  Data Preprocessing ║       ║ PCF Data Enrichment ║
                     ╚══════════┬══════════╝       ╚══════════┬══════════╝
                                │                             │
                     ╔══════════▼══════════╗       ┌----------▼----------┐
                     ║   Model Training &  ║       │       Enriched      │
                     ║      Inference      ║       |    Item Metadata    |
                     ╚══════════┬══════════╝       └----------┬----------┘
                                │                             │
                     ┌----------▼----------┐                  │
                     │   Recommendations   │                  │
                     └----------┬----------┘                  │
                                │                             │
                                └──────────────┬──────────────┘
                                               │
                                ╔══════════════▼══════════════╗
                                ║     Sustainability-aware    ║
                                ║  Recommendation Re-ranking  ║
                                ╚══════════════┬══════════════╝
                                               │
                                ┌--------------▼--------------┐
                                │  Re-ranked Recommendations  │
                                └--------------┬--------------┘
                                               │
                                ╔══════════════▼══════════════╗
                                ║       Model Evaluation      ║
                                ╚═════════════════════════════╝
```
