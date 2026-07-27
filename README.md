<div align="center">

  [![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
  [![RecBole](https://img.shields.io/badge/RecBole-E35A3C?style=for-the-badge&logo=https://recbole.io/docs/_images/logo.png)](https://recbole.io/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Ray](https://img.shields.io/badge/Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white)](https://www.ray.io/)
  [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![SciPy](https://img.shields.io/badge/SciPy-%238CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)  

  <table>
    <tr>
      <td align="center">
        🌱 <b>Multi-objective recommendation pipeline</b><br>
        balancing item relevance with environmental impact<br>
        for sustainable recommendations.
      </td>
    </tr>
  </table>

  <a href="https://github.com/michelepatella/eco-amazon-electronics">**GitHub**</a> •
  <a href="https://dagshub.com/michelepatella/eco-amazon-electronics">**DagsHub**</a>

</div>

---

A **multi-objective recommendation pipeline** for **collaborative filtering** on **implicit feedback**, applied to the **Amazon Reviews '23 Electronics** dataset to demonstrate how enriching e-commerce catalogs with **carbon footprint information** can be used to balance product relevance with environmental impact to recommend **greener products**.

> [!TIP]
> **How are products enriched with Product Carbon Footprint (PCF)?**  
> The pipeline leverages an AI agent integrating LLM reasoning, function calling, and web retrieval for Product Carbon Footprint estimation.  
> [» Discover more](https://github.com/michelepatella/reco2gnizer)

---

### <code>Pipeline</code>

                  ┌─────────────────────────────────────────────────────────┐
                  │                    Amazon Reviews'23                    │                                             
                  │  ┌---------------------┐       ┌---------------------┐  │
                  │  │    Item Metadata    │       │     User Reviews    │  │
                  │  └----------┬----------┘       └----------┬----------┘  │
                  └─────────────┼─────────────────────────────┼─────────────┘
                                │                             │
                     ╔══════════▼══════════╗       ╔══════════▼══════════╗
                     ║      PCF Data       ║       ║  Data Preprocessing ║
                     ║     Augmentation    ║       ╚══════════┬══════════╝
                     ╚══════════┬══════════╝                  │
                                │                  ╔══════════▼══════════╗
                     ┌----------▼----------┐       ║    Model Training   ║
                     │      Augmented      │       ╚══════════┬══════════╝
                     │    Item Metadata    │                  │
                     └----------┬----------┘       ╔══════════▼══════════╗
                                │                  ║   Model Inference   ║
                                │                  ╚══════════┬══════════╝
                                │                             │
                                │                  ┌----------▼----------┐
                                │                  │   Recommendations   │
                                │                  └----------┬----------┘
                                │                             │
                                └──────────────┬──────────────┘
                                               │
                                ╔══════════════▼══════════════╗
                                ║    Sustainability-Aware     ║
                                ║ Recommendations Re-Ranking  ║
                                ╚══════════════┬══════════════╝
                                               │
                                ┌--------------▼--------------┐
                                │  Re-Ranked Recommendations  │
                                └--------------┬--------------┘
                                               │
                                ╔══════════════▼══════════════╗
                                ║       Model Evaluation      ║
                                ╚═════════════════════════════╝

#### <code>Amazon Reviews'23</code>

#### <code>PCF Data Augmentation</code>

#### <code>Data Preprocessing</code>

#### <code>Model Training</code>

#### <code>Model Inference</code>

#### <code>Sustainability-Aware Recommendation Re-Ranking</code>

#### <code>Model Evaluation</code>
