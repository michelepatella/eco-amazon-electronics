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
        🌱 Multi-objective recommendation pipeline<br>
        balancing item relevance with environmental impact<br>
        for sustainable recommendations.
      </td>
    </tr>
  </table>

  <a href="https://github.com/michelepatella/eco-amazon-electronics">**GitHub**</a> •
  <a href="https://dagshub.com/michelepatella/eco-amazon-electronics">**DagsHub**</a>

</div>

---

A **multi-objective recommendation pipeline** for **collaborative filtering** on **implicit feedback**, applied to the **Amazon Reviews'23 Electronics** dataset to demonstrate how augmenting e-commerce catalogs with **carbon footprint information** can be used to balance product relevance with environmental impact for **sustainable recommendations**.

🎉 **Key Result:** Achieved a **25% reduction** in the carbon footprint of recommendation lists with only a **4% drop** in recommendation quality!

<br>

### <code>Recommendation Pipeline</code>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/47c57d15-f968-493c-adc5-86714a75f48d">
  <img alt="Pipeline Diagram" src="https://github.com/user-attachments/assets/c0fb9218-61df-4c70-a4ef-d65367045b22" width="100%">
</picture>

<br>

#### <b><code>Amazon Reviews'23</code></b>

The pipeline leverages a **_15_-core filtered** version of the [**Amazon Reviews'23**](https://amazon-reviews-2023.github.io/) **Electronics** dataset, comprising metadata for **11,495 items** and **464,464 reviews** from **21,751 users** (May 1996 – September 2023).

#### <b><code>PCF Data Augmentation</code></b>

Asynchronously augments **item metadata** with **carbon footprint information** through an [**AI agent**](https://github.com/michelepatella/reco2gnizer), implementing a semaphore mechanism with concurrency limits to guarantee **scalability** and **efficiency** when processing large-scale e-commerce catalogs.

#### <b><code>Data Preprocessing</code></b>

Processes **user reviews** through a **multi-stage pipeline**:
1. **User-Item Interaction Deduplication**: Retains only the most recent review per user-item pair to accurately represent current user preferences.
2. **Mappings**: Converts original user and item IDs into sequential, zero-based indices required by the recommendation algorithms.
3. **Rating Binarization**: Converts explicit ratings (1–5 scale) into implicit feedback (0/1 outcome) by assigning a value of 1 to reviews with a rating $\ge 5$ (and 0 otherwise).
4. **User-Aware Temporal Split**: For each user, interactions are chronologically split into train (80%), validation (10%), and test (10%) sets to replicate a realistic evaluation scenario and prevent data leakage.
  
> [!NOTE]
> Prevents cold-start items in evaluation by ensuring validation and test sets contain only items present in the training set.

#### <b><code>Model Training</code></b>

Performs concurrent **hyperparameter optimization**—using a grid-based variant generation strategy with an early-stopping scheduler (ASHA) to reduce computational cost—and **final training** of two collaborative filtering algorithms: **BPR** and **LightGCN**.

#### <b><code>Model Inference</code></b>

Generates the **top-100 user recommendations** using the trained models by performing a **full-sort inference** procedure across all users in the test set. To ensure **memory efficiency** and **scalability**, users are processed in fixed-size batches.

#### <b><code>Sustainability-Aware Recommendation Re-Ranking</code></b>

Applies a **model-agnostic re-ranking** strategy to each user's top-100 recommendations to balance **item relevance** and **carbon footprint** via the **Sustainability-aware Score (SaS)**:

$$\text{SaS}(u, i) = \alpha \cdot \hat{r}(u, i) + (1 - \alpha) \cdot \hat{s}(i) \in [0, 1]$$
* $\hat{r}(u, i)$: Min-max normalized **recommendation score** from the recommender model.
* $\hat{s}(i)$: **Sustainability score**, computed from the estimated carbon footprint via $\log(1+x)$ scaling (emphasizes differences in lower emission ranges and mitigates high-emission outliers), inverse min-max normalization, and a cubic transformation (penalizes high-emission items).
* $\alpha \in \{0.25, 0.5, 0.75, 1.0\}$: **Weighting factor** controlling the trade-off between item relevance and sustainability (lower values prioritize sustainable items).

#### <b><code>Model Evaluation</code></b>

Evaluates re-ranked recommendations against original ones in terms of **accuracy** (Recall@k), **ranking quality** (NDCG@k), **catalog diversity** (GiniIndex@k), **popularity bias** (AveragePopularity@k), and **carbon fooprint** (Emissions@k), under different $$\alpha$$ and top-_k_ scenarios and applying the **AllItems** evaluation methodology.

