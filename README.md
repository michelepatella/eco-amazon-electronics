<div align="center">

  [![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
  [![RecBole](https://img.shields.io/badge/RecBole-E35A3C?style=for-the-badge&logo=https://recbole.io/docs/_images/logo.png)](https://recbole.io/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Ray](https://img.shields.io/badge/Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white)](https://www.ray.io/)
  [![DVC](https://img.shields.io/badge/DVC-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
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

### <code>• Overview</code>

A **multi-objective recommendation pipeline** for **collaborative filtering** on **implicit feedback**, applied to the **Amazon Reviews'23 Electronics** dataset to demonstrate how augmenting e-commerce catalogs with **carbon footprint information** can be used to balance product relevance with environmental impact for **sustainable recommendations**, without compromising recommendation quality.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/feadaa66-b64d-45a6-8b38-dcd757b05f4c">
    <img src="https://github.com/user-attachments/assets/d910a38e-6f3c-44e1-b86b-9cabd2fb1380" width="40%">
  </picture>
</div>

🎉 Achieved a **25% reduction** in the carbon footprint of recommendation lists with only a **4% drop** in recommendation quality!

---

### <code>• Recommendation Pipeline — Case Study</code>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/47c57d15-f968-493c-adc5-86714a75f48d">
  <img src="https://github.com/user-attachments/assets/c0fb9218-61df-4c70-a4ef-d65367045b22" width="100%">
</picture>

<br>

<table border="0">
  <td style="border: none;">
    <b><code>◦ Amazon Reviews'23</code></b><br><br>
    The pipeline leverages a <b><i>15</i>-core filtered</b> version of the <a href="https://amazon-reviews-2023.github.io/"><b>Amazon Reviews'23</b></a> <b>Electronics</b> dataset, comprising metadata for <b>11,495 items</b> and <b>464,464 reviews</b> from <b>21,751 users</b> (May 1996 – September 2023).
  </td>
</table>

<table border="0">
  <td style="border: none;">
    <b><code>◦ PCF Data Augmentation</code></b><br><br>
    Asynchronously augments <b>item metadata</b> with <b>carbon footprint information</b> through an <a href="https://github.com/michelepatella/reco2gnizer"><b>AI agent</b></a> powered by Google Gemini 2.5 Flash (max 5 web search results and 3 searches with <code>auto</code> modality), implementing a semaphore mechanism with concurrency limits to guarantee <b>scalability</b> and <b>efficiency</b> when processing large-scale e-commerce catalogs.<br><br>
    <details>
      <summary><b>Does the AI Agent Reliably Estimate PCF?</b></summary>
      <br>
      <blockquote>
        As a preliminary step, the <b>quality of the agent's PCF estimates</b> is evaluated across <b>three ground-truth datasets</b> (<i>Electronics</i>, <i>Clothing</i>, and <i>Home & Kitchen</i>, 194 real-world products each) using <b>OpenAI o3-mini</b> and <b>Google Gemini 2.5 Flash</b> (temperature 0.0) as <b>reasoning engines</b>.<br><br>
        The agent is benchmarked against its <b>zero-shot LLM baseline</b> using <a href="https://github.com/michelepatella/eco-amazon-electronics/blob/main/ZERO_SHOT_LLM_BASELINE_PROMPT.md"><b>this prompt</b></a>, collecting four estimates per product based solely on its title. Performance is measured via <b>estimation accuracy</b> (MAE, WAPE) and <b>ranking capacity</b> (Spearman's Rank Correlation Coefficient).<br><br>
        <b>Key Findings:</b>
<ul>
  <li>The <b>agent with Google Gemini 2.5 Flash</b> consistently <b>outperforms</b> its zero-shot baseline, improving <b>estimation accuracy</b> and <b>ranking capability</b> by up to <b>46%</b> and <b>9%</b>, respectively.</li>
  <li>The <b>agent with OpenAI o3-mini</b> shows <b>contrasting</b> and <b>domain-dependent behavior</b> (possibly due to <i>overthinking</i>): <b>estimation accuracy</b> can be <b>improved</b> by up to <b>79%</b> but can also <b>degrade</b> by up to <b>34%</b>; similarly, <b>ranking capability</b> can be <b>improved</b> by up to <b>16%</b> but can also <b>degrade</b> by up to <b>10%</b>.</li>
  <li>The <b>agent with Google Gemini 2.5 Flash</b> in the <i>Electronics</i> domain achieves a <b>37% estimation error</b> and a <b>0.84 ranking capability</b>, <b>outperforming</b> the <b>agent with OpenAI o3-mini</b> (<b>50%</b> and <b>0.72</b>) and justifying its adoption in the pipeline.</li>
</ul>
      </blockquote>
    </details>
  </td>
</table>

<table border="0">
  <td style="border: none;">
    <b><code>◦ Data Preprocessing</code></b><br><br>
    Processes <b>user reviews</b> through a <b>multi-stage pipeline</b>:<br>
    <ol>
      <li><b>User-Item Interaction Deduplication</b>: Retains only the most recent review per user-item pair to accurately represent current user preferences.</li>
      <li><b>Mappings</b>: Converts original user and item IDs into sequential, zero-based indices required by the recommendation algorithms.</li>
      <li><b>Rating Binarization</b>: Converts explicit ratings (1–5 scale) into implicit feedback (0/1 outcome) by assigning a value of 1 to reviews with a rating &ge; 5 (and 0 otherwise).</li>
      <li><b>User-Aware Temporal Split</b>: For each user, interactions are chronologically split into train (80%), validation (10%), and test (10%) sets to replicate a realistic evaluation scenario and prevent data leakage.</li>
      <li><b>Cold-Start Filtering</b>: Prevents evaluation on unseen items by ensuring validation and test sets contain only items present in the training set.</li>
    </ol>
    The <b>final user reviews dataset</b> contains <b>11,466 items</b>, <b>464,001 reviews</b>, and <b>21,751 users</b>.
  </td>
</table>

<table border="0">
  <td style="border: none;">
    <b><code>◦ Model Training</code></b><br><br>
    Performs concurrent <b>hyperparameter optimization</b> using Ray Tune (2 CPU cores, 10 trial samples, max 20 epochs with patience 5, grid search with ASHA early stopping, and NDCG@10 validation metric) and <b>final training</b> of two collaborative filtering algorithms (Adam optimizer and max 500 epochs with patience 15): <b>BPR</b> and <b>LightGCN</b>.<br><br>
    <div align="center">
      <table>
        <thead>
          <tr>
            <th align="left">Hyperparameter</th>
            <th align="center">BPR Search Space</th>
            <th align="center">LightGCN Search Space</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td align="left"><code>train_batch_size</code></td>
            <td align="center"><b>512</b>, 1024, 2048</td>
            <td align="center">512, <b>1024</b>, 2048</td>
          </tr>
          <tr>
            <td align="left"><code>learning_rate</code></td>
            <td align="center">0.003, 0.001, <b>0.0003</b></td>
            <td align="center"><b>0.003</b>, 0.001, 0.0003</td>
          </tr>
          <tr>
            <td align="left"><code>weight_decay</code></td>
            <td align="center"><b>1e-6</b>, 1e-5, 1e-4</td>
            <td align="center"><b>1e-6</b>, 1e-5, 1e-4</td>
          </tr>
          <tr>
            <td align="left"><code>embedding_size</code></td>
            <td align="center"><b>64</b>, 128, 256</td>
            <td align="center">64, <b>128</b>, 256</td>
          </tr>
          <tr>
            <td align="left"><code>n_layers</code></td>
            <td align="center">-</td>
            <td align="center">1, 2, <b>3</b></td>
          </tr>
          <tr>
            <td align="left"><code>reg_weight</code></td>
            <td align="center">-</td>
            <td align="center"><b>1e-5</b>, 1e-4</td>
          </tr>
        </tbody>
      </table>
      <sub><i>Note: Bold values represent the best hyperparameters found for BPR (epoch 30, NDCG@10=0.014336) and LightGCN (epoch 38, NDCG@10=0.015403).<br>Execution Time: ~6h for hyperparameter optimization, ~2h for final training.</i></sub>
    </div>
    
  </td>
</table>

<table border="0">
  <td style="border: none;">
    <b><code>◦ Model Inference</code></b><br><br>
    Generates the <b>top-100 user recommendations</b> using the trained models by performing a <b>full-sort inference</b> procedure across all users in the test set. To ensure <b>memory efficiency</b> and <b>scalability</b>, users are processed in fixed-size batches (batch size = 1,000).
  </td>
</table>

<table border="0">
  <td style="border: none;">
    <b><code>◦ Sustainability-Aware Recommendation Re-Ranking</code></b><br><br>
    Applies a <b>model-agnostic re-ranking</b> strategy to each user's top-100 recommendations to balance <b>item relevance</b> and <b>carbon footprint</b> via the <b>Sustainability-aware Score (SaS)</b>:
    <br><br>
    $$\text{SaS}(u, i) = \alpha \cdot \hat{r}(u, i) + (1 - \alpha) \cdot \hat{s}(i) \in [0, 1]$$
    <br><br>
    <ul>
      <li>$\hat{r}(u, i)$: Min-max normalized <b>recommendation score</b> from the recommender model.</li>
      <li>$\hat{s}(i)$: <b>Sustainability score</b>, computed from the estimated carbon footprint via $\log(1+x)$ scaling (emphasizes differences in lower emission ranges and mitigates high-emission outliers), inverse min-max normalization, and a cubic transformation (penalizes high-emission items).</li>
      <li>$\alpha \in \{0.25, 0.5, 0.75, 1.0\}$: <b>Weighting factor</b> controlling the trade-off between item relevance and sustainability (lower values prioritize sustainable items).</li>
    </ul>
  </td>
</table>

<table border="0">
  <td style="border: none;">
    <b><code>◦ Model Evaluation</code></b><br><br>
    Evaluates re-ranked recommendations against original ones in terms of <b>accuracy</b> (Recall@k), <b>ranking quality</b> (NDCG@k), <b>catalog diversity</b> (GiniIndex@k), <b>popularity bias</b> (AveragePopularity@k), and <b>carbon footprint</b> (Emissions@k), under different $\alpha$ and top- $k \in \{5, 10, 20\}$ scenarios and by applying the <b>AllItems</b> evaluation methodology.<br><br>
    <b>Key Findings:</b>
    <ul>
      <li><b>Product relevance</b> and <b>carbon impact</b> can be effectively balanced for <b>sustainable recommendations</b>, although this depends on the choice of the <b>weighting factor</b> $\alpha$.</li>
      <li>While <b>aggressive optimizations</b> ($\alpha &le; 0.5$) strongly degrade model performance, setting $\alpha=0.75$ (at $k=20$) yields the <b>best trade-off</b>.</li>
      <li>In this <b>optimal scenario</b>, <b>BPR</b> reduces the <b>carbon footprint</b> of recommendation lists by <b>25%</b> with only a <b>4% drop</b> in recommendation quality; <b>LightGCN</b>, which proves to be less sensitive to recommendation re-ranking, reduces the <b>carbon footprint</b> by <b>3%</b> with a <b>4% drop</b> in recommendation quality.</li>
      <li>When scaled across the <b>21,751 test users</b>, these <b>carbon footprint savings</b> correspond to <b>181</b> and <b>16 tons of CO₂e</b> saved for <b>BPR</b> and <b>LightGCN</b>, respectively.</li>
      <li>The <b>sustainability-aware recommendation re-ranking strategy</b> improves <b>popularity debiasing</b>, making models less prone to recommending <b>mainstream items</b>.</li>
    </ul>
  </td>
</table>

<br>

> [!NOTE]
> [**Data**](https://github.com/michelepatella/eco-amazon-electronics/tree/main/data), [**model artifacts**](https://github.com/michelepatella/eco-amazon-electronics/tree/main/models), and [**execution logs**](https://github.com/michelepatella/eco-amazon-electronics/tree/main/logs) (with full results) are tracked via **DVC**.
