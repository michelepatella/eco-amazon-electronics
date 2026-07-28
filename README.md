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
<br>

<details>
  <summary><b><code>Amazon Reviews'23</code></b></summary>
  <br>
  
  > The pipeline leverages a **15-core filtered** version of the [**Amazon Reviews'23**](https://amazon-reviews-2023.github.io/) **Electronics** dataset, comprising **11,495 item metadata** and **464,464 reviews** from **21,751 users** (May 1996-September 2023).
</details>

<details>
  <summary><b><code>PCF Data Augmentation</code></b></summary>
  <br>
  
  > This step asynchronously augments **item metadata** with **carbon footprint information** through an [**AI agent**](https://github.com/michelepatella/reco2gnizer), implementing a semaphore mechanism with concurrency limits to guarantee **scalability** and **efficiency** when processing large-scale e-commerce catalogs.
</details>

<details>
  <summary><b><code>Data Preprocessing</code></b></summary>
  <br>
  
  > This stage processes **user reviews** by applying a **multi-stage pipeline**:
  > 1. **User-Item Interaction Deduplication**: Multiple reviews for the same product by the same user are removed, keeping only the most recent reviews as the most representative indicator of the user's preference, ensuring data consistency.
>   2. **Mapping**: Converts original user and item identifiers into sequential, zero-based indices required by the recommendation algorithms employed in the subsequent pipeline stages.
>   3. **Rating Binarization**: As the pipeline is based on implicit feedbacks, ratings of user reviews are mapped from (0-5) to a binary 0/1 outcome by assigning a value of 1 to all reviews with a rating >= 5, and 0 otherwise.
>   4. **User-Aware Temporal Split**: For each user, reviews are split into train (80%, oldest interactions), validation (10%, more recent interactions), and test sets (10%, most recent interactions), replicating a realistic scenario and preventing temporal data leakage.
>
> 💡 **Note**: The cold-start problem is addressed by ensuring that the validation and test sets contain only interactions involving items already present in the train set, removing the others.

</details>

<details>
  <summary><b><code>Model Training</code></b></summary>
  <br>
  
  > This phase performs concurrent **hyperparameter optimization**—a grid-based variant generation strategy with an early-stopping scheduler (ASHA) to reduct computational cost—and **final training** of two collaborative filtering algorithms: **BPR** and **LightGCN**. 
</details>

<details>
  <summary><b><code>Model Inference</code></b></summary>
  <br>
  
  > This step generates the **top-100 user recommendations** using the trained models by performing a **full-sort inference** procedure over all the users contained in the test set. To ensure **memory efficiency** and **scalability**, users are processed in fixed-size batches.
</details>

<details>
  <summary><b><code>Sustainability-Aware Recommendation Re-Ranking</code></b></summary>
  <br>
  
  Write here...
</details>

<details>
  <summary><b><code>Model Evaluation</code></b></summary>
  <br>
  
  Write here...
</details>
