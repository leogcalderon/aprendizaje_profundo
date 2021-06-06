# 21. Recommender Systems

## 21.1 Matrix Factorization
The model factorizes the user-item interaction matrix (e.g., rating matrix) into the product of two lower-rank matrices, capturing the low-rank structure of the user-item interactions.

- Let **R ∈ Rm×n** denote the interaction matrix with **m** users and **n** items, and the values of **R** represent explicit ratings.

- The user-item interaction will be factorized into a user latent matrix **P ∈ Rm×k** and an item latent matrix **Q ∈ Rn×k**, where k≪m,n, is the latent factor size.

- Let **pu** denote the uth row of **P** and **qi** denote the ith row of **Q**. For a given item i, the elements of **qi** measure the extent to which the item possesses those characteristics such as the genres and languages of a movie. For a given user u, the elements of **pu** measure the extent of interest the user has in items’ corresponding characteristics. These latent factors might measure obvious dimensions as mentioned in those examples or are completely uninterpretable.

The predicted ratings can be estimated by: **R = PQ^T**

One major problem of this prediction rule is that users/items biases can not be modeled. For example, some users tend to give higher ratings or some items always get lower ratings due to poorer quality. These biases are commonplace in real-world applications. To capture these biases, user specific and item specific bias terms are introduced. Specifically, the predicted rating user u gives to item i is calculated by:

![](imgs/bias.png)

Then, we train the matrix factorization model by minimizing the mean squared error between predicted rating scores and real rating scores. The objective function is defined as follows:

![](imgs/objective.png)

## 21.2 AutoRec: Rating Prediction with Autoencoders
In AutoRec, instead of explicitly embedding users/items into low-dimensional space, it uses the column/row of the interaction matrix as the input, then reconstructs the interaction matrix in the output layer.

AutoRec focuses on learning/reconstructing the output layer. It uses a partially observed interaction matrix as the input, aiming to reconstruct a completed rating matrix. In the meantime, the missing entries of the input are filled in the output layer via reconstruction for the purpose of recommendation.

**Item based AutoRec:**
Let **R∗i** denote the ith column of the rating matrix, where unknown ratings are set to zeros by default. The neural architecture is defined as:

![](imgs/model.png)

where **f(⋅)** and **g(⋅)** represent activation functions, **W** and **V** are weight matrices, **μ** and **b** are biases. Let **h(⋅)** denote the whole network of AutoRec. The output **h(R∗i)** is the reconstruction of the ith column of the rating matrix.

The following objective function aims to minimize the reconstruction error:

![](imgs/objective2.png)

where **∥⋅∥O** means only the contribution of observed ratings are considered, that is, only weights that are associated with observed inputs are updated during back-propagation.

## 21.3 Personalized Ranking for Recommender Systems
