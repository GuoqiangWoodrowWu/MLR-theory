# Rethinking and Reweighting the Univariate Losses for Multi-Label Ranking: Consistency and Generalization
This repository is the official implementation of "Guoqiang Wu\*, Chongxuan Li\*, Kun Xu and Jun Zhu. Rethinking and Reweighting the Univariate Losses for Multi-Label Ranking: Consistency and Generalization" accepted at NeurIPS 2021.
## Programming Language
The source code is written by Matlab
## File description
- ./Datasets -- the benchmarks datasets downloaded from the websites http://mulan.sourceforge.net/datasets-mlc.html and http://palm.seu.edu.cn/zhangml/
- ./Datasets_synthetic -- the semi-synthetic datasets with randomly selected label size (i.e., c) based on the delicious datasets
- ./measures -- the measures for MLR on Ranking Loss
- ./Results -- store the experimental results
- ./CrossValidation.m -- used to create cross-validation data
- ./train_logistic_rank_SVRG_BB.m -- utilize SVRG-BB to train the model with surrogate pairwise loss (i.e. A^{pa}) where the base loss is logistic loss
- ./train_logistic_cost_sensitive_SVRG_BB.m -- utilize SVRG-BB to train the model with different surrogate univariate losses (including A^{k}, k = 1,2,3,4.) where the base loss is logistic loss
- ./calculate_cost_matrix.m -- calculate the cost matrix for corresponding univarite loss (including L_{u_k}, k = 1,2,3,4.)
- ./Predict_score.m -- predict the score function
- ./Evaluation_Metrics.m -- evaluate the model on Ranking Loss measure
- ./calculate_risk.m -- calculate the surrogate pairwise or univariate risk
- ./calculate_bound.m -- calculate the (probabilistic) upper bound of algorithms

- run_linear_pa.m -- run the code to evaluate A^{pa}
- run_linear_u1.m -- run the code to evaluate A^{u1}
- run_linear_u2.m -- run the code to evaluate A^{u2}
- run_linear_u3.m -- run the code to evaluate A^{u3}
- run_linear_u4.m -- run the code to evaluate A^{u4}
## Run
Run the run_linear_pa.m, run_linear_u1.m, run_linear_u2.m, run_linear_u3.m and run_linear_u4.m in MATLAB, and it will run as its default parameters on sample datasets.