# Homework 4

Homework 4 considers multivariate time series (multi input multi output). The datasets used here are `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Electricity`, `Traffic`, `Weather`, `Exchange`, `ILI`. Homework 4 is the final examination of our course. You need to choose one of the questions below to complete. PatchTST and Transformer have been implemented in our code. You can use them as baselines.
You can change the source code as needed. (such as Question 3, Question 5)

## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model PatchTST
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model Transformer
```

## Question 1 time series distillation (100 pts)

**Introduction:**

The main idea of knowledge distillation (available at [link](https://arxiv.org/abs/1503.02531)) is to transfer the knowledge from a teacher model to a student model, allowing the student model to acquire the capabilities of the teacher model. In some cases, the performance of the student model can even surpass that of the teacher model. As introduced in class, the PatchTST model (available at [link](https://arxiv.org/abs/2211.14730)), based on the Transformer architecture, achieves better performance compared to the original Transformer. In this context, we can treat PatchTST as a teacher model and transfer its knowledge to the traditional Transformer. Specifically, starting from a traditional teacher-student model framework, we can explore various distillation methods, such as utilizing data augmentation strategies to leverage the capabilities of the teacher model, aligning different network layers (e.g., output layer, hidden layers), modifying the structure of Transformer and using different distill loss functions, among others.

Baseline: PatchTST, Transformer

## Question 2 time series forecasting based on LLMs (100 pts)

**Introduction:**

The primary obstacle impeding the advancement of pre-trained models for time series analysis lies in the scarcity of extensive training data. However, can we harness the power of Large Language Models (LLMs), which have undergone training on billions of tokens, to elevate our model's performance? A notable example, "One Fits All" (available at [link](https://arxiv.org/abs/2302.11939)), illustrates how a pre-trained LLM, when equipped with input embedding and output head layers, can be employed to construct a highly effective time series forecasting model, yielding remarkable results. Although we may not possess the same level of hardware resources as outlined in their work, we can make use of chatGPT's capabilities via an interactive dialogue-based approach, thereby aiding us in achieving superior predictions, as what"PromptCast" (available at [link](https://arxiv.org/abs/2210.08964)) does. Within this framework, we have the opportunity to delve into time series forecasting methodologies that leverage the prowess of large language models. In this context, you need to explore different ways to incorporate the pre-trained LLMs into your time series forecasting model.

![streaming.jpg](imgs%2FPromptCast.png)

Baseline: PatchTST

## Question 3 data leakage of large time series models (100 pts)

**Introduction:**

In large language models, the extensive data used during the pre-training phase often forms the foundation of their powerful capabilities. However, due to the existence of evaluation metrics, if there is an issue of data leakage in the evaluation data, it could potentially make the evaluation of large language models less reliable. For instance, the paper "Training on the Benchmark Is Not All You Need" investigates the problem of pre-training data leakage in large language models based on the interchangeable nature of multiple-choice question options. Currently, time series pre-training models have become a hot topic, as seen in articles like "Timer: Generative Pre-trained Transformers Are Large Time Series Models," which propose time series pre-training models. 
We aim to explore whether these models also suffer from similar pre-training data leakage issues, as such exploration would contribute to the development of more equitable evaluation benchmarks.

## Question 4 time series ensemble learning (100 pts)

**Introduction:**

Ensemble learning, as an important branch of the machine learning field, is centered around the idea of combining the predictions of multiple models to enhance overall performance. This approach stems from the concept of "wisdom of the crowd," where the combination of multiple weak models can form a strong model, thereby exhibiting greater generalization capabilities and robustness in complex tasks. In the realm of time series forecasting, ensemble learning methods also demonstrate significant potential. By integrating the predictions of multiple models, ensemble learning can effectively reduce the bias and variance of individual models, thus improving overall prediction accuracy. However, how to select appropriate base models and design effective ensemble strategies remains a topic worthy of in-depth research. For instance, should homogeneous models (e.g., multiple LSTMs) be used for integration, or should heterogeneous models (e.g., combining LSTMs, Transformers, and traditional statistical models) be introduced to capture different time series characteristics? Additionally, the choice of ensemble methods (such as weighted averaging, stacking, or voting) can significantly impact the final results. We aim to explore what ensemble learning methods are optimal for time series tasks, as such exploration will contribute to providing more robust and efficient solutions for time series forecasting.

Baseline: PatchTST

## Question 5 Zero-shot Capability Preferences (100 pts)

**Introduction:**

As large models are increasingly applied in the field of time series forecasting, their predictive performance has generally reached or even surpassed that of traditional deep learning models. However, due to the unavoidable high computational costs during the pre-training and fine-tuning processes of large models, the zero-shot capability of time series forecasting models has become an important evaluation dimension. Zero-shot capability refers to the ability of a model to make accurate predictions on tasks it has never encountered before, i.e., without task-specific fine-tuning.

Currently, a number of time series forecasting models with strong zero-shot capabilities have emerged, such as Chronos, Moment, Timer, Moirai, VisionTS, TimeGPT, TimesFM, and others. However, in reality, even though these models have been fine-tuned with sufficient pre-training data, their zero-shot capabilities still exhibit biases. That is, some models perform exceptionally well on certain types of data but underperform in other domains or with different sequence lengths.

This topic focuses on exploring the biases of existing large time series forecasting models with zero-shot capabilities. Note that you do not need to compare the zero-shot performance of these models across all standard datasets. Instead, you should strive to uncover the preferences of different models for different types of data and summarize some simple patterns after in-depth analysis.

#### Task Guidelines

1. **Model Selection**: To make your conclusions more representative, select at least three large time series forecasting models with zero-shot capabilities for experimentation.
2. **Data Types**: When analyzing the preferences or limitations of each model in handling different types of time series data, you can use standard datasets for testing or simulate simple data (e.g., strong seasonality, simple trends, irregular residuals, or combinations thereof).
3. **Parameter Settings**: To ensure fairness in comparison, you should ensure that the `seq_len` and `pred_len` settings are the same for all models. Feel free to experiment with specific values.
4. **Evaluation Metrics**: To make the findings more interesting, you are not limited to using standard evaluation metrics (e.g., MSE, MAE). Using more evaluation dimensions is more likely to reveal interesting patterns.
5. **Visualization**: To make your conclusions more intuitive, you can use various plotting or statistical methods to highlight your findings.

## Submission

**1. Modified Code:**

- Provide the modified code for all components of the task.
- Include a `README.md` file in Markdown format that covers the entire task. This file should contain:
    - how to install any necessary dependencies for the entire task.
    - how to run the code and scripts to reproduce the reported results.
    - datasets used for testing in all parts of the task.

**2. PDF Report:**

- Create a detailed PDF report that encompasses the entire task. The report should include sections for each component
  of the task.

**3. Submission Format:**

- Submit the entire task, including all code and scripts, along with the `README.md` file and the PDF report, in a
  compressed archive (.zip).

**4. Submission Deadline:**
2025-02-10 23:55
