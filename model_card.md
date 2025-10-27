# Model Card for RoBERTa-base-reddit-emotions

A RoBERTa-based model fine-tuned for multi-label emotion classification on Reddit text data.

## Model Details

### Model Description

This model is a fine-tuned version of `SamLowe/roberta-base-go_emotions` specifically trained for emotion classification using a reddit text dataset. It can predict 13 different emotion categories (12 emotions + neutral) from text input, making it suitable for understanding emotional content in social media posts, comments, and other informal text.

- **Developed by:** TENCFIM
- **Model type:** RoBERTa (Transformer-based language model for sequence classification)
- **Language(s) (NLP):** English
- **Finetuned from model:** SamLowe/roberta-base-go_emotions

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://huggingface.co/SamLowe/roberta-base-go_emotions

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

This model is designed for single-label emotion classification tasks where you need to identify one or more emotions expressed in text. It's particularly effective for:

- **Social media analysis**: Understanding emotional sentiment in Reddit comments, Twitter posts, and similar platforms
- **Content moderation**: Identifying potentially harmful emotional content
- **Mental health applications**: Analyzing emotional patterns in text (with appropriate caution and professional oversight)
- **Customer feedback analysis**: Understanding emotional responses in reviews and feedback
- **Research applications**: Academic and commercial research into emotion expression in text

The model outputs probability scores for 13 emotion categories, allowing for nuanced emotion detection.

### Downstream Use

This model can be fine-tuned further for:

- **Domain-specific emotion classification**: Adapting to specific platforms or text types
- **Multilingual emotion detection**: As a starting point for transfer learning to other languages
- **Emotion-aware chatbots**: Integrating emotional understanding into conversational AI
- **Content recommendation systems**: Using emotional content to improve recommendations

### Out-of-Scope Use

This model should not be used for:

- **Clinical diagnosis**: The model is not validated for medical or psychological diagnosis
- **High-stakes decision making**: Employment, insurance, or legal decisions should not be based solely on this model
- **Real-time crisis intervention**: The model may not detect urgent mental health situations reliably
- **Non-English text**: The model was trained exclusively on English data
- **Formal or academic text**: Performance may be degraded on text styles very different from Reddit comments
- **Individual surveillance**: The model should not be used to monitor individuals without consent

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model has several important limitations and potential biases:

**Dataset Biases:**
- **Reddit user demographics**: The training data reflects the demographics and perspectives of Reddit users, which may not represent the general population
- **Platform-specific language**: The model is optimized for informal, social media-style text and may perform poorly on formal writing
- **Cultural biases**: Emotion expression varies across cultures, but the model was trained primarily on English-speaking Reddit communities

**Technical Limitations:**
- **Multi-label complexity**: It only ouputs single emotions
- **Threshold dependency**: Performance varies significantly depending on the classification threshold used
- **Class imbalance**: Some emotions have very few training examples and show poor performance

**Ethical Considerations:**
- **Privacy concerns**: The training data contains real Reddit usernames and content
- **Misuse potential**: Emotion detection can be misused for manipulation or surveillance
- **Stereotyping risk**: The model may perpetuate biases about how different groups express emotions

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Specific recommendations include:

- **Use appropriate thresholds**: Evaluate and adjust classification thresholds based on your specific use case
- **Consider ensemble approaches**: Combine with other models or human review for high-stakes applications
- **Monitor for bias**: Regularly evaluate model performance across different demographic groups and text types
- **Provide transparency**: When using this model in applications, inform users about its limitations
- **Avoid over-reliance**: Use as a screening or support tool rather than the sole basis for important decisions

## How to Get Started with the Model


## Training Details

### Training Data

This model was trained on the **emotions-dataset**, a large-scale dataset of Reddit comments annotated for emotion. The dataset contains:

- **131k curated Reddit comments** labeled for 12 emotion categories plus neutral
- **Source**: Reddit comments collected through automated methods
- **Language**: English
- **Splits**: 
- **Multi-label**: Comments can have multiple emotion labels simultaneously

**Dataset**: [emotions-dataset](https://huggingface.co/datasets/boltuix/emotions-dataset)

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing


#### Training Hyperparameters


#### Speeds, Sizes, Times [optional]



## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data



#### Factors



#### Metrics


### Results

#### Summary


## Model Examination 

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions have been estimated using the CodeCarbon library in python.

- **Hardware Type**: GPU Power
- **Hours used**: 1.47 
- **Compute Region**: Andalusia
- **Carbon Emitted**: 0.0256252172527272 kg CO2e

The fine-tuning process used relatively modest computational resources compared to training from scratch, as it builds on the pre-trained RoBERTa base model.

## Technical Specifications [optional]

### Model Architecture and Objective


### Compute Infrastructure

#### Hardware


#### Software


## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc{samlowe2023roberta,
  author = {SamLowe},
  title = {roberta-base-go_emotions: A RoBERTa model for emotion classification},
  year = {2023},
  publisher = {Hugging Face},
  url = {https://huggingface.co/SamLowe/roberta-base-go_emotions}
}

@inproceedings{demszky2020goemotions,
  author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
  year = {2020}
}
```

**APA:**

SamLowe. (2023). roberta-base-go_emotions: A RoBERTa model for emotion classification. Hugging Face. https://huggingface.co/SamLowe/roberta-base-go_emotions

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. In 58th Annual Meeting of the Association for Computational Linguistics (ACL).



## More Information [optional]

For additional information and resources:

- **Original model page**: https://huggingface.co/SamLowe/roberta-base-go_emotions
- **GoEmotions dataset**: https://huggingface.co/datasets/google-research-datasets/go_emotions
- **Evaluation notebook**: https://github.com/samlowe/go_emotions-dataset/blob/main/eval-roberta-base-go_emotions.ipynb
- **ONNX optimized version**: https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx
- **Base model (RoBERTa)**: https://huggingface.co/roberta-base

## Model Card Authors 

This model card was created based on the model developed by TENCFIM and information available on the Hugging Face model hub.

## Model Card Contact

For questions about this model card or the model itself, please refer to:
- **Model developer**: TENCFIM
