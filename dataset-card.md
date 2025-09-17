---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
dataset_name: boltuix/emotions-dataset
pretty_name: Emotions Dataset
language:
	- en
modality:
	- text
task_categories:
	- text-classification
task_ids:
	- multi-class-classification
size_categories:
	- 100K<n<1M
license: mit
tags:
	- emotions
	- emotion-classification
	- sentiment-analysis
	- nlp
configs:
	- config_name: default
		splits:
			- name: train
				num_examples: 131306
dataset_format:
	- parquet
---

# Dataset Card for Emotions Dataset (boltuix/emotions-dataset)

The Emotions Dataset is a collection of 131,306 short English sentences labeled with 13 emotion categories. It is distributed in Parquet format with two columns: Sentence (string) and Label (string). The dataset is suitable for building and evaluating emotion classification and sentiment analysis systems.

## Dataset Details

### Dataset Description

This dataset provides single-label emotion annotations over short text snippets. The label set contains 13 emotions: Happiness, Sadness, Neutral, Anger, Love, Fear, Disgust, Confusion, Surprise, Shame, Guilt, Sarcasm, and Desire. The dataset page indicates there are no missing values and that the file is compact (about 7.41 MB as Parquet).

- Curated by: boltuix
- Language(s) (NLP): English
- License: MIT

### Dataset Sources

- Repository: https://huggingface.co/datasets/boltuix/emotions-dataset
- Demo: https://huggingface.co/datasets/boltuix/emotions-dataset/viewer/default/train

## Uses

### Direct Use

- Training and evaluating emotion classification models.
- Fine-tuning general-purpose text classifiers on emotion labels.
- Benchmarking and prototyping for sentiment and affective computing tasks.
- Exploratory analysis of emotion distributions in short English text.

### Out-of-Scope Use

- Clinical or diagnostic decision-making without domain oversight.
- High-stakes or safety-critical decisions based solely on emotion predictions.
- Identification or profiling of individuals or groups.
- Uses requiring demographic attributes; these are not provided.

## Dataset Structure

- Instances: 131,306 rows (single split: train)
- Features:
	- Sentence: string (short text)
	- Label: string (one of 13 emotion classes)
- Splits: train only (no official validation/test splits). Users should create their own splits.
- File format: Parquet

### Label Set and Reported Counts

Class counts reported on the dataset page (total 131,306):
- Happiness: 31,205 (23.76%)
- Sadness: 17,809 (13.56%)
- Neutral: 15,733 (11.98%)
- Anger: 13,341 (10.16%)
- Love: 10,512 (8.00%)
- Fear: 8,795 (6.70%)
- Disgust: 8,407 (6.40%)
- Confusion: 8,209 (6.25%)
- Surprise: 4,560 (3.47%)
- Shame: 4,248 (3.24%)
- Guilt: 3,470 (2.64%)
- Sarcasm: 2,534 (1.93%)
- Desire: 2,483 (1.89%)

Notes from the page: missing values are reported as zero; duplicate and unique row statistics are not confirmed and may require local analysis.

## Dataset Creation

### Curation Rationale

Provide a medium-scale, compact dataset for multi-class emotion classification in English suitable for training and benchmarking.

### Source Data

The dataset page describes text as user-generated content aggregated and curated by the maintainer. Specific upstream datasets or collection sources are not enumerated.

#### Data Collection and Processing

- Distribution format: Parquet (auto-converted on the Hugging Face Hub).
- Preprocessing details, filtering, and deduplication procedures are not documented on the page.
- Average sentence length is described as roughly 14 words on the page; users should verify locally.

#### Who are the source data producers?

Not specified. The data appears to originate from user-generated text; no demographic metadata is included.

### Annotations

#### Annotation process

Labels are provided with the dataset. The page does not supply details about annotation guidelines, tooling, inter-annotator agreement, or validation procedures.

#### Who are the annotators?

Not specified on the dataset page.

#### Personal and Sensitive Information

Short text entries may include personal or sensitive content typical of user-generated text. No explicit PII removal process is described. Users should assess and mitigate privacy risks in their applications.

## Bias, Risks, and Limitations

- Class imbalance exists (e.g., Happiness has the largest share; several classes have relatively few examples such as Sarcasm and Desire).
- Only English is included; results may not transfer to other languages or domains.
- The dataset provides a single train split without an official validation/test partition.
- Upstream data sources and annotation methodology are not fully documented, which can affect reproducibility and bias assessment.
- Potential presence of toxicity, offensive language, or sensitive topics given the nature of user-generated text.

### Recommendations

- Create stratified train/validation/test splits appropriate for your use case.
- Consider class rebalancing techniques (e.g., weighting, resampling) and report per-class metrics.
- Perform deduplication and basic text hygiene as needed; verify assumptions made on the dataset page (duplicates, average length).
- Conduct domain adaptation or additional evaluation if deploying beyond the dataset’s domain.
- Implement content filters and privacy safeguards where appropriate.

## Citation

If you use this dataset, please cite the Hugging Face dataset page:

**BibTeX:**

@dataset{boltuix_emotions_2025,
	title        = {Emotions Dataset},
	author       = {boltuix},
	year         = {2025},
	publisher    = {Hugging Face},
	howpublished = {\url{https://huggingface.co/datasets/boltuix/emotions-dataset}},
	license      = {MIT}
}

**APA:**

boltuix. (2025). Emotions Dataset [Dataset]. Hugging Face. https://huggingface.co/datasets/boltuix/emotions-dataset

## Glossary

- Emotion classification: Assigning a discrete emotion label to a text input.
- Sarcasm: A figurative use of language where the literal sentiment differs from the intended sentiment; can be challenging for models.

## More Information

- Dataset page: https://huggingface.co/datasets/boltuix/emotions-dataset
- Viewer: https://huggingface.co/datasets/boltuix/emotions-dataset/viewer/default/train
- License: MIT (https://opensource.org/licenses/MIT)

## Dataset Card Authors

Originally shared by: boltuix. Card updated to provide a concise, emoji-free summary and risk guidance.

## Dataset Card Contact

Hugging Face user: @boltuix (see dataset page for discussions and issues)