# SDML Prompt Quality

**Neural network training for Stable Diffusion prompt quality prediction**

Trains deep learning models to predict image quality scores from text prompts, enabling high-throughput prompt optimization and filtering. Features custom tokenization, PyTorch-based training pipeline, and ONNX export for production deployment.

## 🎯 What This Does

A PyTorch-based training system that learns to predict Stable Diffusion image quality directly from prompt text, without generating actual images.

### For Production Users & Researchers

- **Quality prediction**: Train models to score prompts 0-1 based on expected image quality
- **Custom tokenization**: Advanced tokenizer supporting LoRA models, samplers, and technical parameters
- **ETL pipeline**: Automated data processing from image quality scores and prompt datasets
- **ONNX export**: Deploy trained models with FP16 precision for inference
- **High throughput**: Optimized for scoring millions of prompts efficiently
- **Integration ready**: Direct compatibility with [sdml-generator](https://github.com/Haalt/sdml-generator/): Consumes ONNX models for high-throughput prompt generation & scoring and other ecosystem tools

### Key Features

- **Advanced tokenization**: Handles complex prompt structures including LoRA weights, samplers, and parameters
- **Multi-modal inputs**: Processes prompt tokens, LoRA configurations, sampler settings, and generation parameters
- **Production export**: ONNX models with FP16 precision for deployment
- **Ecosystem integration**: Seamless data flow from [sd-image-viewer](https://github.com/Haalt/sd-image-viewer) to prompt generator

## 🔬 How It Works

Multi-stage training pipeline:

```
Data ETL → Tokenization → Model Training → ONNX Export
```

### 1. **Data Processing (ETL)**

- Consumes prompt datasets from [sd-image-viewer](https://github.com/Haalt/sd-image-viewer) SQLite database and legacy JSON files.
- Merges image quality predictions from [sdml-image-classifier](https://github.com/Haalt/sdml-image-classifier/)
- Creates training pairs: (prompt_structure, quality_score)
- Handles data cleaning and filtering

### 2. **Advanced Tokenization**

- Custom tokenizer supporting complex prompt structures
- LoRA model recognition with weight extraction
- Sampler and upscaler parameter handling
- Special token management (negative prompt support planned)

### 3. **Model Training**

- V3 model architecture (non-transformer) for production release
- Multi-modal embedding approach: tokens, LoRAs, samplers, upscalers
- Advanced optimization with learning rate scheduling and early stopping
- V4 model with transformer architecture under development

### 4. **Production Export**

- ONNX export with FP16 precision support
- Compatibility with sdml-generator for high-throughput inference

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9+ with PyTorch 2.6+
- **GPU**: NVIDIA GPU with CUDA support (training)
- **Data**: Prompts dataset database from [sd-image-viewer](https://github.com/Haalt/sd-image-viewer) or compatible format
- **Data**: Image quality predictions from [sdml-image-classifier](https://github.com/Haalt/sdml-image-classifier)
- **Storage**: SSD recommended for large dataset processing

### Installation

```bash
# Install the package
pip install -e .

# For ONNX export (optional)
pip install -e .[onnx]

# Verify installation
sdpq --help
```

### Basic Usage

**Train a model:**

```bash
sdpq train
```

**Export to ONNX:**

```bash
sdpq export-onnx --checkpoint best_model.pt --output model.onnx --fp16
```

## 📊 Performance

**Training Performance** (RTX 4070):

- **Memory efficiency**: 8GB VRAM for batch size 256
- **Convergence**: Typically 30-50 epochs for stable results
- **Early stopping**: Patience of 8 epochs on validation MSE

**Inference Performance**:

- **ONNX export**: FP16 precision for production deployment
- **Memory**: Efficient GPU utilization during inference

## ⚙️ Configuration

The current implementation uses minimal configuration with sensible defaults:

**Training Parameters:**

- Batch size: 256
- Learning rate: 3e-4 (base), 1e-3 (attention layers)
- Weight decay: 1e-4
- Max epochs: 150
- Early stopping patience: 8
- Dropout: 0.4

**Model Architecture (V3):**

- Token embedding dimension: 256
- LoRA embedding dimension: 128
- Head hidden dimensions: 512, 160
- Multi-modal fusion of tokens, LoRAs, samplers, upscalers

**Data Processing:**

- Train/validation/test split handled automatically
- Bad prompt filtering via `bad.json` exclusion list
- Tokenizer trained on dataset or loaded from saved file

## 🛠️ CLI Commands

### Model Training

**Train the model:**

```bash
sdpq train
```

The training command automatically:

- Loads dataset from SQLite database (using `load_dataset_model_scores()`)
- Preprocesses data and creates train/validation splits
- Trains or loads existing tokenizer from `tokenizer/weighted_tokenizer.json`
- Saves best model as `best_model.pt`

### ONNX Export

**Export trained model to ONNX:**

```bash
sdpq export-onnx --checkpoint best_model.pt --output model.onnx
```

**Available export options:**

```bash
sdpq export-onnx \
  --checkpoint best_model.pt \
  --output model.onnx \
  --device cuda \
  --fp16 \
  --keep-io-types
```

- `--device`: Export device (cpu/cuda)
- `--fp16`: Convert to FP16 after export
- `--keep-io-types`: Keep I/O as FP32 when using FP16

### Data ETL

The ETL process is handled by the separate `etl.py` module:

```bash
python -m sdml_prompt_quality.data.etl --include-model-scores --predictions-file predictions.json
```

This merges image quality predictions with prompt data from [sd-image-viewer](https://github.com/Haalt/sd-image-viewer).

## 🔧 Technical Architecture

### Current Model: V3 (Production)

**Multi-Modal Embedding Architecture:**

The V3 model processes multiple input modalities simultaneously:

- **Token Path**: Prompt text → Token embeddings → Phi transformation → Masked pooling
- **LoRA Path**: LoRA IDs + weights → Embedding fusion → Weighted aggregation
- **Sampler Path**: Sampler ID + step buckets + log steps → Joint embedding
- **Upscaler Path**: Upscaler ID + denoise + steps → Parameter fusion

**Architecture Details:**

- Token embeddings (d_t=256) with learned transformations
- LoRA embeddings (d_L=128) with weight-aware fusion
- Sampler/upscaler embeddings with parameter conditioning
- Multi-layer head with batch normalization and dropout
- BCEWithLogitsLoss for binary quality prediction

**Model Inputs:**

- `tokens`: Tokenized prompt text (padded)
- `token_mask`: Valid token positions
- `lora_ids`: LoRA model identifiers
- `lora_w`: LoRA weights (normalized)
- `cfg`: CFG scale (normalized)
- `n_loras`: Number of LoRAs (normalized)
- `sampler_id`, `steps_log`, `steps_bucket`: Sampling parameters
- `upscaler_id`, `up_has`, `up_steps`, `denoise`: Upscaling parameters

### V4 Model (Under Development)

**Transformer-Enhanced Architecture:**

The V4 model adds transformer layers for improved prompt understanding:

- Self-attention mechanisms for token relationships
- Periodic encoding for scalar parameters
- Enhanced context modeling for complex prompts

### Training Pipeline

```
SQLite DB → ETL → Tokenization → DataLoader → V3/V4 Model → BCE Loss → AdamW
    ↓         ↓        ↓          ↓              ↓           ↓          ↓
sd-image-   Clean   Custom    Batched      Multi-modal   Binary    Differential
viewer DB   Filter   Tokens   Samples      Embeddings    Logits      LR
```

## 📈 Training Details

### Data Processing

- **ETL Pipeline**: Merges image classifier outputs with [sd-image-viewer](https://github.com/Haalt/sd-image-viewer) prompt database
- **Quality Filtering**: Bad prompt exclusion via `bad.json` filter list
- **Data Splitting**: Automatic train/validation/test splits
- **Tokenization**: Custom tokenizer for prompt structure and LoRA/sampler parameters

### Model Training

- **Loss Function**: BCEWithLogitsLoss for binary quality prediction
- **Optimization**: AdamW with differential learning rates (base: 3e-4, attention: 1e-3)
- **Scheduling**: Linear warmup (3 epochs) followed by cosine annealing
- **Regularization**: Dropout (0.4), weight decay (1e-4), early stopping
- **Monitoring**: Console logging of train/validation metrics per epoch

## 🔗 Ecosystem Integration

This project is part of the complete SDML pipeline:

- **[SDML Specs](https://github.com/Haalt/sdml-specs/)**: Schemas and format specifications
- **[SDML Image Classifier](https://github.com/Haalt/sdml-image-classifier/)**: Provides training labels via image quality scores
- **[SDML Rec](https://github.com/Haalt/sdml-rec/)**: Dataset format and efficient data loading
- **[SDML Generator](https://github.com/Haalt/sdml-generator/)**: Consumes ONNX models for high-throughput prompt generation & scoring
- **[SD Image Viewer](https://github.com/Haalt/sd-image-viewer)**: Labeling tools and prompt database source

### Data Flow

```
Images → Image Classifier → Quality Scores → Prompt Quality Training → ONNX Model → Generator
   ↓           ↓                 ↓                    ↓                    ↓           ↓
Labeled    Predictions      Training Data       Trained Model        Fast Scoring  High-Quality
Dataset    (JSON)           (Prompt,Score)      (PyTorch)            (ONNX)       Prompts
```
