# Pairwise Bayesian Optimization for Preference Elicitation

This repository contains the core implementation of the models and algorithms described in the paper: [Supporting High-Stakes Decision Making Through Interactive Preference Elicitation in the Latent Space](https://openreview.net/forum?id=ra7CSHcVCv) (ICLR 2026)

It provides a modular framework for performing pairwise Bayesian optimization (BO) to elicit user preferences. The repository supports both standard feature-space BO and latent-space BO via an autoencoder.

## Repository Structure

The codebase is organized into several key modules under `src/pairwise_bo/`:

- **`core.py`**: Contains the `PreferenceElicitator` and `AutoencoderPreferenceElicitator` classes. Implements the Pairwise Gaussian Process (PairwiseGP) and the Analytic Expected Utility of the Best Option (EUBO) acquisition function.
- **`autoencoder.py`**: Provides the `Autoencoder` class for compressing feature spaces into lower-dimensional latent representations for latent BO.
- **`data.py`**: Includes the `CsvDatasetLoader` for schema-driven loading, scaling, and bounding of tabular datasets.
- **`factory.py`**: Exposes the `get_elicitator` helper for instantiating the appropriate BO workflow.
- **`cli.py`**: Offers a command-line interface for running deterministic tests and executing the preference elicitation loop.

## Requirements and Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and requires **Python 3.10 or newer**.

1. Clone the repository and navigate to the project root.
2. Install the dependencies and synchronize the environment:
   ```bash
   uv sync
   ```

## Datasets

The models are designed to be evaluated using real-world preference datasets. A recommended public sample dataset is **Idealista18**, which contains real estate listings.

You can obtain the dataset from the following repository:
[paezha/idealista18](https://github.com/paezha/idealista18)

Download the dataset locally in CSV format before running the experiments.
We recommend using the "sale" datasets for the respective cities.

## Usage

You can run a deterministic end-to-end smoke iteration of the BO loop using the CLI. Provide the path to the dataset and specify the preset schema.

```bash
uv run pairwise-bo smoke --data-path /path/to/madrid_sale.csv --dataset-preset idealista
```

This command initializes the BO loop, proposes an optimal candidate pair using EUBO, simulates a user choice, updates the posterior, and ranks the available listings.

### LLM Elicitation Flow

The repository also includes an end-to-end evaluation flow where a Large Language Model (LLM) simulates a user's preferences based on a provided persona. The LLM initializes the utility weights and bounds, and actively selects the preferred listing at each iteration of the BO loop.

**Important**: Users must provide their own API token to use this script. By default, it uses Google's Gemini models, but also supports Ollama for local execution.

You can run the full LLM evaluation loop using the `llm-eval` command:

```bash
# Export your API key
export GEMINI_API_KEY="your-api-key"

# Run the LLM evaluation loop
uv run pairwise-bo llm-eval \
    --data-path /path/to/madrid_sale.csv \
    --dataset-preset idealista-madrid \
    --model "gemini:gemini-2.5-flash" \
    --persona "student" \
    --num-loops 15
```

### Statistical Evaluation Flow

To programmatically run the active learning BO algorithm without an LLM in the loop, use the `stat` command. It simulates choices deterministically using predefined ground-truth utility weights for various personas and records metrics (NDCG, Spearman, Pairwise Accuracy) at each step.

```bash
uv run pairwise-bo stat \
    --data-path /path/to/madrid_sale.csv \
    --dataset-preset idealista-madrid \
    --profile "budget_conscious" \
    --num-loops 50
```

This will print the final evaluation metrics to the console after all loops complete.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{eichelbeck2026supporting,
  title={Supporting High-Stakes Decision Making Through Interactive Preference Elicitation in the Latent Space},
  author={Eichelbeck, Michael and Voigt, Tim and Althoff, Matthias},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=ra7CSHcVCv}
}
```
