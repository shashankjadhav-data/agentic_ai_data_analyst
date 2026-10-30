# Agentic AI for Autonomous Data Analysis

**A multi-agent LLM framework with iterative refinement, built on LangGraph and local open-source models (Llama 3.1 8B via Ollama).**

> Internship research project, MSc Big Data & Artificial Intelligence, SRH University of Applied Sciences, Berlin.
> **Author:** Shashank Valmik Jadhav  
> **Advisor:** Prof. Dr. Ajinkya Prabhune

---

## Overview

Classic AutoML tools (Auto-sklearn, H2O AutoML, AutoGluon, TPOT) automate model selection and tuning, but they need a human to supply the prediction target and task type first. They also offer little exploratory analysis, little narrative reporting, and no self-evaluation.

This project closes that gap. Give it a raw CSV and an optional one-line business context. The system then:

1. Profiles the dataset with no prior schema knowledge.
2. Decides which analytical task fits (classification, regression or clustering) and picks a target column.
3. Generates and runs Python code for each analysis step, with automatic error recovery.
4. Critiques its own results and re-plans if quality is insufficient (up to 3 refinement cycles).
5. Writes a business-friendly HTML report with metrics and visualizations.

It uses a **hybrid reasoning-execution** design. The LLM handles reasoning (interpretation, planning, critique, narrative). Generated Python code handles numerical work (statistics, model training, plotting), which limits numerical hallucination.

Everything runs locally on open-source components. No proprietary API or cloud service is required.

## Key Results

| Metric | Result |
|---|---|
| Task-detection accuracy | Correct on all 3 test scenarios |
| Code execution success (first attempt) | 68% |
| Code execution success (after up to 3 retries) | **91%** |
| Critique agent flagged genuine issues | 35% of initial executions |
| Average model accuracy gain from refinement | +5 to 12 percentage points |
| Visualization completeness | 85% to 98% |
| Expert report ratings (1-5) | Clarity 4.2, Completeness 3.8, Actionability 3.5 |
| End-to-end runtime (10K rows) | about 8-12 minutes |
| LLM throughput (Llama 3.1 8B, RTX 3080) | about 15 tokens/s |
| Peak memory | 4-6 GB |

Evaluation datasets: IMDB movie reviews (sentiment classification), an e-commerce customer dataset (segmentation), and a retail sales dataset (regression on total sales).

## Architecture

Five specialized agents share one state object and run as a LangGraph `StateGraph`:

```
 CSV + optional business context
            |
            v
   +-------------------+
   | 1. Ingestion &    |  pandas profiling + LLM summary
   |    Understanding  |
   +---------+---------+
             v
   +-------------------+ <-----------------------------+
   | 2. Planner        |  task detection, target       |
   |                   |  selection, plan generation   |
   +---------+---------+                               |
             v                                         |
   +-------------------+                               | REFINEMENT_NEEDED: true
   | 3. Analysis       |  code generation, subprocess  |  (max 3 iterations)
   |    Executor       |  execution, retry on error    |
   +---------+---------+                               |
             v                                         |
   +-------------------+                               |
   | 4. Critique &     |  warnings + LLM quality check-+
   |    Reflection     |
   +---------+---------+
             | REFINEMENT_NEEDED: false (or iteration limit reached)
             v
   +-------------------+
   | 5. Reporting      |  metrics table + narrative -> HTML report
   +-------------------+
```

| Agent | Responsibility |
|---|---|
| **Data Ingestion & Understanding** | Loads the CSV and computes a programmatic profile (columns, dtypes, missing values, unique counts, shape). The LLM then writes a natural-language summary of data quality and analytical opportunities. |
| **Planner** | Detects the task type from business-context keywords and column properties. Selects a target column (regex on the context, or a low-cardinality categorical heuristic). Builds a step plan from templates (EDA, then a baseline RandomForest, then an AutoGluon model). |
| **Analysis Executor** | Asks the LLM for Python code for each step and runs it in a sandboxed subprocess (300 s timeout). On failure it retries up to 3 times, feeding the stderr back to the LLM. It parses metrics from stdout and collects plots. |
| **Critique & Reflection** | Collects warnings (failed steps, ML steps without plots). The LLM judges the results and ends with `REFINEMENT_NEEDED: true/false`. Routes back to the Planner or on to Reporting. |
| **Reporting** | Builds a metrics table and an LLM-written narrative (executive summary, dataset overview, methodology, findings, recommendations) into an HTML report with visualizations. |

### Shared State

```python
class AgentState(TypedDict):
    csv_path: str
    business_context: Optional[str]
    data_profile: Optional[dict]
    analysis_plan: Optional[List[dict]]
    execution_results: Optional[List[dict]]
    critique_feedback: Optional[str]
    refinement_iteration: int
    final_report: Optional[str]
    logs: List[dict]
```

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph (>= 0.0.20), LangChain-Ollama (>= 0.0.2) |
| LLM runtime | Ollama with Llama 3.1 8B (quantized), temperature 0.2, 120 s timeout per call |
| Data science | pandas (>= 2.0.0), NumPy (>= 1.24.0), scikit-learn (>= 1.3.0), AutoGluon (>= 1.0.0) |
| Visualization | matplotlib (>= 3.7.0), seaborn (>= 0.12.0) |

## Project Structure

```
.
├── state.py     # AgentState TypedDict schema
├── utils.py     # Data profiling, CodeExecutor (subprocess), EvaluationLogger
├── agents.py    # Implementations of the five agents
├── main.py      # LangGraph workflow definition and entry point
└── README.md
```

## Getting Started

> The commands below are a standard setup based on the report's stack. Adjust file names, flags and the requirements file to match your repository.

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A GPU is recommended (the report used an RTX 3080), but not strictly required

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install langgraph langchain-ollama pandas numpy scikit-learn autogluon matplotlib seaborn

ollama pull llama3.1:8b
```

### Usage

```bash
python main.py --csv path/to/data.csv --context "segment customers by purchase behavior"
```

The business context is optional. Keywords steer task detection:

| Context keywords | Detected task |
|---|---|
| classify, classification, sentiment, category | Classification |
| predict, forecast, price, revenue | Regression |
| cluster, segment, group | Clustering |
| none matched | Clustering (default) |

You can also name the target directly, for example `"target: sentiment"` or `"predict total_sales"`.

## Scalability

| Dataset size | Behavior |
|---|---|
| 1K to 10K rows | Fully functional |
| 10K to 100K rows | Automatic sampling in ML steps |
| 100K to 1M rows | Sampling essential; distributed processing suggested |

## Limitations

- **Analysis scope:** only basic feature engineering; no causal inference, SHAP/LIME explanations or hypothesis testing.
- **Data scope:** tabular CSV only. Time-series forecasting, NLP on text corpora, computer vision and streaming are out of scope.
- **Unresolved failures (9%):** very high-cardinality features causing memory errors, complex nested data structures, and rare package-version conflicts.
- **LLM issues:** occasional hallucinated or semantically flawed code, context-window truncation of large profiles, and variable critique quality.
- **Reports:** limited domain-specific interpretation, some redundancy, and generic recommendations.
- **Evaluation:** exploratory analyses have no objective ground truth, and results depend on data quality.
- **Task detection:** it is heuristic (keywords and cardinality rules), so it is not guaranteed to generalize to unseen datasets.

## Comparison with Related Systems

| System | Autonomous planning | Reflection | Code generation | Reporting | Open source |
|---|---|---|---|---|---|
| Auto-sklearn | No | No | No | Minimal | Yes |
| H2O AutoML | No | No | No | Basic | Yes |
| AutoGluon | Partial | No | No | Basic | Yes |
| Tellius Agent | Yes (reactive) | No | Yes | Yes | No |
| **This work** | **Yes (proactive)** | **Yes** | **Yes** | **Yes** | **Yes** |

## Roadmap

- Advanced feature engineering (AutoFeat)
- Causal discovery (DoWhy, CausalML)
- Explainability (SHAP/LIME) and a statistical-testing agent
- Larger or code-specialized models (Llama 3.1 70B, CodeLlama) and fine-tuning on data-analysis examples
- Multi-modal and time-series support
- Human-in-the-loop refinement
- Distributed processing (Dask/Spark), Docker packaging, a REST API for BI tools, and observability

## Key References

- Hutter, Kotthoff, Vanschoren (eds.), *Automated Machine Learning: Methods, Systems, Challenges*, Springer, 2019
- Erickson et al., *AutoGluon-Tabular*, arXiv:2003.06505
- Wu et al., *AutoGen*, arXiv:2308.08155
- Shinn et al., *Reflexion*, arXiv:2303.11366
- Xiong et al., *Watch Every Step! Iterative Step-Level Process Refinement*, EMNLP 2024
- Chen et al., *Evaluating Large Language Models Trained on Code*, arXiv:2107.03374
- LangChain AI, *LangGraph*

## Citation

```bibtex
@misc{jadhav2025agenticdataanalyst,
  author = {Jadhav, Shashank Valmik},
  title  = {Agentic AI System Composed of Multiple LLM-Based Agents for Autonomous Data Analysis},
  note   = {Internship research project, SRH University of Applied Sciences, Berlin. Advisor: Prof. Dr. Ajinkya Prabhune},
  year   = {2025}
}
```

## Acknowledgments

Thanks to Prof. Dr. Ajinkya Prabhune for guidance, and to the open-source community behind LangGraph, Ollama and Meta AI's Llama.

## License

Add a license (for example MIT) before publishing.

## Contact

Shashank Valmik Jadhav: shashankvalmik.jadhav@stud.srh-university.de
