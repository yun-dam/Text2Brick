# Text2Brick

An AI-powered framework for converting natural language questions about building data into SPARQL queries using the Brick schema.

## Overview

Text2Brick is an iterative agent-based system that translates natural language questions about HVAC systems into executable SPARQL queries. The agent builds queries incrementally by exploring the Brick schema step-by-step, validating each fragment before assembling the final query.

**Key Features:**
- **Iterative Query Building**: Constructs SPARQL queries piece-by-piece with validation at each step
- **Query Decomposition**: Automatically breaks down complex questions into sensors, temporal constraints, and aggregations
- **Temporal Handling**: Intelligently applies time-based filters (latest, range, specific dates)
- **Auto-correction**: Validates and fixes common SPARQL syntax errors
- **Few-shot Learning**: Optional support for example-based learning

## Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud account with Vertex AI access
- Google Cloud project ID (configured in `brick_agent.py:320`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Text2Brick

# Install dependencies
pip install -r requirements.txt
```

### Test a Single Question

The easiest way to test the system is using `run_single_question.py`:

```bash
# Basic usage
python run_single_question.py "What is the latest room temperature?"

# With few-shot examples
python run_single_question.py "Show me discharge air temperature over time" --fewshot

# Disable query decomposer
python run_single_question.py "Get fan speed" --no-decomposer

# Disable temporal handler
python run_single_question.py "Show temperature data" --no-temporal

# Custom output location
python run_single_question.py "Compare temperatures" --output logs/my_test.json

# Limit CSV rows for faster loading
python run_single_question.py "What is the outdoor air temperature?" --max-rows 1000
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `question` | Natural language question (required) | - |
| `--fewshot` | Enable few-shot examples | Disabled |
| `--no-decomposer` | Disable query decomposer | Enabled |
| `--no-temporal` | Disable temporal handler | Enabled |
| `--engine` | LLM engine to use | `gemini-flash` |
| `--output` | Output log file path | Auto-generated |
| `--max-rows` | Maximum CSV rows to load | 525600 |

### Example Questions

```bash
# Latest values
python run_single_question.py "What is the latest room temperature?"

# Time-based queries
python run_single_question.py "Show me discharge air temperature on Nov 15th at 2pm" --fewshot

# Comparisons
python run_single_question.py "Compare discharge and return air temperatures" --fewshot

# Aggregations
python run_single_question.py "What was the average outdoor air temperature in November?"

# Trends over time
python run_single_question.py "Show me room temperature over time" --fewshot
```

## Project Structure

```
Text2Brick/
├── brick_agent.py              # Main agent class and orchestration
├── brick_decomposer.py         # Query decomposition logic
├── brick_temporal_handler.py   # Temporal constraint handling
├── brick_utils.py              # Brick schema utilities
├── run_single_question.py      # Single question evaluation CLI
├── prompts/                    # Prompt templates
│   ├── brick_controller.prompt
│   ├── brick_decomposition.prompt
│   ├── brick_temporal_patterns.txt
│   ├── brick_format_actions.prompt
│   └── brick_controller_fewshot_examples.txt
├── LBNL_FDD_Data_Sets_FCU_ttl.ttl       # Brick schema (FCU dataset)
├── LBNL_FDD_Dataset_FCU/                # Timeseries data
│   └── FCU_FaultFree.csv
├── requirements.txt            # Python dependencies
└── logs/                       # Auto-generated execution logs
```

## How It Works

### 1. Query Decomposition (Optional)
The system first analyzes the question to identify:
- **Sensors**: Which sensors are referenced (e.g., room temperature, discharge air)
- **Temporal constraints**: Time-based requirements (latest, specific date, range)
- **Aggregations**: Operations needed (MIN, MAX, AVG, COUNT)

### 2. Iterative SPARQL Generation
The agent builds queries incrementally using these actions:
- `search_brick(string)`: Search for Brick entities/sensors
- `get_brick_entity(id)`: Explore entity properties and relationships
- `get_property_examples(property)`: See how properties are used
- `execute_sparql(query)`: Test SPARQL fragments
- `stop()`: Finalize the answer

### 3. Temporal Constraint Application
After generating the base query, temporal constraints are applied:
- **latest**: `ORDER BY DESC(?timestamp) LIMIT 1`
- **range**: `FILTER (?timestamp >= "start" && ?timestamp <= "end")`
- **specific**: `FILTER (?timestamp = "exact_time")`

### 4. Validation & Auto-correction
The system automatically:
- Adds missing PREFIX declarations
- Corrects malformed PREFIX URIs
- Fixes timestamp formatting issues
- Adds LIMIT clauses where appropriate

## Dataset Information

The system uses the **LBNL Fault Detection and Diagnosis (FDD) Dataset** for Fan Coil Units:
- **Temporal Coverage**: January 1, 2018 - December 31, 2018
- **Default Year**: When dates lack a year (e.g., "November 15th"), the system assumes 2018
- **Sensors**: Includes temperature, humidity, flow, and control sensors

### Available Sensors

| Sensor ID | Description |
|-----------|-------------|
| `FCU_OAT` | Outdoor Air Temperature |
| `FCU_RAT` | Return Air Temperature |
| `FCU_DAT` | Discharge Air Temperature |
| `FCU_MAT` | Mixed Air Temperature |
| `RM_TEMP` | Room Temperature |
| `FCU_DA_CFM` | Discharge Air Flow |
| `FCU_OA_CFM` | Outdoor Air Flow |
| And more... | See TTL file for complete list |

## Output & Logs

Each run generates detailed logs in the `logs/` directory:

```json
{
  "timestamp": "2025-11-26T10:30:00",
  "question": "What is the latest room temperature?",
  "configuration": {
    "engine": "gemini-flash",
    "fewshot": true,
    "decomposer": true,
    "temporal_handler": true
  },
  "execution_time_seconds": 12.5,
  "num_iterations": 5,
  "status": "success",
  "actions": [...],
  "final_sparql": "PREFIX brick: ...",
  "final_results": [...]
}
```

## Configuration

### Google Cloud Setup

1. Update your GCP project ID in `brick_agent.py` line 320:
```python
vertexai.init(project="your-project-id", location="us-central1")
```

2. Authenticate with Google Cloud:
```bash
gcloud auth application-default login
```

### Caching

The system automatically caches the loaded Brick graph to speed up subsequent runs:
- Cache file: `brick_graph_cache_<rows>rows.ttl`
- To force rebuild: Delete cache file or set `use_cache=False` in code

## Advanced Usage

### Interactive Mode

Run the agent interactively by executing the main script:

```bash
python brick_agent.py
```

This starts a session where you can ask multiple questions sequentially.

### Programmatic Usage

```python
from brick_agent import BrickAgent

# Initialize agent
agent = BrickAgent(
    engine="gemini-flash",
    use_decomposer=True,
    use_temporal_handler=True,
    use_fewshot=True
)

# Load data
agent.initialize_graph(
    ttl_file="LBNL_FDD_Data_Sets_FCU_ttl.ttl",
    csv_file="LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv",
    max_csv_rows=10000
)

# Ask a question
state, final_sparql = agent.run("What is the latest outdoor air temperature?")

# Get results
if final_sparql and final_sparql.has_results():
    print(final_sparql.execution_result)
```

## Troubleshooting

### Common Issues

**Issue**: `429 Too Many Requests` error
- **Solution**: The system includes rate limiting (6.5s delay + 60s cooldown). Ensure you're not hitting quota limits.

**Issue**: No results returned
- **Solution**: Check that dates match the dataset year (2018). Try adding `--fewshot` for better performance.

**Issue**: SPARQL syntax errors
- **Solution**: The system auto-corrects common errors, but complex queries may need manual adjustment.

**Issue**: Slow first run
- **Solution**: First run processes CSV and builds cache. Subsequent runs load from cache instantly.

## Performance Notes

- **First Run**: ~30-60 seconds (builds graph + processes CSV)
- **Cached Runs**: ~2-5 seconds (loads from cache)
- **Query Execution**: ~10-30 seconds per question (includes LLM calls)
- **Rate Limits**: System enforces delays to respect Vertex AI quotas

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional Brick schemas
- More comprehensive few-shot examples
- Support for other LLM backends (OpenAI, Anthropic)
- Query optimization strategies
- Extended temporal pattern support

## License

[Add your license information here]

## Citation

If you use this work, please cite:
```
[Add citation information if applicable]
```

## Acknowledgments

- Built on the [Brick Schema](https://brickschema.org/) standard
- Uses LBNL FDD Dataset for Fan Coil Units
- Powered by Google Gemini via Vertex AI
