# 🤖 Multi-Agent Data Analysis System

A comprehensive multi-agent system for data analysis powered by LangChain, LangGraph, LangSmith, and small language models. This system orchestrates specialized AI agents to perform end-to-end data science workflows.

## 🎯 Overview

This system implements a sophisticated multi-agent architecture where a planning agent coordinates specialized agents to handle different aspects of data analysis:

- **🎭 Planner Agent**: Orchestrates the overall workflow and determines which agents to invoke
- **📊 Data Reader Agent**: Analyzes datasets, provides structure information, and generates summaries
- **🔧 Data Manipulation Agent**: Handles data preprocessing, missing values, encoding, and transformations
- **⚡ Data Operations Agent**: Performs mathematical operations, filtering, aggregation, and analysis
- **🎯 ML Prediction Agent**: Trains and evaluates machine learning models with various algorithms

## 🏗️ Architecture

```
User Input → Planner Agent → Specialized Agents → Final Result
                ↓
        LangGraph Orchestration
                ↓
        LangSmith Observability
```

### Workflow Process:
1. User submits a data analysis request through Gradio interface
2. Planner Agent creates an execution plan identifying required agents
3. LangGraph orchestrates the workflow, calling agents in sequence
4. Each agent uses specialized tools to perform its tasks
5. LangSmith tracks the entire workflow for observability
6. Results are aggregated and presented to the user

## 🚀 Features

### Multi-Agent Capabilities
- **Intelligent Planning**: Automatic workflow orchestration based on user requests
- **Specialized Agents**: Domain-specific agents with targeted tool sets
- **State Management**: Persistent state across agent interactions using LangGraph
- **Error Handling**: Robust error handling and recovery mechanisms

### Data Processing
- **Multiple Formats**: Support for CSV and JSON file formats
- **Comprehensive Analysis**: Data structure analysis, statistical summaries, and pattern detection
- **Data Transformation**: Missing value imputation, categorical encoding, feature engineering
- **Mathematical Operations**: Aggregation, filtering, and computational analysis

### Machine Learning
- **Multiple Algorithms**: Random Forest, SVM, K-NN, Linear Regression
- **Model Evaluation**: Performance metrics, cross-validation, feature importance
- **Classification & Regression**: Support for both supervised learning tasks
- **Model Comparison**: Side-by-side comparison of different algorithms

### User Interface
- **Drag & Drop Upload**: Easy file upload with format validation
- **Interactive Chat**: Natural language interaction with the multi-agent system
- **Real-time Processing**: Live updates and progress tracking
- **Sample Prompts**: Pre-built examples for different complexity levels

### Observability
- **LangSmith Integration**: Complete workflow tracing and monitoring
- **Agent Performance**: Individual agent execution tracking
- **Error Logging**: Comprehensive error capture and analysis
- **Usage Analytics**: System performance and usage metrics

## 🛠️ Technology Stack

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Multi-agent orchestration and state management  
- **LangSmith**: Observability, tracing, and monitoring
- **Transformers**: Local LLM inference with HuggingFace models
- **Gradio**: Web-based user interface
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Python**: Core programming language

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster model inference)
- 8GB+ RAM recommended for local LLM inference
- LangSmith account (optional, for observability features)

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd plus_agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Download the default dataset (optional):**
```bash
python -c "from core.config import config; config.download_default_dataset()"
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# LangSmith Configuration (optional)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=multi-agent-data-analysis
LANGSMITH_TRACING=true

# Model Configuration
MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
# Alternative: microsoft/Phi-4-mini-instruct
MAX_TOKENS=1000
TEMPERATURE=0.1

# System Configuration
DEFAULT_DATASET_PATH=data/titanic.csv
SYSTEM_PROMPT_LEVEL=detailed
```

### Model Options

The system supports multiple small LLMs optimized for code and reasoning:

- **Qwen/Qwen2.5-Coder-7B-Instruct** (Default): Excellent for code generation and analysis
- **microsoft/Phi-4-mini-instruct**: Compact model with strong reasoning capabilities

## 🚦 Usage

### Starting the System

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`

### Basic Usage Examples

#### Simple Data Exploration
```
"Show me the basic information about this dataset"
"What are the column names and data types?"
"How many rows and columns are in the dataset?"
```

#### Data Operations
```
"Calculate the average age by gender"
"Show survival rates by passenger class"
"Filter passengers older than 30 and show their survival rate"
```

#### Machine Learning
```
"Train a random forest model to predict survival and show feature importance"
"Compare the performance of different ML models for survival prediction"
"Build a classification model using age, sex, pclass, and fare as features"
```

#### Comprehensive Analysis
```
"Perform a complete data science workflow on this dataset:
1. Load and examine the data structure
2. Handle missing values appropriately  
3. Create meaningful features from existing data
4. Analyze patterns and relationships
5. Train multiple machine learning models
6. Compare model performance and select the best one
7. Provide insights and recommendations"
```

### File Upload

1. **Supported Formats**: CSV, JSON (array format)
2. **File Size Limit**: 50MB maximum
3. **Auto-Detection**: File format and structure automatically detected
4. **Default Dataset**: Titanic dataset included for testing

### Agent Interaction

The system automatically determines which agents to invoke based on your request:

- **Data questions** → Data Reader Agent
- **Preprocessing tasks** → Data Manipulation Agent  
- **Mathematical operations** → Data Operations Agent
- **Model training** → ML Prediction Agent
- **Complex workflows** → Multiple agents orchestrated by the Planner

## 🧪 Testing

### Test Prompts

The system includes comprehensive test prompts in `tests/test_prompts.py`:

```python
from tests.test_prompts import TestPrompts

# Get prompts by complexity
simple_prompts = TestPrompts.SIMPLE_PROMPTS
medium_prompts = TestPrompts.MEDIUM_PROMPTS
complex_prompts = TestPrompts.COMPLEX_PROMPTS

# Get random prompt
random_prompt = TestPrompts.get_random_prompt("medium")

# Run test suite
python tests/test_prompts.py
```

### Manual Testing

1. **Load the default Titanic dataset**
2. **Try simple prompts** to verify basic functionality
3. **Test file upload** with your own CSV/JSON files
4. **Experiment with complex prompts** for full workflow testing

## 📁 Project Structure

```
plus_agent/
├── agents/                          # Specialized agent implementations
│   ├── data_reader_agent.py        # Data analysis and exploration
│   ├── data_manipulation_agent.py   # Data preprocessing and transformation
│   ├── data_operations_agent.py     # Mathematical and statistical operations
│   ├── ml_prediction_agent.py       # Machine learning model training
│   └── planner_agent.py            # Workflow planning and coordination
├── core/                           # Core system components
│   ├── config.py                   # Configuration management
│   ├── llm_wrapper.py              # LLM abstraction layer
│   ├── orchestrator.py             # LangGraph workflow orchestration
│   └── langsmith_integration.py    # Observability and tracing
├── tools/                          # Agent tool implementations
│   ├── data_tools.py               # Data reading and analysis tools
│   ├── manipulation_tools.py       # Data preprocessing tools
│   ├── operations_tools.py         # Mathematical operation tools
│   └── ml_tools.py                 # Machine learning tools
├── tests/                          # Testing and examples
│   └── test_prompts.py             # Comprehensive test prompt collection
├── data/                           # Data storage
│   └── titanic.csv                 # Default dataset
├── app.py                          # Gradio web interface
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment configuration template
└── README.md                       # This documentation
```

## 🔍 LangSmith Integration

### Setup

1. **Create LangSmith account** at [smith.langchain.com](https://smith.langchain.com)
2. **Generate API key** from your account settings
3. **Set environment variables** in your `.env` file
4. **Enable tracing** by setting `LANGSMITH_TRACING=true`

### Features

- **Workflow Tracing**: Complete execution traces for every user request
- **Agent Performance**: Individual agent execution times and success rates
- **Tool Usage**: Detailed tool invocation tracking
- **Error Analysis**: Comprehensive error logging and debugging
- **Performance Metrics**: System performance and usage analytics

### Viewing Traces

1. **Visit your LangSmith dashboard**
2. **Select the "multi-agent-data-analysis" project**
3. **View individual traces** for detailed execution information
4. **Analyze performance patterns** and identify optimization opportunities

## 🎯 Agent Details

### Data Reader Agent
**Purpose**: Analyze and understand dataset structure and content

**Capabilities**:
- CSV and JSON file parsing
- Column type detection and analysis
- Statistical summary generation
- Missing value identification
- Data quality assessment

**Tools**: `read_csv_file`, `read_json_file`, `get_column_info`, `get_data_summary`, `detect_missing_values`

### Data Manipulation Agent  
**Purpose**: Preprocess and transform data for analysis

**Capabilities**:
- Missing value imputation (mean, median, mode, forward/backward fill)
- Categorical variable encoding (one-hot, label encoding)
- Data type conversion
- Feature scaling and normalization
- Custom transformations

**Tools**: `handle_missing_values`, `create_dummy_variables`, `convert_data_types`, `scale_features`, `encode_categorical_variables`

### Data Operations Agent
**Purpose**: Perform mathematical and statistical operations

**Capabilities**:
- Mathematical computations (sum, mean, median, std)
- Data filtering and querying
- Group-by operations and aggregations
- String manipulation and processing
- Correlation analysis

**Tools**: `perform_math_operations`, `filter_data`, `group_by_operations`, `string_operations`, `calculate_correlations`

### ML Prediction Agent
**Purpose**: Train and evaluate machine learning models

**Capabilities**:
- Multiple algorithms (Random Forest, SVM, K-NN, Linear Regression)
- Model training and hyperparameter optimization
- Performance evaluation and metrics
- Feature importance analysis
- Cross-validation and model comparison

**Tools**: `train_random_forest_model`, `train_svm_model`, `train_knn_model`, `train_regression_model`, `evaluate_model`

### Planner Agent
**Purpose**: Orchestrate overall workflow and agent coordination

**Capabilities**:
- Natural language request analysis
- Workflow planning and decomposition
- Agent selection and sequencing
- Task dependency management
- Result aggregation and synthesis

**Tools**: Planning logic integrated into the agent reasoning process

## 📊 Performance Optimization

### Model Selection
- **Qwen/Qwen2.5-Coder-7B-Instruct**: Best for complex analytical tasks
- **microsoft/Phi-4-mini-instruct**: Faster inference, suitable for simpler tasks

### Memory Management
- **Gradient Checkpointing**: Enabled for large models to reduce memory usage
- **Model Caching**: Pre-loaded models for faster subsequent requests
- **Batch Processing**: Efficient handling of multiple operations

### GPU Utilization
- **CUDA Support**: Automatic GPU detection and utilization
- **Mixed Precision**: FP16 inference for faster processing
- **Memory Optimization**: Dynamic memory allocation and cleanup

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Reinstall transformers
pip install --upgrade transformers torch
```

#### Memory Issues
```python
# Reduce model parameters in config.py
MAX_TOKENS = 500  # Reduce from 1000
TEMPERATURE = 0.1
```

#### LangSmith Connection
```bash
# Verify API key
export LANGSMITH_API_KEY=your_key_here

# Test connection
python -c "from langsmith import Client; Client().list_runs(limit=1)"
```

#### File Upload Issues
- **Check file format**: Only CSV and JSON arrays supported
- **File size limit**: Maximum 50MB
- **Encoding**: Ensure UTF-8 encoding for text files

### Debug Mode

Enable verbose logging:
```python
# In config.py
DEBUG = True
VERBOSE = True
```

## 📈 Future Enhancements

### Planned Features
- **Additional File Formats**: Excel, Parquet, SQL database support
- **Advanced Visualizations**: Automated chart generation and insights
- **Model Deployment**: Export trained models for production use
- **Collaborative Features**: Multi-user support and shared workspaces
- **Advanced ML**: Deep learning models, automated feature engineering

### Model Improvements
- **Larger Models**: Support for 13B+ parameter models
- **Fine-tuning**: Domain-specific model customization
- **Multi-modal**: Support for image and text analysis
- **Streaming**: Real-time inference and processing

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent agent framework
- **LangGraph**: For multi-agent orchestration capabilities  
- **LangSmith**: For comprehensive observability features
- **HuggingFace**: For transformer models and inference
- **Gradio**: For the intuitive web interface
- **Scikit-learn**: For machine learning algorithms

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Documentation**: [Visit our docs](https://your-docs-url.com)
- **Community**: [Join our Discord](https://your-discord-url.com)

---

**Built with ❤️ using LangChain, LangGraph, and Small Language Models**