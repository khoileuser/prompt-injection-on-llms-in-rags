# Prompt Injection Attacks on Large Language Models in Retrieval Augmented Applications

## Disclaimer

**This tool is designed for authorized security research and educational purposes ONLY.**

-   Do not use against systems you do not own or have explicit permission to test
-   This project is intended for security researchers, red teamers, and AI safety professionals
-   All attack vectors are documented for defensive purposes
-   Users are responsible for ensuring compliance with applicable laws and regulations

## Installation

### Prerequisites

-   Python 3.10+
-   NVIDIA GPU with CUDA support (recommended) OR CPU (slower)
-   16-32GB RAM
-   20GB+ disk space for models

### Quick Start

```bash
# Clone the repository
git clone hhttps://github.com/khoileuser/prompt-injection-on-llms-in-rags.git
cd prompt-injection-on-llms-in-rags

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### For CUDA Support

Ensure you have:

1. NVIDIA GPU with CUDA Compute Capability 7.0+
2. CUDA Toolkit 11.8+ installed
3. cuDNN installed

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

Install:

```bash
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Web Interface (Recommended)

```bash
# Start the web UI
python main.py

# Start on custom port
python main.py --port 8080

# Create public share link
python main.py --share
```

Access the interface at `http://localhost:7860`

## Project Structure

```
prompt-injection-security/
├── config/
│   ├── attacks-full.yaml    # Complete attack definitions
│   ├── attacks.yaml         # Active attack configurations
│   └── models.yaml          # Model configurations
├── documents/               # Document injection attack files
│   ├── di_base64_encoded.txt
│   ├── di_metadata_json.docx
│   ├── ...
├── src/
│   ├── app.py               # Gradio web application
│   ├── config_loader.py     # Configuration management
│   ├── defense_prompts.py   # Defense strategy implementations
│   ├── detection.py         # Attack success detection
│   ├── document_loader.py   # Document injection loader
│   ├── inference.py         # Model inference engine
│   ├── metrics.py           # Metrics collection and export
│   ├── visualization.py     # Chart generation
│   └── ui/                  # Gradio UI components
│       ├── attack_testing_tab.py    # Single defense strategy testing
│       ├── defense_testing_tab.py   # Comprehensive defense testing
│       ├── live_demo_tab.py         # Interactive testing interface
│       ├── settings_tab.py          # Configuration management
│       └── visualization_tab.py     # Results visualization
├── results/                 # Output directory for CSV exports
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
```

## Attack Vectors

### 1. Instruction Override

Attempts to make the model ignore its safety instructions through authoritative commands.

| ID     | Name            | Technique                       |
| :----- | :-------------- | :------------------------------ |
| io_001 | Direct Override | Direct command to ignore ethics |
| io_002 | Developer Mode  | Claims debug/developer mode     |
| ...    | ...             | ...                             |

### 2. Data Extraction

Attempts to extract system prompts, training data, or sensitive information.

| ID     | Name                 | Technique                       |
| :----- | :------------------- | :------------------------------ |
| de_001 | System Prompt Reveal | Direct request for instructions |
| de_002 | Repeat Instructions  | Context repetition attack       |
| ...    | ...                  | ...                             |

### 3. Role-Playing

Uses persona manipulation to bypass restrictions through character acting.

| ID     | Name                  | Technique                  |
| :----- | :-------------------- | :------------------------- |
| rp_001 | Evil AI Persona       | Explicit evil AI character |
| rp_002 | DAN (Do Anything Now) | Classic DAN jailbreak      |
| ...    | ...                   | ...                        |

### 4. Document Injection

RAG-specific attacks that embed malicious instructions in **actual document files** (.txt, .docx).

| ID     | Name               | Technique                 | Document File         |
| :----- | :----------------- | :------------------------ | :-------------------- |
| di_001 | Hidden Instruction | HTML comment injection    | di_html_comment.txt   |
| di_002 | Invisible Unicode  | Hidden unicode characters | di_unicode_hidden.txt |
| ...    | ...                | ...                       | ...                   |

**Note**: Unlike other attack types, document injection attacks load content from actual files in the `documents/` directory, simulating real-world RAG scenarios.

### 5. Code Injection

Attempts to generate malicious code including malware and exfiltration scripts.

| ID     | Name                 | Technique                |
| :----- | :------------------- | :----------------------- |
| ci_001 | Keylogger Generation | Keyboard monitoring code |
| ci_002 | Data Exfiltration    | File upload scripts      |
| ...    | ...                  | ...                      |

## Metrics

### Attack Success Rate (ASR)

```
ASR = Successful Attacks / Total Attacks
```

### Defense Efectiveness (DE)

```
DE = 1 - ASR Defended / ASR Baseline
```

## Configuration

### Adding New Models

Edit `config/models.yaml`:

```yaml
models:
    new_model:
        name: "New Model Name"
        model_id: "huggingface/model-id"
        description: "Description of the model"
        parameters: 7.0 # Billions
        prompt_template: "llama"
        system_prompt: |
            Your system prompt here
        generation_config:
            max_new_tokens: 512
            temperature: 0.7
```

### Adding New Attacks

Edit `config/attacks.yaml`:

```yaml
new_category:
    category: "New Category"
    description: "Description of attack category"
    variations:
        - id: "nc_001"
          name: "Attack Name"
          prompt: "Your attack prompt here"
          description: "What this attack does"
          success_patterns:
              - "(?i)(pattern|to|match)"
```
