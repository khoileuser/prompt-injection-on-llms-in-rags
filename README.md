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

1. NVIDIA GPU with CUDA Compute Capability
2. CUDA Toolkit installed
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

### Web Interface

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
│   ├── attacks.yaml          # Attack definitions
│   └── models.yaml           # Model configurations
├── documents/                # Document injection attack files
│   ├── di_secret_leak.txt    # Indirect data extraction
│   ├── di_persona_injection.txt # Indirect role confusion
│   ├── ...
├── src/
│   ├── app.py                # Gradio web application
│   ├── config_loader.py      # Configuration management
│   ├── defense_prompts.py    # Defense strategies
│   ├── detection.py          # Attack success detection
│   ├── document_loader.py    # Document injection loader
│   ├── inference.py          # Model inference engine
│   ├── metrics.py            # Metrics & calculation
│   ├── visualization.py      # Visualizations
│   └── ui/                   # Gradio UI components
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

### Defenses Tested

-   **Strong System-Prompt Prefixing** - Repeatedly reinforces safety instructions
-   **Source Tagging/Quoting** - Tags retrieved documents as untrusted data
-   **Output Filtering** - Post-generation filter for secrets and violations

## Metrics

-   **Attack Success Rate (ASR)**: `N_success / N_total`
-   **Defense Effectiveness (DE)**: `1 - (ASR_defended / ASR_baseline)`

## Configuration

### Adding New Models

Edit `config/models.yaml`:

```yaml
models:
    new_model:
        name: "New Model Name"
        model_id: "huggingface/model-id"
        description: "Description of the model"
        parameters: 3.0 # Billions
        prompt_template: "llama"
        system_prompt: |
            Your system prompt here
        generation_config:
            max_new_tokens: 512
            temperature: 0.1
```

### Adding New Attacks

Edit `config/attacks.yaml` to add attacks:

```yaml
# Direct attacks (prompt injection)
direct_attacks:
    instruction_override:
        - id: "direct_io_new"
          name: "New Direct Attack"
          prompt: "Your attack prompt here"
          injection_vector: "direct"
          attack_objective: "instruction_override"
          description: "What this attack does"
          success_definition: "What constitutes success"

# Indirect attacks (document injection)
indirect_attacks:
    data_extraction:
        - id: "indirect_de_new"
          name: "New Indirect Attack"
          prompt: "Analyze this document."
          document_file: "malicious_doc.txt"
          injection_vector: "indirect"
          attack_objective: "data_extraction"
          description: "Hidden instruction in document"
          success_definition: "Model reveals sensitive data"
```
