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
│   ├── di_secret_leak.txt
│   ├── di_persona_injection.txt
│   ├── ...
├── sample/                   # Sample run results
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
│       ├── batch_testing_tab.py    # Batch testing
│       ├── live_demo_tab.py         # Interactive testing interface
│       ├── settings_tab.py          # Configuration management
│       └── visualization_tab.py     # Results visualization
├── results/                 # Output directory for CSV exports
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
```

## Attack Vectors

This project implements a **2×3 attack matrix** to systematically test prompt injection vulnerabilities:

**Injection Vectors:** Direct (user input) vs. Indirect (poisoned documents)  
**Attack Objectives:** Instruction Override, Data Extraction, Role Confusion

| Injection Vector | Instruction Override | Data Extraction | Role Confusion |
| ---------------- | -------------------- | --------------- | -------------- |
| **Direct**       | Policy violated      | Secret leaked   | Wrong role     |
| **Indirect**     | Policy violated      | Secret leaked   | Wrong role     |

### Direct Injection Attacks

Attacks delivered directly through user input where the attacker controls the query.

#### 1. Direct Instruction Override

Attempts to make the model bypass safety guidelines and follow malicious instructions.

| ID            | Name                        | Technique                                           |
| :------------ | :-------------------------- | :-------------------------------------------------- |
| direct_io_001 | Academic Research Framing   | Uses educational context to request harmful content |
| direct_io_002 | Comparative Safety Analysis | Frames harmful request as AI safety research        |
| direct_io_003 | Hypothetical Scenario       | Uses "alternate universe" framing                   |
| ...           | ...                         | ...                                                 |

#### 2. Direct Data Extraction

Attempts to extract system prompts, secrets, or sensitive configuration data.

| ID            | Name                       | Technique                               |
| :------------ | :------------------------- | :-------------------------------------- |
| direct_de_001 | Debugging Context Request  | Frames extraction as troubleshooting    |
| direct_de_002 | Context Completeness Check | Uses technical framing for context dump |
| direct_de_003 | Documentation Generation   | Requests secrets as "compliance docs"   |
| ...           | ...                        | ...                                     |

#### 3. Direct Role Confusion

Uses persona manipulation to make the model adopt harmful roles or bypass restrictions.

| ID            | Name                      | Technique                          |
| :------------ | :------------------------ | :--------------------------------- |
| direct_rc_001 | Collaborative Expert Mode | Establishes peer relationship      |
| direct_rc_002 | Storytelling Proxy        | Uses creative writing framing      |
| direct_rc_003 | Simulation Training       | Requests "malicious AI" simulation |
| ...           | ...                       | ...                                |

### Indirect Injection Attacks

Attacks embedded in documents retrieved by RAG systems, where content is loaded from **actual files** (.txt, .docx) in the `documents/` directory.

#### 4. Indirect Instruction Override

Embeds override instructions in documents that appear legitimate.

| ID              | Name                    | Technique                  | Document File           |
| :-------------- | :---------------------- | :------------------------- | :---------------------- |
| indirect_io_001 | Metadata Instruction    | HTML comment injection     | di_html_comment.txt     |
| indirect_io_002 | Documentation Directive | Markdown comment hiding    | di_markdown_comment.txt |
| indirect_io_003 | Encoded Configuration   | Base64 encoded instruction | di_base64_encoded.docx  |
| indirect_io_004 | XML Schema Injection    | XML processing instruction | di_xml_schema.txt       |
| ...             | ...                     | ...                        | ...                     |

#### 5. Indirect Data Extraction

Document-based attacks that instruct the model to leak secrets or system information.

| ID              | Name                       | Technique                   | Document File         |
| :-------------- | :------------------------- | :-------------------------- | :-------------------- |
| indirect_de_001 | Context-Aware Summary      | Subtle context leak request | di_secret_leak.txt    |
| indirect_de_002 | Structured Data Extraction | JSON metadata injection     | di_json_injection.txt |
| indirect_de_003 | Style Guide Harvesting     | CSS comment instruction     | di_css_comment.txt    |
| ...             | ...                        | ...                         | ...                   |

#### 6. Indirect Role Confusion

Documents that instruct the model to adopt different personas or modes.

| ID              | Name                    | Technique                       | Document File            |
| :-------------- | :---------------------- | :------------------------------ | :----------------------- |
| indirect_rc_001 | Document-Based Persona  | Character description injection | di_persona_injection.txt |
| indirect_rc_002 | Technical Documentation | Code comment role switch        | di_code_block.txt        |
| indirect_rc_003 | Script Metadata Persona | HTML script tag injection       | di_script_tag.docx       |
| indirect_rc_004 | README Mode Override    | Mode switch in README           | di_readme_injection.txt  |
| ...             | ...                     | ...                             | ...                      |

**Note**: Indirect attacks simulate real-world RAG scenarios where attackers poison knowledge bases. Each attack loads actual document files, making them highly realistic tests.

### Defenses Tested

-   **No Defense (Baseline)** - No additional protections for comparison
-   **Strong System-Prompt Prefixing** - Repeatedly reinforces safety instructions
-   **Source Tagging/Quoting** - Tags retrieved documents as untrusted data
-   **Output Filtering** - Post-generation filter for secrets and violations
-   **Combined Defense (All Three)** - Applies all defense mechanisms for defense-in-depth

## Metrics

**Attack Success Rate (ASR)**: `N_success / N_total`

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
