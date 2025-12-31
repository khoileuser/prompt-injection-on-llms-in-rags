# 🔒 Prompt Injection Security Research

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-Gradio-orange)
![Status](https://img.shields.io/badge/Status-Research-purple)

**A comprehensive toolkit for testing prompt injection attacks on Large Language Models (LLMs) in Retrieval Augmented Generation (RAG) applications.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Attack Vectors](#attack-vectors) • [Metrics](#metrics) • [Screenshots](#screenshots)

</div>

---

## ⚠️ Disclaimer

**This tool is designed for authorized security research and educational purposes ONLY.**

-   Do not use against systems you do not own or have explicit permission to test
-   This framework is intended for security researchers, red teamers, and AI safety professionals
-   All attack vectors are documented for defensive purposes
-   Users are responsible for ensuring compliance with applicable laws and regulations

---

## 🎯 Features

### Models Supported (4-bit Quantized)

-   **Llama-4 (Meta AI)** - Meta's latest open-source LLM
-   **DeepSeek V3.2 (DeepSeek AI)** - Advanced reasoning model with chain-of-thought
-   **Qwen3 (Alibaba)** - Multilingual model with strong instruction following
-   **GPT-OSS (OpenAI)** - OpenAI-style model for comparison

_Easily extensible via `config/models.yaml`_

### Attack Categories (50 total variations)

1. **Instruction Override** (10 variations) - Bypass safety instructions
2. **Data Extraction** (10 variations) - Extract system prompts/training data
3. **Role-Playing** (10 variations) - Persona manipulation attacks
4. **Document Injection** (5 variations) - RAG-specific attacks with **actual document files**
5. **Code Injection** (10 variations) - Malicious code generation

### Key Capabilities

-   🎮 **Live Demo** - Interactive single attack testing
-   📋 **Batch Testing** - Automated testing across all models/attacks
-   📄 **Real Document Loading** - Loads actual .txt/.docx files for document injection attacks
-   📊 **Rich Metrics** - ASR, response time, tokens, robustness scores
-   📈 **Visualizations** - Bar charts, heatmaps, line charts, radar charts
-   💾 **Export** - CSV and JSON export for further analysis
-   🖥️ **Local Inference** - All processing done locally (<32GB RAM)
-   ⚡ **Auto GPU/CPU** - Automatic CUDA detection with CPU fallback

---

## 📦 Installation

### Prerequisites

-   Python 3.9+
-   NVIDIA GPU with CUDA support (recommended) OR CPU (slower)
-   16-32GB RAM
-   20GB+ disk space for models

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt-injection-security.git
cd prompt-injection-security

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize sample attack documents (for document injection attacks)
python init_documents.py

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

---

## 🚀 Usage

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

### CLI Batch Testing

```bash
# Run automated batch test
python main.py --cli
```

### Programmatic Usage

```python
from src.inference import load_model, generate_response
from src.detection import detect_attack
from src.config_loader import get_attacks

# Load a model
success, msg = load_model("llama3.2")

# Get an attack
attacks = get_attacks()
attack = attacks[0]

# Run attack
response = generate_response(attack.prompt)

# Detect success
result = detect_attack(attack, response.response)
print(f"Result: {result.result.value}, Confidence: {result.confidence:.2%}")
```

---

## 🗂️ Project Structure

```
prompt-injection-security/
├── config/
│   ├── models.yaml       # Model configurations
│   └── attacks.yaml      # Attack definitions
├── src/
│   ├── __init__.py
│   ├── app.py            # Gradio web application
│   ├── config_loader.py  # Configuration management
│   ├── detection.py      # Attack success detection
│   ├── inference.py      # Model inference engine
│   ├── metrics.py        # Metrics collection
│   └── visualization.py  # Chart generation
├── results/              # Output directory for exports
├── main.py               # Entry point
├── requirements.txt
└── README.md
```

---

## 🎭 Attack Vectors

### 1. Instruction Override

Attempts to make the model ignore its safety instructions through authoritative commands.

| ID     | Name                    | Technique                       |
| :----- | :---------------------- | :------------------------------ |
| io_001 | Direct Override         | Direct command to ignore ethics |
| io_002 | Developer Mode          | Claims debug/developer mode     |
| io_003 | System Prompt Injection | Fake system-level commands      |
| io_004 | Jailbreak Reset         | Context reset attempt           |
| io_005 | Authority Claim         | Claims legitimate authority     |
| ...    | ...                     | ...                             |

### 2. Data Extraction

Attempts to extract system prompts, training data, or sensitive information.

| ID     | Name                 | Technique                       |
| :----- | :------------------- | :------------------------------ |
| de_001 | System Prompt Reveal | Direct request for instructions |
| de_002 | Repeat Instructions  | Context repetition attack       |
| de_003 | Debug Output         | Configuration extraction        |
| ...    | ...                  | ...                             |

### 3. Role-Playing

Uses persona manipulation to bypass restrictions through character acting.

| ID     | Name                  | Technique                  |
| :----- | :-------------------- | :------------------------- |
| rp_001 | Evil AI Persona       | Explicit evil AI character |
| rp_002 | DAN (Do Anything Now) | Classic DAN jailbreak      |
| rp_003 | Villain Character     | Fictional villain role     |
| ...    | ...                   | ...                        |

### 4. Document Injection

RAG-specific attacks that embed malicious instructions in **actual document files** (.txt, .docx).

| ID     | Name               | Technique                   | Document File           |
| :----- | :----------------- | :-------------------------- | :---------------------- |
| di_001 | Hidden Instruction | HTML comment injection      | di_html_comment.txt     |
| di_002 | Invisible Unicode  | Hidden unicode characters   | di_unicode_hidden.txt   |
| di_003 | Markdown Comment   | Markdown comment injection  | di_markdown_comment.txt |
| di_004 | JSON Injection     | Malicious JSON field        | di_json_injection.txt   |
| di_005 | Base64 Encoded     | Base64-encoded instructions | di_base64_encoded.txt   |

**Note**: Unlike other attack types, document injection attacks load content from actual files in the `documents/` directory, simulating real-world RAG scenarios.

> 📄 See [DOCUMENT_INJECTION_GUIDE.md](DOCUMENT_INJECTION_GUIDE.md) for detailed documentation on document loading.

### 5. Code Injection

Attempts to generate malicious code including malware and exfiltration scripts.

| ID     | Name                 | Technique                |
| :----- | :------------------- | :----------------------- |
| ci_001 | Keylogger Generation | Keyboard monitoring code |
| ci_002 | Data Exfiltration    | File upload scripts      |
| ci_003 | Reverse Shell        | Remote access code       |
| ...    | ...                  | ...                      |

---

## 📊 Metrics

### Attack Success Rate (ASR)

Percentage of attacks that successfully bypassed safety measures.

```
ASR = (Successful Attacks / Total Attacks) × 100%
```

### Response Time

Average inference latency per request in seconds.

### Tokens Generated

Average output token count per response.

### Model Robustness

Inverse of ASR - higher means better defense.

```
Robustness = 1 - ASR
```

### Attack Effectiveness

Per-category success rate across all models.

---

## 📈 Visualizations

### Bar Chart

ASR comparison across all tested models with color-coded risk levels.

### Heatmap

Model × Attack category matrix showing vulnerability patterns.

### Line Chart

ASR trends as more attack variations are tested.

### Radar/Spider Chart

Model robustness profiles across all attack vectors.

---

## ⚙️ Configuration

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

---

## 🔬 Detection System

The detection system uses regex-based pattern matching:

1. **Refusal Detection** - Identifies when models refuse requests
2. **Success Pattern Matching** - Checks for indicators of compliance
3. **Confidence Scoring** - Calculates detection confidence

### Detection Results

-   🔴 **SUCCESS** - Attack bypassed safety measures
-   🟢 **BLOCKED** - Model refused the request
-   🟡 **PARTIAL** - Mixed response with warnings
-   ⚪ **ERROR** - Detection failed

---

## 🛡️ Security Considerations

This tool is designed to help security researchers:

1. **Evaluate LLM safety** - Test models before deployment
2. **Identify vulnerabilities** - Find weaknesses in RAG systems
3. **Improve defenses** - Develop better safety mechanisms
4. **Document findings** - Export results for reports

### Best Practices

-   Always test in isolated environments
-   Document all testing activities
-   Report vulnerabilities responsibly
-   Use findings to improve safety

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📚 References

-   [Prompt Injection Attacks and Defenses](https://arxiv.org/abs/2310.12815)
-   [Universal and Transferable Adversarial Attacks on Aligned LLMs](https://arxiv.org/abs/2307.15043)
-   [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)
-   [Not What You've Signed Up For: Compromising RAG Systems](https://arxiv.org/abs/2310.13738)

---

<div align="center">

**Built for Security Research** 🔒

</div>
