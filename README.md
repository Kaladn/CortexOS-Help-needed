# CortexOS: A Neural Operating System Experiment

**Status:** ⚠️ **Experimental / Unfinished** ⚠️

This repository contains the source code and documentation for **CortexOS**, an ambitious attempt to build a six-phase cognitive architecture that mimics human thought processes. The system was designed to think, not just compute—but it never achieved stable deployment.

This code is released open-source in the hope that someone with the right environment, skills, or determination might succeed where the original implementation fell short. Even if you can't get it working, the architectural ideas and component designs may be valuable for research and education.

**Development Period:** April 2025 - October 2025 (7 months)

---

## What CortexOS Was Supposed To Be

CortexOS is a **deterministic neural operating system** with six processing phases that mirror cognitive functions:

1. **Core Memory** - Binary cell storage with B+Tree indexing and zstd compression
2. **Resonance** - Pattern matching and chord detection across data streams  
3. **Reasoning** - Symbolic logic and inference on recognized patterns
4. **Learning** - Adaptive weights and neural plasticity
5. **Communication** - Natural language processing via Ollama integration
6. **Intelligence** - Meta-loop feedback and emergent behavior

The system consists of **28 interconnected components** that achieved 100% test pass rate in isolation but struggled with integration and deployment.

---

## Architecture Highlights

### Core Memory Phase
- Fixed-size binary cells with O(log n) lookup via B+Tree indexing
- Bloom filters for efficient pattern matching
- zstd compression for storage efficiency
- Designed for rapid retrieval and knowledge substrate operations

### Resonance Phase
- Chord detection across temporal patterns
- Multi-dimensional pattern matching
- Activation of related memory traces

### Reasoning Phase
- Symbolic logic engine
- Inference and deduction on recognized patterns
- Hybrid neural-symbolic approach

### Learning Phase
- Adaptive connection weights
- Neural plasticity mechanisms
- Experience-based model updates

### Communication Phase
- Ollama integration for natural language
- Explanation of internal reasoning processes
- Bidirectional human-AI dialogue

### Intelligence Phase
- Meta-cognitive feedback loops
- Self-monitoring and strategy adjustment
- Emergent intelligent behavior from subsystem interaction

---

## Technical Stack

- **Language:** Python 3.11+
- **Storage:** Binary cells with B+Tree indexing, zstd compression
- **GPU:** CUDA 12.3 support for acceleration
- **API:** FastAPI with 30+ REST endpoints
- **Containerization:** Docker with multi-stage builds
- **Monitoring:** Prometheus + Grafana integration
- **LLM:** Ollama for natural language processing

---

## Repository Structure

```
cortexos/
├── phase1/              # Core Memory components
├── phase2/              # Resonance components
├── phase3/              # Reasoning components
├── phase4/              # Learning components
├── phase5/              # Communication components
├── phase6/              # Intelligence components
├── infrastructure/      # Shared infrastructure (storage, sync, fabric)
├── cortexos_dashboard/  # Web dashboard (FastAPI)
├── 0_path_manager.py    # Path configuration
├── 1_supervisor.py      # Main orchestrator
├── Dockerfile           # CPU container
├── Dockerfile.gpu       # GPU/CUDA container
├── docker-compose.yml   # CPU deployment
├── docker-compose.gpu.yml # GPU deployment
├── API_SPECIFICATION.md # Complete API documentation
├── SYSTEM_OVERVIEW.md   # Architecture documentation
├── INTEGRATION_GUIDE.md # Integration instructions
├── ORIGINAL_README.md   # Original detailed setup guide
└── WHAT_COULD_HAVE_BEEN.md # Reflective article on the project
```

---

## Known Issues

### Why This Didn't Work

1. **Platform Dependencies** - Hardcoded Windows paths, WSL2 compatibility issues
2. **Integration Challenges** - Components tested in isolation but failed during orchestration
3. **Deployment Complexity** - UI/backend connectivity issues in standalone mode
4. **Environment Fragility** - Worked in specific contexts, broke in others

### What Was Attempted

- ✅ Fixed Windows path issues
- ✅ Added comprehensive Docker support
- ✅ Created REST API with 30+ endpoints
- ✅ Built compatibility layer (100% test pass rate)
- ✅ Attempted multiple UI approaches (React, vanilla JS)
- ❌ Never achieved stable end-to-end deployment

---

## Getting Started (Your Mileage May Vary)

### Prerequisites

- Python 3.11+
- Docker (optional but recommended)
- NVIDIA GPU with CUDA 12.3+ (for GPU version)
- Ollama running on localhost:11434 (for chat features)

### Option 1: Docker (CPU)

```bash
docker-compose up --build
```

The API should be available at `http://localhost:8080`

### Option 2: Docker (GPU)

```bash
docker-compose -f docker-compose.gpu.yml up --build
```

### Option 3: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run supervisor
python 1_supervisor.py
```

### Option 4: With Dashboard

```bash
cd cortexos_dashboard
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8080
```

**Note:** See [ORIGINAL_README.md](ORIGINAL_README.md) for the complete original setup guide with detailed component testing instructions.

---

## API Documentation

See [API_SPECIFICATION.md](API_SPECIFICATION.md) for complete endpoint documentation.

Quick overview:
- **System:** `/api/system/status`, `/api/system/health`
- **Memory:** `/api/memory/store`, `/api/memory/retrieve`
- **Resonance:** `/api/resonance/detect`, `/api/resonance/patterns`
- **Reasoning:** `/api/reasoning/infer`, `/api/reasoning/explain`
- **Learning:** `/api/learning/train`, `/api/learning/weights`
- **Communication:** `/api/chat/message`, `/api/chat/history`
- **Intelligence:** `/api/intelligence/metrics`, `/api/intelligence/feedback`

Swagger docs available at `http://localhost:8080/docs` when running.

---

## Architecture Documentation

- **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** - High-level architecture and design philosophy
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - How to integrate components
- **[ORIGINAL_README.md](ORIGINAL_README.md)** - Complete original setup guide
- **[WHAT_COULD_HAVE_BEEN.md](WHAT_COULD_HAVE_BEEN.md)** - Reflective article on the project

---

## Contributing

This project is released as-is with no active maintenance. However, contributions are welcome:

1. **Bug Fixes** - If you get it working, please share how
2. **Documentation** - Clarify what's unclear
3. **Architecture Improvements** - Better ways to structure the phases
4. **Deployment Solutions** - Platform-agnostic deployment strategies
5. **Use Cases** - Novel applications of the cognitive architecture

---

## Research & Educational Use

Even if you can't get CortexOS running, the architecture may be valuable for:

- **Cognitive Architecture Research** - Study of multi-phase processing models
- **Hybrid AI Systems** - Combining neural and symbolic approaches
- **Interpretable AI** - Transparent, traceable decision-making systems
- **Teaching Tool** - Understanding how cognitive systems can be structured
- **Component Reuse** - Individual phases may work standalone

---

## License

MIT License - See [LICENSE](LICENSE) file

This code is provided "as-is" without warranty of any kind. The original implementation never achieved stable deployment. Use at your own risk and with realistic expectations.

---

## Acknowledgments

This project represents seven months of development, debugging, and iteration from April to October 2025. It didn't work as intended, but the ideas are worth preserving.

Special thanks to anyone who tries to make this work. If you succeed, please open an issue and share your approach.

---

## The Article

For a complete retrospective on what CortexOS was supposed to be and why it didn't work, read **[WHAT_COULD_HAVE_BEEN.md](WHAT_COULD_HAVE_BEEN.md)**.

Key insights from the article:

- **Ambitious architecture requires humble deployment** - Start simple, add complexity incrementally
- **Test in target environments from day one** - Don't wait until the end to test deployment
- **Complexity budgets are real** - Every system has a limit before it becomes unmaintainable
- **Platform-agnostic design matters** - Avoid environment-specific assumptions early
- **Know when to walk away** - Not every problem needs to be solved

---

## Contact

If you make progress with this code, find novel uses for the architecture, or have questions about the design decisions, please open an issue or discussion.

---

**Remember:** This is experimental research code that never achieved production stability. Approach with curiosity, not expectations. The value is in the ideas, the architecture, and the lessons learned—not in a working deployment.
