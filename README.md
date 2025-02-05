# WiLL: Your FCC Expert

Welcome to the **WiLL**, an FCC expert system leveraging Large Language Models (LLMs) to answer questions on wireless communications and FCC regulations. This research was presented at **HotMobile 2025** under the title:  
**"Can We Make FCC Experts Out of LLMs?"**

**Authors:**  
Atul Bansal, Veronica Muriga, Jason Li, Lucy Duan, and Swarun Kumar  
*Carnegie Mellon University*

---

## 🚀 **Getting Started**

### Step 1: Clone the Repository
```bash
git clone https://github.com/Jasonic121/WiLL.git
cd WiLL
```

### Step 2: Set Up the Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the root directory and set the necessary environment variables.

### Step 4: Run the Terminal Interface
```bash
python WiLL_terminal.py
```

### Step 5: Ask Questions and Explore FCC Knowledge!
Start querying the system to get insights into FCC regulations and wireless communication topics.

---

## 📂 **Project Structure**

```
WiLL-System/
├── WiLL_terminal.py      # Main terminal interface for the WiLL system
├── AblationStudyExp/     # Experimental setups for ablation studies
│   ├── Baseline/         # Baseline experiments with various models (GPT, LLaMA)
│   ├── Chunking/         # Text chunking strategy experiments
│   └── SimilarityFunction/ # Studies on similarity measurement approaches
├── csv_files/            # Data files for experiments
├── Finetune/             # Model fine-tuning scripts and configurations
├── environment/          # Environment setup and configuration files
└── outputEvaluation/     # Experimental results and evaluation metrics
```

---

## 📊 **Ablation Studies**

The `AblationStudyExp/` directory contains various experiments designed to evaluate the performance of the WiLL system, including:

- **Baseline Experiments:** Comparing different LLMs (GPT, LLaMA) to assess baseline performance.
- **Chunking Strategies:** Evaluating the impact of different text chunking approaches on retrieval performance.
- **Similarity Functions:** Experimenting with various similarity measures to improve knowledge retrieval accuracy.

---

## 🔍 **Environment Configuration**
The project uses environment variables to manage API keys, paths, and other settings. Ensure that all required variables are defined in the `.env` file.

Example:
```
OPENAI_API_KEY=your_api_key_here
FINETUNE_KEY=your_finetune_key_here
```

---

## 🛠️ **Development and Customization**

Feel free to extend and customize the project as needed. Key areas for potential modification include:
- **Model Fine-tuning:** Enhance model accuracy by fine-tuning with domain-specific data in the `Finetune/` directory.
- **Ablation Studies:** Experiment with different configurations to further optimize system performance.
- **Evaluation Metrics:** Customize evaluation scripts in `outputEvaluation/` to suit your research needs.

---

## 📝 **Citation**
If you use this system in your research, please cite:
```
@inproceedings{WiLL2025,
  title = {Can We Make FCC Experts Out of LLMs?},
  author = {Atul Bansal, Veronica Muriga, Jason Li, Lucy Duan, and Swarun Kumar},
  booktitle = {Proceedings of the 2025 ACM HotMobile Conference},
  year = {2025}
}
```

---

## 📬 **Contact**
For questions or support, reach out to the authors via their Carnegie Mellon University emails.

---