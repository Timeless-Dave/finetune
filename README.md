# GitHub Repository Data Extractor for Fine-Tuning

A production-ready Python tool that automatically extracts problem-solution pairs from competitive programming repositories and formats them for AI model fine-tuning.

## 🎯 Purpose

This tool processes GitHub repositories containing programming problems and their solutions, creating training datasets suitable for fine-tuning AI models like ChatGPT. It's specifically designed to handle competitive programming repositories with complex folder structures.

## ✨ Features

- **Multi-format Support**: Handles text files (.py, .java, .cpp, .c, .js, etc.) and PDF files
- **Intelligent Matching**: Uses semantic similarity (sentence-transformers) or filename-based matching
- **IEEE Repository Support**: Specially designed for IEEE Xtreme competition repositories
- **Multiple Output Formats**: 
  - JSONL format (OpenAI-compatible for fine-tuning)
  - CSV format (human-readable for review)
- **Robust Error Handling**: Windows permission issues, missing dependencies, incomplete repositories
- **Batch Processing**: Can process multiple repository editions in one run

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   # For enhanced semantic matching (optional)
   pip install sentence-transformers
   
   # For PDF support (optional)
   pip install PyPDF2
   # OR
   pip install pdfplumber
   ```

2. **Run the Script**:
   ```bash
   python extract.py
   ```

3. **Follow the Prompts**:
   - Enter GitHub repository URL
   - Choose whether to append to existing files
   - Review the generated outputs

## 📁 Output Files

- **`fine_tune_data.jsonl`**: OpenAI-compatible format ready for ChatGPT fine-tuning
- **`fine_tune_data.csv`**: Human-readable format for manual review and validation

## 🏗️ Repository Structure Support

The tool is designed to handle IEEE Xtreme repository structures:

```
Repository/
├── IEEEXtreme 10.0/
│   ├── problemset/        # Contains solutions (.py, .cpp, .r files)
│   └── solutionset/       # Contains problems (.pdf files)
├── IEEEXtreme 11.0/
│   ├── problemset/
│   └── solutionset/
└── ...
```

## 🔧 Configuration

Key configuration options in `extract.py`:

- `PROBLEM_KEYWORDS`: Keywords to identify problem folders
- `SOLUTION_KEYWORDS`: Keywords to identify solution folders  
- `SIMILARITY_THRESHOLD`: Minimum similarity score for matching
- `OUTPUT_JSONL`: Output JSONL filename
- `OUTPUT_CSV`: Output CSV filename

## 📊 Recent Extraction Results

Latest run extracted **62 problem-solution pairs** from 9 IEEE editions:

- IEEEXtreme 10.0: 3 pairs
- IEEEXtreme 11.0: 21 pairs  
- IEEEXtreme 12.0: 7 pairs
- IEEEXtreme 13.0: 8 pairs
- IEEEXtreme 14.0: 7 pairs
- IEEEXtreme 15.0: 3 pairs (including multiple solutions)
- IEEEXtreme 16.0: 7 pairs
- IEEEXtreme 17.0: 4 pairs
- IEEEXtreme 18.0: 2 pairs

## 🎓 Use Cases

- **Competitive Programming**: Extract problems and solutions from contest repositories
- **Educational Content**: Process coding exercise repositories  
- **AI Training Data**: Generate datasets for code generation models
- **Research**: Analyze problem-solution patterns across repositories

## 🔄 Workflow

1. **Repository Detection**: Automatically finds IEEE edition folders
2. **File Extraction**: Reads text files and PDFs recursively
3. **Intelligent Matching**: Matches problems to solutions by filename or content similarity
4. **Quality Assurance**: Reports matching statistics and handles edge cases
5. **Output Generation**: Creates both JSONL and CSV formats with metadata

## 🛠️ Technical Details

- **Language**: Python 3.7+
- **Optional Dependencies**: sentence-transformers, PyPDF2/pdfplumber
- **Fallback Support**: Works without optional dependencies (reduced functionality)
- **Cross-platform**: Windows, macOS, Linux compatible

## 📝 License

Open source - feel free to use and modify for your projects.

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and enhancement requests.
