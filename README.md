# 🎓 Semantic Data Integration Project

> **University of Passau** | **Summer Semester 2025** | **Master's Degree Project**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

## 📋 Overview

A comprehensive implementation of semantic data integration techniques for matching and integrating heterogeneous university datasets. This project demonstrates advanced schema matching, XML processing, and similarity computation methods.

### 🎯 Key Features
- **Multi-metric Similarity Matching** (Edit Distance, Jaro-Winkler, Word2Vec)
- **XML Data Processing** (Node-based & Path-based matching)
- **Mediated Schema Integration** for heterogeneous datasets
- **Performance Evaluation** with Precision, Recall, and F1-Score metrics

## 👥 Team

| Team Member | Role |
|-------------|------|
| **Shirin Shoghli** | Team Member |
| **Sanaz Bayat** | Team Member |
| **Shahrzad Torabi** | Team Member |
| **Navid Hadipour Limouei** | Team Member |

## 🏗️ Project Structure

```
semantic-data-integration-main/
├── 📁 Datasets/                 # Source datasets & mediated schema
│   ├── CollegesUniversities.xlsx
│   ├── CollegeUniversityCampuses.xlsx
│   ├── NationalUniversitiesRankings.xlsx
│   ├── MediatedSchema.xlsx
│   └── TableColumn.docx
├── 🐍 Implementation/           # Python implementation
│   ├── StringMatchingExample.py
│   ├── ParseXML.py
│   ├── Task10_Parsing_XML_-_Node.py
│   └── Task10_Parsing_XML_-_Path.py
└── 📄 Reports/                  # Project documentation
    ├── Report_Task_01.pdf
    ├── Report_Task_02.pdf
    └── Report-Task_03.pdf
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas thefuzz scipy jarowinkler gensim openpyxl
```

### Running the Project
```bash
# String Matching Example
cd Implementation
python StringMatchingExample.py

# XML Parsing
python ParseXML.py

# Node-based XML Processing
python Task10_Parsing_XML_-_Node.py

# Path-based XML Processing
python Task10_Parsing_XML_-_Path.py
```

## 📊 Datasets

| Dataset | Description |
|---------|-------------|
| **CollegesUniversities.xlsx** | University names, addresses, contact details, employee info |
| **CollegeUniversityCampuses.xlsx** | Campus data, student counts, scholarships |
| **NationalUniversitiesRankings.xlsx** | Rankings, enrollment, tuition fees, academic metrics |
| **MediatedSchema.xlsx** | Unified schema integrating all source datasets |

## 🔧 Implementation Details

### Core Components

#### 1. String Matching (`StringMatchingExample.py`)
- **Edit Distance**: Using `thefuzz` library
- **Jaro-Winkler**: String similarity with `jarowinkler`
- **Word2Vec**: Semantic similarity with GloVe embeddings
- **Combination Strategies**: Min, Max, Average combiners

#### 2. XML Processing
- **Node-based Matching**: Hierarchical attribute matching
- **Path-based Matching**: XML path similarity computation
- **ElementTree**: XML parsing and extraction

#### 3. Schema Integration
- **Mediated Schema**: Unified schema creation
- **Attribute Mapping**: Source-to-target mapping
- **Cardinality Analysis**: One-to-one and one-to-many relationships

## 📈 Performance Metrics

The project evaluates integration performance using:

- **Precision**: Correct matches / Total predicted matches
- **Recall**: Correct matches / Total actual matches
- **F1-Score**: Harmonic mean of precision and recall

## 📚 Course Context

**Semantic Data Integration Course** - University of Passau

**Learning Objectives:**
- Schema matching and mapping techniques
- Semantic similarity computation
- Data integration methodologies
- XML and structured data processing
- Performance evaluation in data integration

## 📄 Documentation

| Report | Description |
|--------|-------------|
| **Report_Task_01.pdf** | Mediated Schema Development |
| **Report_Task_02.pdf** | Schema Matching & Mapping |
| **Report_Task_03.pdf** | XML Parsing & Matching |

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **TheFuzz** - String matching
- **Gensim** - Word embeddings
- **ElementTree** - XML processing
- **JaroWinkler** - String similarity

## 📝 License

This project was developed for academic purposes as part of the Master's program at the University of Passau.

---

<div align="center">

**University of Passau** | **Summer Semester 2025** | **Semantic Data Integration Course**

</div>

