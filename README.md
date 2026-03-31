# RAG Project

A Retrieval-Augmented Generation (RAG) application built with Python and web technologies, enabling intelligent document processing and AI-powered question answering.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

RAG Project is an intelligent document retrieval and generation system that combines document storage, semantic search, and AI language models to provide context-aware answers to user queries. It supports multiple LLM providers including OpenAI and Cohere.

## ✨ Features

- 📄 **Document Upload & Processing**: Upload and process various document formats
- 🔍 **Semantic Search**: Retrieve relevant documents using intelligent search capabilities
- 🤖 **AI Integration**: Support for multiple AI providers (OpenAI, Cohere)
- 🌐 **Web Interface**: Interactive HTML-based user interface for easy interaction
- 📦 **Document Storage**: Persistent storage for uploaded documents
- ✅ **API Key Validation**: Built-in validation for API keys from different providers
- 🚀 **Cross-Platform Support**: Batch files and PowerShell scripts for Windows, Python scripts for all platforms

## 🏗️ Project Structure

```
RAG_project/
├── main.py                 # Main application entry point
├── check_api_key.py        # OpenAI API key validation
├── check_cohere_key.py     # Cohere API key validation
├── index.html              # Web UI interface
├── requirements.txt        # Python dependencies
├── run.bat                 # Windows batch startup script
├── start.ps1               # PowerShell startup script
├── controllers/            # API controllers (expandable)
├── services/               # Service layer modules
├── storage/                # Persistent storage for documents
└── uploads/                # Temporary upload directory
```

## 💻 Technology Stack

- **Backend**: Python 3.x
- **Frontend**: HTML5
- **API Integration**: OpenAI, Cohere
- **Deployment**: Cross-platform (Windows, Linux, macOS)

**Language Composition**:
- Python: 52.7%
- HTML: 44.0%
- PowerShell: 1.8%
- Batchfile: 1.5%

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for:
  - OpenAI (for GPT models)
  - Cohere (optional, for alternative LLM)
- Modern web browser for UI access

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Nuvesh/RAG_project.git
cd RAG_project
```

### 2. Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### API Key Setup

Before running the application, configure your API keys:

#### OpenAI API Key

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

#### Cohere API Key (Optional)

1. Get your API key from [Cohere Dashboard](https://dashboard.cohere.ai/)
2. Set the environment variable:

**Windows:**
```cmd
set COHERE_API_KEY=your_api_key_here
```

**macOS/Linux:**
```bash
export COHERE_API_KEY="your_api_key_here"
```

### Validate API Keys

Verify your API key configuration:

**For OpenAI:**
```bash
python check_api_key.py
```

**For Cohere:**
```bash
python check_cohere_key.py
```

## 🚀 Usage

### Option 1: Run with Python (All Platforms)

```bash
python main.py
```

### Option 2: Run on Windows with Batch File

```bash
run.bat
```

### Option 3: Run on Windows with PowerShell

```powershell
.\start.ps1
```

### Accessing the Application

1. Start the application using one of the methods above
2. Open your web browser
3. Navigate to `http://localhost:5000` (or the configured port)
4. Use the web interface to upload documents and ask questions

## 📡 API Endpoints

The application provides the following endpoints:

### Document Management
- `POST /upload` - Upload a document
- `GET /documents` - List all documents
- `DELETE /documents/<id>` - Delete a document

### Query Processing
- `POST /query` - Submit a query for RAG processing
- `GET /query/history` - Retrieve query history

### Health Check
- `GET /health` - Check application status

## 🗂️ Project Structure Details

### Core Files

- **main.py**: Application entry point, initializes Flask/FastAPI server and routing
- **check_api_key.py**: Validates OpenAI API credentials and connection
- **check_cohere_key.py**: Validates Cohere API credentials and connection
- **index.html**: Single-page application UI with document upload and query interface
- **requirements.txt**: All Python package dependencies

### Directories

- **controllers/**: Contains API route handlers and request processing logic
- **services/**: Business logic layer for document processing, retrieval, and LLM interactions
- **storage/**: Persistent document storage directory
- **uploads/**: Temporary directory for incoming file uploads

## 📦 Dependencies

Key dependencies (see `requirements.txt` for full list):

- Flask or FastAPI - Web framework
- OpenAI - GPT API integration
- Cohere - Alternative LLM integration
- Additional libraries for document processing, embeddings, and utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Notes

- Ensure you keep your API keys secure and never commit them to the repository
- Use environment variables for sensitive configuration
- For production deployment, implement proper error handling and security measures
- Consider implementing rate limiting and authentication for the API endpoints

## ⚠️ Troubleshooting

### Issue: "API key not found" error

**Solution**: Ensure your API key environment variable is properly set before starting the application.

### Issue: Port already in use

**Solution**: Change the port in `main.py` or stop the process using the current port.

### Issue: Module not found errors

**Solution**: Verify all dependencies are installed with `pip install -r requirements.txt`

## 📞 Support

For issues, questions, or suggestions, please create an issue in the repository.

---

**Last Updated**: March 24, 2026  
**Repository**: [Nuvesh/RAG_project](https://github.com/Nuvesh/RAG_project)
