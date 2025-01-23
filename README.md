# financial-query-bot

A powerful financial document analysis tool that extracts and processes data from PDF financial statements, enabling natural language queries for financial information retrieval.

## 🌟 Features

- PDF financial statement processing and data extraction
- Table detection and structured data parsing
- Intelligent text segmentation and analysis
- Natural language querying capability
- MongoDB integration for data persistence
- Interactive web interface using Streamlit
- TF-IDF based similarity search for relevant information retrieval



## 📋 Prerequisites

- Python 3.8 or higher
- MongoDB installed and running locally
- pip (Python package manager)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-query-bot.git
cd financial-query-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Ensure MongoDB is running locally on default port (27017)

## 📁 Project Structure

```
financial-query-bot/
├── app.py                    # Streamlit web application
├── pdf_data_extraction.py    # Core PDF processing classes
├── requirements.txt          # Project dependencies
└── temp/                     # Temporary storage for uploaded PDFs
```

## 🔧 Components

### 1. TableDataExtractor
- Handles extraction of tabular data from PDFs
- Processes and normalizes table headers
- Converts cell values to appropriate data types
- Handles currency symbols and number formatting

### 2. PDFDataExtractor
- Manages overall PDF data extraction process
- Extracts both tables and relevant text sections
- Implements TF-IDF based text vectorization
- Filters content using financial keywords

### 3. MongoDBHandler
- Manages database operations
- Stores extracted tables and text data
- Implements similarity-based search
- Handles batch processing for large datasets

## 💡 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload a financial PDF document (e.g., annual report, P&L statement)

4. Click "Process PDF" to extract and store the data

5. Enter natural language queries in the search box, such as:
   - "What is the total revenue?"
   - "Show me the operating expenses"
   - "What was the net profit in the last quarter?"

## 🔍 Query Processing

The system processes queries using the following approach:
1. Converts query to TF-IDF vector representation
2. Calculates similarity with stored text sections
3. Returns top-k most relevant results
4. Displays results with page numbers for reference

## ⚙️ Configuration

Key configurations are stored in the respective class initializations:
- MongoDB connection settings in `MongoDBHandler`
- Financial keywords list in `PDFDataExtractor`
- Table extraction settings in `extract_tables` method

## 📊 Performance Optimization

- Batch processing for MongoDB operations
- Indexed collections for faster queries
- Multiple table extraction strategies
- Efficient text sectioning and filtering

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF extraction capabilities
- [Streamlit](https://streamlit.io/) for the web interface
- [MongoDB](https://www.mongodb.com/) for database functionality

## ⚠️ Limitations

- Currently supports PDFs with well-structured tables
- Requires local MongoDB installation
- Processing time depends on PDF complexity
- Limited to text-based financial data extraction
