import pdfplumber
import pandas as pd
from pymongo import MongoClient
import re
from typing import Dict, List, Any
from datetime import datetime
from bson import ObjectId
import numpy as np

class TableDataExtractor:
    def __init__(self):
        self.numerical_pattern = re.compile(r'[-+]?\d*\.?\d+')

    def extract_value(self, cell: str) -> Any:
        if cell is None or str(cell).strip() == '':
            return None
        value = str(cell).strip()
        value = re.sub(r'[\u20b9$,]', '', value)
        try:
            if '.' in value:
                return float(value)
            elif value.replace('-', '').isdigit():
                return int(value)
            return value
        except ValueError:
            return value

    def process_table(self, table: List[List[str]]) -> List[Dict[str, Any]]:
        if not table or len(table) < 2:
            return []
        headers = []
        seen = set()
        for col in table[0]:
            header = str(col).strip().lower().replace(' ', '_') if col else 'unnamed'
            base_header = header
            counter = 1
            while header in seen:
                header = f"{base_header}_{counter}"
                counter += 1
            seen.add(header)
            headers.append(header)
        processed_rows = []
        for row in table[1:]:
            row_dict = {}
            has_data = False
            for idx, cell in enumerate(row):
                if idx < len(headers):
                    value = self.extract_value(cell)
                    if value is not None:
                        row_dict[headers[idx]] = value
                        has_data = True
            if has_data:
                processed_rows.append(row_dict)
        return processed_rows

class PDFDataExtractor:
    def __init__(self):
        self.table_extractor = TableDataExtractor()
        self.financial_keywords = [
            'revenue', 'profit', 'assets', 'liabilities', 'equity',
            'expenses', 'cash', 'tax', 'income', 'balance', 'total',
            'net', 'gross', 'operating', 'current', 'consolidated',
            'statement', 'notes', 'financial'
        ]

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        tables_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                table_settings = [
                    {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
                    {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
                ]
                for settings in table_settings:
                    try:
                        tables = page.extract_tables(table_settings=settings)
                        for table in tables:
                            if table and len(table) > 1:
                                processed_rows = self.table_extractor.process_table(table)
                                if processed_rows:
                                    table_data = {
                                        '_id': ObjectId(),
                                        'page_number': page_num,
                                        'data': processed_rows,
                                        'extraction_method': str(settings),
                                        'type': 'table',
                                        'extracted_at': datetime.now()
                                    }
                                    tables_data.append(table_data)
                    except Exception as e:
                        print(f"Warning: Error on page {page_num} with settings {settings}: {str(e)}")
        return tables_data

    def extract_text(self, pdf_path: str) -> List[Dict]:
        text_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    sections = self.split_into_sections(text)
                    for section in sections:
                        if self.is_relevant_financial_text(section):
                            embedding = self.tf_idf_embedding(section)
                            para_dict = {
                                '_id': ObjectId(),
                                'page_number': page_num,
                                'content': section.strip(),
                                'embedding': embedding,
                                'type': 'text',
                                'extracted_at': datetime.now()
                            }
                            text_data.append(para_dict)
        return text_data

    def tf_idf_embedding(self, text: str) -> List[float]:
        words = text.split()
        word_set = list(set(words))
        tf_idf_vector = [words.count(word) / len(words) for word in word_set]
        return tf_idf_vector[:300] + [0] * (300 - len(tf_idf_vector))

    def split_into_sections(self, text: str) -> List[str]:
        sections = re.split(r'\n\s*\n', text)
        final_sections = []
        for section in sections:
            subsections = re.split(r'(?:\d+\.|[A-Z]\.|\u2022|\*)\s+', section)
            final_sections.extend([s.strip() for s in subsections if s.strip()])
        return final_sections

    def is_relevant_financial_text(self, text: str) -> bool:
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.financial_keywords):
            return True
        if re.search(r'\d+\.?\d*', text):
            return True
        return False

class MongoDBHandler:
    def __init__(self, db_name: str = 'financial_db'):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.tables_collection = self.db['tables']
        self.text_collection = self.db['text']
        self.tables_collection.create_index([('page_number', 1)])
        self.text_collection.create_index([('page_number', 1)])

    def store_data(self, tables_data: List[Dict], text_data: List[Dict]):
        self.tables_collection.delete_many({})
        self.text_collection.delete_many({})
        if tables_data:
            batch_size = 50
            for i in range(0, len(tables_data), batch_size):
                batch = tables_data[i:i + batch_size]
                try:
                    self.tables_collection.insert_many(batch, ordered=False)
                except Exception as e:
                    print(f"Warning: Some tables may not have been inserted: {str(e)}")
        if text_data:
            batch_size = 50
            for i in range(0, len(text_data), batch_size):
                batch = text_data[i:i + batch_size]
                try:
                    self.text_collection.insert_many(batch, ordered=False)
                except Exception as e:
                    print(f"Warning: Some text data may not have been inserted: {str(e)}")

    def retrieve_similar_text(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = np.array(PDFDataExtractor().tf_idf_embedding(query))
        cursor = self.text_collection.find()
        results = []
        for doc in cursor:
            doc_embedding = np.array(doc['embedding'])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            results.append((similarity, doc))
        results.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in results[:top_k]]

def main():
    pdf_path = "dataset.pdf"
    pdf_extractor = PDFDataExtractor()
    mongo_handler = MongoDBHandler()
    try:
        print("Extracting tables...")
        tables_data = pdf_extractor.extract_tables(pdf_path)
        print(f"Extracted {len(tables_data)} tables.")
        print("Extracting text...")
        text_data = pdf_extractor.extract_text(pdf_path)
        print(f"Extracted {len(text_data)} text sections.")
        print("Storing data in MongoDB...")
        mongo_handler.store_data(tables_data, text_data)
        print("Data stored successfully.")
        print("Testing QA system...")
        query = "What is the total revenue?"
        results = mongo_handler.retrieve_similar_text(query)
        print("Relevant documents:")
        for result in results:
            print(f"Page {result['page_number']}: {result['content']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
