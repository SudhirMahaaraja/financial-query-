import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pymongo import MongoClient
from datetime import datetime
import torch
import pypdf
import re


class MongoDBVectorStore:
    def __init__(self, connection_string: str, db_name: str):
        """Initialize MongoDB connection with vector search capabilities"""
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]

        # Collections
        self.financial_statements = self.db.financial_statements
        self.embeddings = self.db.embeddings
        self.metrics = self.db.financial_metrics
        self.query_logs = self.db.query_logs

        # Create indexes
        self.create_indexes()

    def create_indexes(self):
        """Create necessary indexes including vector search index"""
        # Regular indexes
        self.financial_statements.create_index([("statement_type", 1), ("date", -1)])
        self.financial_statements.create_index([("company_name", 1)])
        self.metrics.create_index([("metric_name", 1), ("date", -1)])

        # Vector search index
        self.embeddings.create_index([("vector", "2dsphere")])

    def store_embedding(self, text: str, vector: List[float], metadata: Dict[str, Any]):
        """Store text embedding in MongoDB"""
        document = {
            'text': text,
            'vector': vector,
            'metadata': metadata,
            'created_at': datetime.utcnow()
        }
        return self.embeddings.insert_one(document)

    def vector_search(self, query_vector: List[float], limit: int = 3):
        """Perform vector similarity search using MongoDB"""
        pipeline = [
            {
                "$search": {
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "vector",
                        "k": limit
                    }
                }
            },
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"}
                }
            }
        ]
        return list(self.embeddings.aggregate(pipeline))


class FinancialRAGSystem:
    def __init__(self, mongo_connection_string: str, mongo_db_name: str):
        # Initialize MongoDB
        self.db = MongoDBVectorStore(mongo_connection_string, mongo_db_name)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    def extract_table_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract and structure financial data from PDF"""
        # Example structure based on the Infosys statements
        financial_data = {
            'revenue': {
                'total_revenue': 40824,
                'other_income': 1141
            },
            'expenses': {
                'employee_benefit_expenses': 20303,
                'travel_expenses': 457,
                'professional_charges': 889,
                'other_expenses': 3941
            },
            'profit': {
                'profit_before_tax': 7173,
                'profit_for_period': 5923
            }
        }
        return financial_data

    def process_financial_statement(self, pdf_path: str, company_name: str, statement_date: str):
        """Process financial statement and store in MongoDB"""
        # Extract data from PDF
        financial_data = self.extract_table_from_pdf(pdf_path)

        # Store raw financial data
        statement_doc = {
            'company_name': company_name,
            'date': statement_date,
            'data': financial_data,
            'statement_type': 'consolidated',
            'source_file': pdf_path,
            'created_at': datetime.utcnow()
        }
        self.db.financial_statements.insert_one(statement_doc)

        # Create and store embeddings
        self.create_and_store_embeddings(financial_data, company_name, statement_date)

        # Calculate and store metrics
        metrics = self.calculate_financial_metrics(financial_data)
        metrics.update({
            'company_name': company_name,
            'date': statement_date
        })
        self.db.metrics.insert_one(metrics)

    def create_and_store_embeddings(self, data: Dict[str, Any], company_name: str, date: str):
        """Create and store embeddings for financial data"""
        for category, items in data.items():
            for item, value in items.items():
                text = f"{category} - {item}: {value}"
                vector = self.embedding_model.encode(text).tolist()

                metadata = {
                    'company_name': company_name,
                    'date': date,
                    'category': category,
                    'item': item,
                    'value': value
                }

                self.db.store_embedding(text, vector, metadata)

    def calculate_financial_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key financial metrics"""
        metrics = {}

        if 'revenue' in financial_data and 'total_revenue' in financial_data['revenue']:
            total_revenue = financial_data['revenue']['total_revenue']

            if 'profit' in financial_data and 'profit_for_period' in financial_data['profit']:
                profit = financial_data['profit']['profit_for_period']
                metrics['profit_margin'] = (profit / total_revenue) * 100

            if 'expenses' in financial_data:
                total_expenses = sum(financial_data['expenses'].values())
                metrics['expense_ratio'] = (total_expenses / total_revenue) * 100

        return metrics

    def process_query(self, query: str) -> str:
        """Process user query and generate response"""
        # Create query embedding
        query_vector = self.embedding_model.encode(query).tolist()

        # Retrieve relevant context
        vector_results = self.db.vector_search(query_vector)

        # Generate answer
        answer = self.generate_answer(query, vector_results)

        # Log query
        self.log_query(query, answer)

        return answer

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer using retrieved context"""
        # Format context
        context_text = " ".join([item['text'] for item in context])

        # Create prompt
        prompt = f"Context: {context_text}\nQuestion: {query}\nAnswer:"

        # Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def log_query(self, query: str, response: str):
        """Log query and response"""
        log = {
            'query': query,
            'response': response,
            'timestamp': datetime.utcnow()
        }
        self.db.query_logs.insert_one(log)


# Example usage
def main():
    # Initialize system
    rag_system = FinancialRAGSystem(
        mongo_connection_string='mongodb://localhost:27017',
        mongo_db_name='financial_query'
    )

    # Process financial statement
    rag_system.process_financial_statement(
        pdf_path="dataset.pdf",
        company_name="Infosys Limited",
        statement_date="2024-03-31"
    )

    # Example queries
    queries = [
        "What was Infosys's total revenue for Q1 2024?",
        "Calculate the profit margin for the latest quarter",
        "How have employee benefit expenses changed compared to last year?",
        "What are the main components of operating expenses?"
    ]

    for query in queries:
        answer = rag_system.process_query(query)
        print(f"\nQ: {query}")
        print(f"A: {answer}")


if __name__ == "__main__":
    main()