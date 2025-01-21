import streamlit as st
from pdfplumber import pdfplumber
import os
from pymongo import MongoClient
import numpy as np
from typing import List
from pdf_data_extraction import PDFDataExtractor, MongoDBHandler  # Assuming this file is named pdf_data_extraction.py

# Initialize components
pdf_extractor = PDFDataExtractor()
mongo_handler = MongoDBHandler()

# Streamlit App
st.title("Financial Data QA Bot")
st.sidebar.header("Upload and Query")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload a P&L PDF", type=["pdf"])
query = st.text_input("Enter your financial query", placeholder="What is the total revenue?")

if uploaded_file:
    pdf_path = os.path.join("temp", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("File uploaded successfully!")

    if st.sidebar.button("Process PDF"):
        st.write("Extracting data...")
        tables_data = pdf_extractor.extract_tables(pdf_path)
        text_data = pdf_extractor.extract_text(pdf_path)

        if tables_data or text_data:
            mongo_handler.store_data(tables_data, text_data)
            st.write(f"Extracted {len(tables_data)} tables and {len(text_data)} text sections.")
        else:
            st.warning("No data extracted from the PDF.")

if query:
    st.write(f"Query: {query}")
    results = mongo_handler.retrieve_similar_text(query, top_k=5)

    if results:
        st.write("Relevant Results:")
        for result in results:
            st.write(f"Page {result['page_number']}: {result['content']}")
    else:
        st.warning("No relevant results found.")

# Create temporary folder
if not os.path.exists("temp"):
    os.makedirs("temp")
