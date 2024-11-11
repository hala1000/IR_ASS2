import logging
import os
import pandas as pd
import csv
from typing import List, Tuple, Dict, Any
import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import MMapDirectory
from java.nio.file import Paths
lucene.initVM()

# Configure logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("program_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def init_lucene_index(index_path: str):
    """
    Initializes the Lucene index at the specified path.
    """
    try:
        path = Paths.get(index_path)
        index_dir = MMapDirectory(path)
        analyzer = StandardAnalyzer()
        config = IndexWriterConfig(analyzer)
        writer = IndexWriter(index_dir, config)
        logger.info("Lucene index initialized.")
        return writer, index_dir
    except Exception as e:
        logger.error(f"Error initializing Lucene index: {e}")
        return None, None

def index_document(filename: str, content: str, index_writer: IndexWriter):
    """
    Creates a Lucene Document from a filename and its content and adds it to the index.

    Args:
    filename (str): The name of the file.
    content (str): The content of the file.
    index_writer (IndexWriter): The Lucene IndexWriter object.
    """
    try:
        doc = Document()
        doc.add(TextField("filename", filename, Field.Store.YES))
        doc.add(TextField("content", content, Field.Store.YES))
        index_writer.addDocument(doc)
        logger.info(f"Document {filename} indexed successfully.")
    except Exception as e:
        logger.error(f"Error indexing document {filename}: {str(e)}")

def index_documents(data_path, index_writer):
    """
    Index all documents in the specified directory using the provided IndexWriter.

    Args:
    data_path (str): The path to the directory containing the documents.
    index_writer (IndexWriter): The Lucene IndexWriter used for indexing documents.

    This function reads each file in the directory, creates a document from its content,
    and adds it to the Lucene index.
    """
    # Check if the data path exists
    if not os.path.exists(data_path):
        logging.error(f"The directory {data_path} does not exist.")
        return

    # Iterate over each file in the directory
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Index the document using the index_document function
                index_document(filename, content, index_writer)
        except Exception as e:
            logging.error(f"Failed to read or index file {file_path}: {str(e)}")

def read_queries(queries_csv: str) -> List[Tuple[Any, str]]:
    """
    Reads queries from a CSV file and returns a list of tuples containing (query_id, query_text).
    """
    queries = []
    try:
        df = pd.read_csv(queries_csv, sep='\t')
        required_columns = {'Query number', 'Query'}
        if not required_columns.issubset(df.columns):
            logger.error(f"CSV missing required columns: {required_columns - set(df.columns)}")
            return queries

        for _, row in df.iterrows():
            queries.append((row['Query number'], row['Query']))
        logger.info(f"Total queries read: {len(queries)}")
    except Exception as e:
        logger.error(f"Error reading queries file {queries_csv}: {e}")
    return queries

def search_index(searcher: IndexSearcher, query_parser: QueryParser, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Searches the index and retrieves top n results for the given query.
    """
    results = []
    try:
        lucene_query = query_parser.parse(query)
        #logger.info(f"1--lucene_query are: {lucene_query}")
        hits = searcher.search(lucene_query, top_n).scoreDocs
        #logger.info(f"2--hits are: {hits}")
        storedFields = searcher.storedFields()
        for hit in hits:
            #logger.info(f"3--hit are: {hit}")
            doc_id = hit.doc
            document = storedFields.document(doc_id)
            #logger.info(f"4--file name are: {document.get("filename")}")
            results.append({"filename": document.get("filename"), "score": hit.score})
        logger.info(f"Search completed for query: {query}")
    except Exception as e:
        logger.error(f"Failed to search index for query '{query}': {e}")
    return results

def save_results_to_csv(results: List[Tuple[Any, Any, float]], output_csv: str):
    """
    Saves the search results to a CSV file.
    """
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Query_number', 'doc_id', 'score'])
            for result in results:
                writer.writerow(result)
        logger.info("Results successfully saved to CSV.")
    except Exception as e:
        logger.error(f"Error saving results to {output_csv}: {e}")

# Main execution
if __name__ == "__main__":
    index_path = 'index3'  # Path to Lucene index directory
    data_path = "full_docs"  # Path to the directory containing documents for indexing
    queries_csv = 'queries.csv'  # Path to queries file
    output_csv = 'search_results2.csv'  # Path to output CSV file

    writer, index_dir = init_lucene_index(index_path)
    if writer is not None and index_dir is not None:
        try:
            # Perform indexing operations
            index_documents(data_path, writer)
        finally:
            # Ensure the writer is closed properly
            writer.close()
            logger.info("IndexWriter closed.")
        print("hala\n")
        reader = DirectoryReader.open(index_dir)
        searcher = IndexSearcher(reader)
        query_parser = QueryParser("content", StandardAnalyzer())

        queries = read_queries(queries_csv)
        all_results = []
        for query_id, query_text in queries:
            results = search_index(searcher, query_parser, query_text)
            all_results.extend([(query_id, res['filename'], res['score']) for res in results])

        save_results_to_csv(all_results, output_csv)

        # Clean up
        try:
            writer.close()
            index_dir.close()
        except Exception as e:
            logger.error(f"Failed to close index resources: {e}")
