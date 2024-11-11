import pandas as pd
import re
import logging
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("lucene_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_number(file_name: str) -> Optional[str]:
    """
    Extracts the first numeric sequence from a given file name using regular expressions.

    Args:
        file_name (str): The name of the file containing numeric sequences.

    Returns:
        Optional[str]: The first numeric sequence found or None if no numbers are present.
    """
    logger.debug(f"Extracting number from file name: {file_name}")
    numbers = re.findall(r'\d+', file_name)
    if numbers:
        logger.debug(f"Numbers found: {numbers}")
        return numbers[0]
    logger.debug("No numbers found.")
    return None


def average_precision_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
    """
    Calculates the Average Precision at K (AP@K) for a single query.

    Args:
        relevant_docs (List[str]): List of relevant document IDs for the query.
        retrieved_docs (List[str]): List of retrieved document IDs for the query.
        k (int): The cutoff rank.

    Returns:
        float: The average precision at K.
    """
    logger.debug(f"Calculating AP@{k} for query.")
    relevant_set = set(relevant_docs)
    retrieved_k = retrieved_docs[:k]
    score = 0.0
    num_hits = 0

    for i, doc in enumerate(retrieved_k, start=1):
        if doc in relevant_set:
            num_hits += 1
            precision_at_i = num_hits / i
            score += precision_at_i
            logger.debug(f"Hit #{num_hits} at position {i}: {doc} (Precision: {precision_at_i:.4f})")

    if not relevant_set:
        logger.warning("No relevant documents provided for this query.")
        return 0.0

    average_precision = score / min(len(relevant_set), k)
    logger.debug(f"Average Precision@{k}: {average_precision:.4f}")
    return average_precision


def mean_average_precision(
    queries_relevant_docs: Dict[str, List[str]],
    queries_retrieved_docs: Dict[str, List[str]],
    k: int
) -> float:
    """
    Computes the Mean Average Precision at K (MAP@K) over multiple queries.

    Args:
        queries_relevant_docs (Dict[str, List[str]]): Mapping from query IDs to relevant document IDs.
        queries_retrieved_docs (Dict[str, List[str]]): Mapping from query IDs to retrieved document IDs.
        k (int): The cutoff rank.

    Returns:
        float: The mean average precision at K.
    """
    logger.info(f"Calculating MAP@{k} for all queries.")
    scores = []
    for query_id, relevant_docs in queries_relevant_docs.items():
        retrieved_docs = queries_retrieved_docs.get(query_id, [])
        ap_k = average_precision_at_k(relevant_docs, retrieved_docs, k)
        scores.append(ap_k)
        logger.debug(f"Query ID: {query_id} - AP@{k}: {ap_k:.4f}")

    map_k = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"Mean Average Precision@{k}: {map_k:.4f}")
    return map_k


def recall_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
    """
    Calculates the Recall at K (Recall@K) for a single query.

    Args:
        relevant_docs (List[str]): List of relevant document IDs for the query.
        retrieved_docs (List[str]): List of retrieved document IDs for the query.
        k (int): The cutoff rank.

    Returns:
        float: The recall at K.
    """
    logger.debug(f"Calculating Recall@{k} for query.")
    relevant_set = set(relevant_docs)
    retrieved_k_set = set(retrieved_docs[:k])
    intersection_count = len(relevant_set & retrieved_k_set)

    if not relevant_set:
        logger.warning("No relevant documents provided for this query.")
        return 0.0

    recall = intersection_count / len(relevant_set)
    logger.debug(f"Recall@{k}: {recall:.4f}")
    return recall


def mean_average_recall(
    queries_relevant_docs: Dict[str, List[str]],
    queries_retrieved_docs: Dict[str, List[str]],
    k: int
) -> float:
    """
    Computes the Mean Average Recall at K (MAR@K) over multiple queries.

    Args:
        queries_relevant_docs (Dict[str, List[str]]): Mapping from query IDs to relevant document IDs.
        queries_retrieved_docs (Dict[str, List[str]]): Mapping from query IDs to retrieved document IDs.
        k (int): The cutoff rank.

    Returns:
        float: The mean average recall at K.
    """
    logger.info(f"Calculating MAR@{k} for all queries.")
    recalls = []
    for query_id, relevant_docs in queries_relevant_docs.items():
        retrieved_docs = queries_retrieved_docs.get(query_id, [])
        recall = recall_at_k(relevant_docs, retrieved_docs, k)
        recalls.append(recall)
        logger.debug(f"Query ID: {query_id} - Recall@{k}: {recall:.4f}")

    mar_k = sum(recalls) / len(recalls) if recalls else 0.0
    logger.info(f"Mean Average Recall@{k}: {mar_k:.4f}")
    return mar_k


def load_data(relevant_filepath: str, retrieved_filepath: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads relevant and retrieved documents data from CSV files.

    Args:
        relevant_filepath (str): Path to the CSV file containing relevant documents.
        retrieved_filepath (str): Path to the CSV file containing retrieved documents.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for relevant and retrieved documents.
    """
    logger.info("Loading data from CSV files.")
    try:
        relevant_data = pd.read_csv(relevant_filepath)
        logger.info(f"Loaded relevant data with {len(relevant_data)} records.")
    except Exception as e:
        logger.error(f"Failed to load relevant data: {e}")
        raise

    try:
        retrieved_data = pd.read_csv(retrieved_filepath)
        logger.info(f"Loaded retrieved data with {len(retrieved_data)} records.")
    except Exception as e:
        logger.error(f"Failed to load retrieved data: {e}")
        raise

    return relevant_data, retrieved_data


def prepare_document_dicts(
    relevant_data: pd.DataFrame,
    retrieved_data: pd.DataFrame
) -> (Dict[str, List[str]], Dict[str, List[str]]):
    """
    Prepares dictionaries mapping query IDs to relevant and retrieved document IDs.

    Args:
        relevant_data (pd.DataFrame): DataFrame containing relevant documents.
        retrieved_data (pd.DataFrame): DataFrame containing retrieved documents.

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
            - queries_relevant_docs: Mapping from query IDs to relevant document IDs.
            - queries_retrieved_docs: Mapping from query IDs to retrieved document IDs.
    """
    logger.info("Preparing dictionaries for relevant and retrieved documents.")

    queries_relevant_docs: Dict[str, List[str]] = {}
    for _, row in relevant_data.iterrows():
        query_id = str(row['Query_number'])
        document_name = str(row['doc_number'])
        queries_relevant_docs.setdefault(query_id, []).append(document_name)
        logger.debug(f"Relevant - Query ID: {query_id}, Document: {document_name}")

    queries_retrieved_docs: Dict[str, List[str]] = {}
    for _, row in retrieved_data.iterrows():
        query_id = str(row['Query_number'])
        document_id = extract_number(str(row['doc_number']))
        if document_id:
            queries_retrieved_docs.setdefault(query_id, []).append(document_id)
            logger.debug(f"Retrieved - Query ID: {query_id}, Document ID: {document_id}")
        else:
            logger.warning(f"Document ID extraction failed for: {row['doc_number']}")

    logger.info("Finished preparing document dictionaries.")
    return queries_relevant_docs, queries_retrieved_docs


def main():
    # Define file paths
    relevant_filepath = "dev_query_results.csv"
    retrieved_filepath = "result.csv"

    # Load data
    relevant_data, retrieved_data = load_data(relevant_filepath, retrieved_filepath)

    # Prepare dictionaries
    queries_relevant_docs, queries_retrieved_docs = prepare_document_dicts(relevant_data, retrieved_data)

    print (queries_relevant_docs)
    print("="*40)
    print (queries_retrieved_docs)

    # Define k values
    k_values = [1,3,5,10]

    # Calculate and display metrics
    for k in k_values:
        logger.info(f"\nEvaluating metrics for k={k}")
        map_k = mean_average_precision(queries_relevant_docs, queries_retrieved_docs, k)
        mar_k = mean_average_recall(queries_relevant_docs, queries_retrieved_docs, k)
        print(f"MAP@{k}: {map_k:.4f}")
        print(f"MAR@{k}: {mar_k:.4f}")


if __name__ == "__main__":
    main()
