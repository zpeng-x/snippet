timport sys
import pandas as pd
import json
import os
from datetime import datetime
sys.path.append("gen-py")  # Add gen-py to Python path
import logging
from gen.twitter.strato.columns.ads.ads_prediction.thriftpy.ttypes import UserConversionHistoryInfo, UserConversionHistoryInfoList
from async_http_client import make_single_read
import asyncio
import aiohttp
import requests
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError, ConnectionError, Timeout
from typing import List, Dict, Any, Tuple
import time
from collections import defaultdict
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_date", type=str, required=True)
args = parser.parse_args()

###################CONSTANTS###########################
REQUEST_DELAY = 0.1
USER_CONVERSION_HISTORY_SIZE = 100
MAX_CONCURRENCY = 100
BATCH_SIZE = 1000
########################################################

# Define retry configuration
@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s
    retry=retry_if_exception_type((HTTPError, ConnectionError, Timeout)),  # Retry on these exceptions
    reraise=True  # Reraise the last exception if all retries fail
)
def make_request(url, payload):
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()  # This will raise if status >= 400
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        # Handle the error here
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error occurred: {e}")
        # Handle other request-related errors

        

############### Initialize BigQuery Client ###############

from google.cloud import bigquery, bigquery_storage
from google.oauth2 import service_account


WD = "/var/lib/mesos/slaves/1773e61b-66cc-4c76-9f4e-6a16e4569303-S6739/frameworks/201205082337-0000000003-0000/executors/thermos-ads-prediction-devel-zhejianp-spark-notebook-0-28b245fb-90b5-4c0c-a70a-445d37a1cd2f/runs/0754985d-601f-4748-a7de-aaf52c448cb5/sandbox/workspace/"
# Path to your service account JSON key file
key_path = os.path.join(WD, "twttr/twttr-rev-core-data-prod-d3dac275bdaf.txt")

# Load credentials
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Initialize BigQuery client with the service account
client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,  # Or specify your project ID: "your-project-id"
    location="US"  # Match the dataset's location (e.g., US, EU)
)

bqstorage_client = bigquery_storage.BigQueryReadClient(credentials=credentials)
########################################################


# Global map of user_id -> asyncio.Lock
user_locks = defaultdict(asyncio.Lock)

def deduplicate_by_impression(conversions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate conversions by impressionId while preserving order."""
    # seen = set()
    # deduped = []
    # for conv in conversions:
    #     imp_id = int(conv.get("impressionId"))
    #     if imp_id not in seen:
    #         seen.add(imp_id)
    #         deduped.append(conv)
    
    # One liner
    deduped = list({int(conv.get("impressionId")): conv for conv in conversions}.values())
    
    return deduped


class AsyncUserConversionWriter:
    """Async HTTP client for writing user conversion signals to Manhattan."""

    def __init__(self, max_concurrency: int = 50, max_retries: int = 3):
        self.put_url = "https://strato.twitter.biz/op/put/targeting/sourced_ads/prod/userConversionHistoryInfoManhattan"
        self.fetch_url = "https://strato.twitter.biz/op/fetch/targeting/sourced_ads/prod/userConversionHistoryInfoManhattan"
        self.kafka_url = "https://strato.twitter.biz/op/insert/targeting/sourced_ads/prod/userConversionHistoryInfoKafka"
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.session: aiohttp.ClientSession = None
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Context manager enter - initialize session."""
        connector = aiohttp.TCPConnector(limit=self.max_concurrency, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=120, connect=30)  # Increased timeouts
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        if self.session:
            await self.session.close()

    async def _write_single_conversion(self, user_id: int, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Write a single user conversion record asynchronously to both Manhattan and Kafka."""
        async with self.semaphore:
            json_payload = json.dumps([user_id, payload])

            for attempt in range(self.max_retries):
                try:
                    # Make concurrent requests to both URLs
                    manhattan_task = self.session.post(self.put_url, data=json_payload)
                    kafka_task = self.session.post(self.kafka_url, data=json_payload)

                    # Execute both requests simultaneously
                    results = await asyncio.gather(manhattan_task, kafka_task, return_exceptions=True)

                    # Check results from both requests
                    manhattan_success = True
                    kafka_success = True
                    error_messages = []

                    # Process Manhattan result
                    if isinstance(results[0], Exception):
                        manhattan_success = False
                        error_messages.append(f"Manhattan error: {str(results[0])}")
                        self.logger.warning(f"Manhattan request failed for user {user_id}, attempt {attempt + 1}: {results[0]}")
                    else:
                        async with results[0] as response:
                            if response.status >= 400:
                                manhattan_success = False
                                error_text = await response.text()
                                error_messages.append(f"Manhattan HTTP {response.status}: {error_text}")
                                self.logger.warning(f"Manhattan HTTP {response.status} for user {user_id}: {error_text}")

                    # Process Kafka result
                    if isinstance(results[1], Exception):
                        kafka_success = False
                        error_messages.append(f"Kafka error: {str(results[1])}")
                        self.logger.warning(f"Kafka request failed for user {user_id}, attempt {attempt + 1}: {results[1]}")
                    else:
                        async with results[1] as response:
                            if response.status >= 400:
                                kafka_success = False
                                error_text = await response.text()
                                error_messages.append(f"Kafka HTTP {response.status}: {error_text}")
                                self.logger.warning(f"Kafka HTTP {response.status} for user {user_id}: {error_text}")

                    # Both requests must succeed
                    if manhattan_success and kafka_success:
                        return True, ""
                    elif attempt < self.max_retries - 1:
                        # If not the last attempt, wait and retry
                        error_msg = "; ".join(error_messages)
                        self.logger.warning(f"Both requests failed for user {user_id}, attempt {attempt + 1}: {error_msg}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        # Last attempt failed
                        error_msg = "; ".join(error_messages)
                        return False, error_msg

                except Exception as e:
                    self.logger.warning(f"Unexpected error for user {user_id}, attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False, str(e)

            return False, "Max retries exceeded"

    async def _fetch_existing_conversions(self, user_id: int) -> List[Dict[str, Any]]:
        """Fetch existing conversion history for a user."""
        async with self.semaphore:
            json_payload = json.dumps([user_id, None])

            for attempt in range(self.max_retries):
                try:
                    async with self.session.post(self.fetch_url, data=json_payload) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            self.logger.warning(f"HTTP {response.status} for user {user_id}: {error_text}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return []

                        # Success - parse response
                        resp_json = await response.json()
                        curr_conversion_list = resp_json.get('v', {}).get('userConversionHistoryInfoList', [])
                        return curr_conversion_list

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    self.logger.warning(f"Fetch failed for user {user_id}, attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return []

            return []

    async def write_conversions_batch(self, batch_data: List[Tuple[int, Dict[str, Any]]]) -> Tuple[int, int, List[Tuple[int, str]]]:
        """
        Write a batch of user conversion records concurrently.

        Returns:
            Tuple of (success_count, total_count, failed_records)
            where failed_records is List[(user_id, error_message)]
        """
        if not batch_data:
            return 0, 0, []

        tasks = [
            self._write_single_conversion(user_id, payload)
            for user_id, payload in batch_data
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        failed_records = []

        for (user_id, _), result in zip(batch_data, results):
            if isinstance(result, Exception):
                self.logger.error(f"Unexpected error for user {user_id}: {result}")
                failed_records.append((user_id, str(result)))
            else:
                success, error_msg = result
                if success:
                    success_count += 1
                else:
                    failed_records.append((user_id, error_msg))

        return success_count, len(batch_data), failed_records


def read_conversion_signal_from_bq():
    # table_most_recent_date_query = """
    #     SELECT MAX(_date) AS most_recent_date
    #     FROM `twttr-rev-core-data-prod.dpa.dpa_attributed_conversion_30d_20250901`
    #     WHERE _date IS NOT NULL
    # """
    # fallback_df = client.query(table_most_recent_date_query).to_dataframe(bqstorage_client=bqstorage_client)
    # most_recent_date = fallback_df["most_recent_date"].iloc[0]
    logging.info(f"Most recent date: {args.run_date}")
    query = f"""
            SELECT 
                userId64, 
                impressionId, 
                isDpa, 
                ARRAY(
                    SELECT x
                    FROM UNNEST(uniqueConversionTypes) AS x
                    WHERE x IN (2, 12, 13)
                ) AS uniqueConversionTypes,
                productKeys, 
                clickedProductKey, 
                conversion_time_ms, 
                impressionCallbackEpochTimeMilliSec, 
                octOrMact, 
                advertiserId, 
                accountId
            FROM `twttr-rev-core-data-prod.dpa.dpa_attributed_conversion_30d_{args.run_date}`
            WHERE DATE(_date) = '{datetime.strptime(args.run_date, "%Y%m%d").strftime("%Y-%m-%d")}'
                AND EXISTS (
                    SELECT 1
                    FROM UNNEST(uniqueConversionTypes) AS x
                    WHERE x IN (2, 12, 13)
                ) 
            """     
    # print("processing most recent date:", args.run_date)
    logging.info(f"Processing most recent date: {args.run_date}")
    query_job = client.query(query)
    df = query_job.to_dataframe()
    # print(f"Number of rows: {df.shape[0]}")
    logging.info(f"Number of rows: {df.shape[0]}")
    return df


df = read_conversion_signal_from_bq()

target_conversions = [2, 12, 13]
mask = df["uniqueConversionTypes"].apply(
    lambda lst: any(x in target_conversions for x in lst)
)

filtered_df = df[mask]
filtered_df = filtered_df.replace({pd.NA: None})
int_cols = filtered_df.select_dtypes(include=[np.integer]).columns
filtered_df[int_cols] = filtered_df[int_cols].map(int)

# Process data and write to mh
# Write to manhattan
failed_impressions_file = "failed_impressions.txt"

async def process_conversions_async(filtered_df: pd.DataFrame, batch_size: int = 1000, max_concurrency: int = 50) -> Tuple[int, int]:
    """
    Process user conversion data asynchronously with batching and per-batch fetching.
    
    Args:
        filtered_df: DataFrame containing filtered conversion data
        batch_size: Number of records to process in each batch
        max_concurrency: Maximum concurrent requests
    
    Returns:
        Tuple of (total_success, total_processed)
    """
    logging.info(f"Starting async processing of {len(filtered_df)} conversion records")
    logging.info(f"Batch size: {batch_size}, Max concurrency: {max_concurrency}")

    start_time = time.time()
    total_success = 0
    total_processed = 0
    all_failed_records = []

    async with AsyncUserConversionWriter(max_concurrency=max_concurrency) as writer:
        # Process in batches
        for i in range(0, len(filtered_df), batch_size):
            batch_df = filtered_df.iloc[i:i + batch_size]
            batch_start_time = time.time()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(filtered_df) + batch_size - 1)//batch_size} "
                  f"({len(batch_df)} records)")
            
            # Get unique user_ids for this batch
            batch_user_ids = [int(x) for x in batch_df['userId64'].unique()]
            print(f"Fetching existing conversions for {len(batch_user_ids)} unique users in this batch")
            
            # Fetch existing conversions for users in this batch
            read_start_time = time.time()
            fetch_tasks = [writer._fetch_existing_conversions(user_id) for user_id in batch_user_ids]
            existing_conversions_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            read_end_time = time.time()
            
            # Create mapping of user_id to existing conversions for this batch
            batch_user_existing_conversions = defaultdict(list) # user_id -> list of existing conversions
            for user_id, result in zip(batch_user_ids, existing_conversions_results):
                if isinstance(result, Exception):
                    print(f"Failed to fetch existing conversions for user {user_id}: {result}")
                    batch_user_existing_conversions[user_id] = list()
                else:
                    batch_user_existing_conversions[user_id] = result
            
            # Prepare batch data for this batch
            batch_data = defaultdict(list[dict]) # user_id -> list of payloads
            for idx, row in batch_df.iterrows():
                user_id = row['userId64']
                user_conversion_info = {
                    "impressionId": safe_int(row.get('impressionId')),
                    "isDpa": safe_str(row.get('isDpa')),
                    "uniqueConversionTypes": [safe_int(x) for x in row.get('uniqueConversionTypes', [])],
                    "productKeyIds": [safe_int(x) for x in row.get('productKeys', [])],
                    "clickedProductKey": safe_int(row.get('clickedProductKey')),
                    "conversion_time_ms": safe_int(row.get('conversion_time_ms')),
                    "impressionCallbackEpochTimeMilliSec": safe_int(row.get('impressionCallbackEpochTimeMilliSec')),
                    "octOrMact": safe_str(row.get('octOrMact')),
                    "advertiserId": safe_int(row.get('advertiserId')),
                    "accountId": safe_int(row.get('accountId'))
                }

                batch_data[user_id].append(user_conversion_info)
            
            # Merge all the batch_data for each user_id into one write request
            batch_data_merged = defaultdict(list[dict]) # user_id -> list of user_conversion_info dict
            for user_id, data in batch_data.items():
                # Get existing conversions and append the new one
                existing_conversions = batch_user_existing_conversions.get(user_id, [])
                combined_conversions = existing_conversions + data   
                # Dedup the data by impressionId
                deduped_data = deduplicate_by_impression(combined_conversions)
                # Truncate the data to 100 records
                batch_data_merged[user_id] = deduped_data[:USER_CONVERSION_HISTORY_SIZE] # Adjust this number to change max user conversion history size
            
            # Convert batch_data_merged to payload
            batch_data_merged_payload: List[Tuple[int, Dict[str, Any]]] = [] 
            for user_id, data in batch_data_merged.items():
                payload = {"userConversionHistoryInfoList": data}
                batch_data_merged_payload.append((user_id, payload))

            # Write this batch
            success_count, batch_total, failed_records = await writer.write_conversions_batch(batch_data_merged_payload)
            
            total_success += success_count
            total_processed += batch_total
            all_failed_records.extend(failed_records)

            batch_time = time.time() - batch_start_time
            qps = len(batch_data) / batch_time if batch_time > 0 else 0

            logger.info(f"Batch completed: {success_count}/{batch_total} successful, "
                        f"QPS: {qps:.2f}, Time: {batch_time:.2f}s, Read Time: {read_end_time - read_start_time:.2f}s")

            # Small delay between batches to be respectful to the server
            if i + batch_size < len(filtered_df):
                await asyncio.sleep(REQUEST_DELAY)

    # Write failed records to file
    if all_failed_records:
        logging.warning(f"Writing {len(all_failed_records)} failed records to {failed_impressions_file}")
        with open(failed_impressions_file, 'a') as f:
            for user_id, error_msg in all_failed_records:
                f.write(f"{user_id}: {error_msg}\n")

    total_time = time.time() - start_time
    overall_qps = total_processed / total_time if total_time > 0 else 0

    logging.info(f"Async processing completed: {total_success}/{total_processed} successful")
    logging.info(f"Total time: {total_time:.2f}s, Overall QPS: {overall_qps:.2f}")
    logging.info(f"Failed records written to: {failed_impressions_file}")

    return total_success, total_processed

def safe_int(val):
    """Convert to int; if None or NA, return -1"""
    import pandas as pd
    import numpy as np

    if val is None or pd.isna(val):
        return -1
    return int(val)

def safe_str(val):
    """Convert to str; if None or NA, return empty string"""
    import pandas as pd
    if val is None or pd.isna(val):
        return ""
    return str(val)


# Configure logging at the start of the script
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Get current timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/conversion_processing_{timestamp}.log'

    # Clear any existing handlers to avoid duplicates (important for notebooks or re-runs)
    logging.getLogger().handlers.clear()

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),  # Log to file
            logging.StreamHandler(sys.stdout)   # Log to console (use sys.stdout explicitly)
        ]
    )

    # Ensure the logger for this module is properly configured
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully")
    return logger

# Initialize logger
logger = setup_logging()

# Run async processing
if __name__ == "__main__":
    print("filtered_df size: ", len(filtered_df))
    start_time = time.time()
    success, total = asyncio.run(process_conversions_async(
        filtered_df,
        batch_size=BATCH_SIZE,      # Process X records at a time
        max_concurrency=MAX_CONCURRENCY,   # Allow up to X concurrent requests
    ))
    end_time = time.time()
    
    logger.info(f"Processing completed: {success}/{total} successful, time: {end_time - start_time:.2f}s")
    print(f"Processing completed: {success}/{total} successful, time: {end_time - start_time:.2f}s")
