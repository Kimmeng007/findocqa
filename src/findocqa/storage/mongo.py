from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from findocqa.config import settings


@lru_cache
def get_client() -> MongoClient:
    return MongoClient(settings.mongodb_uri)


def get_filings_collection() -> Collection:
    coll = get_client()[settings.mongodb_db_name]["filings"]
    coll.create_index([("doc_name", ASCENDING)], unique=True)
    coll.create_index([("company", ASCENDING)])
    coll.create_index([("fiscal_year", ASCENDING)])
    coll.create_index([("filing_type", ASCENDING)])
    return coll


def upsert_filing(doc: dict) -> None:
    get_filings_collection().replace_one({"doc_name": doc["doc_name"]}, doc, upsert=True)


def iter_filings():
    yield from get_filings_collection().find()


def get_existing_doc_names() -> set[str]:
    return {d["doc_name"] for d in get_filings_collection().find({}, {"doc_name": 1})}
