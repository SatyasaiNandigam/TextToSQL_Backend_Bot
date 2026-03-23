# =============================================================================
# schema_rag_pipeline.py
#
# Text-to-SQL Schema RAG Pipeline
#
# Pipeline stages:
#   1. schema_extractor()   — pulls schema + FK graph from Postgres
#   2. build_documents()    — enriches each table into an embeddable document
#   3. build_vector_store() — embeds with bge-small-en-v1.5 into ChromaDB
#   4. smart_retrieve()     — vector retrieval + keyword re-ranking
#   5. fk_expand()          — deterministic FK graph expansion (no LLM)
#   6. build_schema_context() — formats final context for SQL generation LLM
#
# Key fixes over previous version:
#   - FK expansion added as a post-retrieval step (solves order_items →
#     product_variants → products chain misses)
#   - Upgraded embedding model: bge-small-en-v1.5 (better asymmetric retrieval)
#   - k_return raised to 6 (multi-join queries need 5-6 tables)
#   - Removed fragile ALWAYS_FETCH_IF_MISSING keyword patch
#   - Noise penalties kept but rebalanced now that base retrieval is stronger
#   - retrieve_and_expand() is the single function to call from your chatbot
# =============================================================================

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(url=os.getenv("DATABASE_URL"), echo=False)

_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.normpath(os.path.join(_THIS_DIR, "chroma_db"))

# Upgraded model: bge-small-en-v1.5 is purpose-built for asymmetric retrieval
# (short query → long document) which is exactly the Text-to-SQL use case.
# Same size/speed as all-MiniLM-L6-v2 but significantly better recall.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

COLLECTION_NAME = "ecommerce_schema"

# Global embeddings instance to avoid reloading on every request
embeddings = None

def init_embeddings():
    """Initialize the embedding model once at startup."""
    global embeddings
    if embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


# =============================================================================
# TABLE METADATA
# Business descriptions and sample queries used to enrich embeddings.
# Good descriptions are the single highest-leverage thing you can do for RAG
# retrieval quality — the LLM matching happens here.
# =============================================================================

TABLE_DESCRIPTIONS: dict[str, str] = {
    "users": (
        "Stores all registered customer accounts including profile info, contact details, "
        "email verification status, and account status (active/inactive/banned). "
        "Contains city field — join with orders to get city-level or region-level analysis."
    ),
    "user_roles": (
        "Defines available roles in the system: super_admin, admin, customer, seller, support_agent."
    ),
    "user_sessions": (
        "Tracks login sessions per user including device, IP address, token, and expiry time."
    ),
    "user_role_mapping": (
        "Maps users to their assigned roles. A user can have multiple roles (e.g. customer + seller)."
    ),
    "categories": (
        "Product categories with support for parent-child hierarchy (e.g. Electronics → Mobile Phones). "
        "Self-referencing via parent_category_id. Join with products to filter by category name."
    ),
    "brands": (
        "Brand master data including country of origin and product segment. "
        "Use only when the query specifically asks about brand names, brand performance, or filtering by brand."
    ),
    "attributes": (
        "Defines product attributes like Color, Size, Storage, RAM used to differentiate variants."
    ),
    "attribute_values": (
        "Stores specific values for each attribute (e.g. Color → Red, Blue; Size → S, M, L)."
    ),
    "products": (
        "Core product catalog with name, description, brand, category, base price, and status. "
        "Join with product_variants to get stock levels and identify low stock products. "
        "Join with order_items and orders to calculate revenue per product in a date range."
    ),
    "product_variants": (
        "Each product can have multiple variants (SKUs) with distinct price, stock quantity, and weight. "
        "Use stock_qty to identify low stock (stock_qty < 10) or out of stock (stock_qty = 0) variants. "
        "Bridge table between products and order_items, "
        "Required for product-level revenue queries (e.g., revenue by product or variant), Not required for overall revenue calculations."
        "Join with products to get product names and with order_items to calculate revenue per variant."
    ),
    "variant_attributes": (
        "Links each product variant to its specific attribute values (e.g. Variant 5 = Color:Black + Size:M)."
    ),
    "product_images": (
        "Product and variant images stored as CDN URLs. One image is marked as primary per product."
    ),
    "inventory_log": (
        "Historical audit trail of stock quantity changes per variant. "
        "Records reasons like restock, sale, return, damage write-off, and correction. "
        "Do NOT use this table to find current stock levels — use product_variants.stock_qty instead. "
        "Do NOT use for revenue calculations — use order_items instead."
    ),
    "addresses": (
        "Delivery addresses per user. Each user can have multiple addresses; one is marked as default."
    ),
    "orders": (
        "Customer orders with full financial breakdown: subtotal, discount, GST tax (18%), "
        "shipping fee, grand total. Join with users table to get city-level or region-level order analysis. "
        "Use grand_total for average order value (AOV) calculations. "
        "Use created_at to filter by date range (e.g. last 30 days, last 6 months)."
    ),
    "order_items": (
        "Individual line items within an order, each referencing a product variant with quantity. "
        "Use this table ONLY when the query requires product-level analysis (e.g., revenue by product, category, or variant), "
        "Do NOT join this table when calculating overall order revenue, "
        "For total business revenue or monthly revenue, use SUM(orders.grand_total) from the orders table."
        "Join with orders to apply date filters, join with product_variants → products to get product names."
    ),
    "order_status_history": (
        "Full status trail for each order: pending → confirmed → processing → shipped → delivered/cancelled/returned."
    ),
    "payments": (
        "Payment records per order including method (UPI/Card/COD/Wallet), gateway, transaction ID, and status. "
        "Join with orders and users to analyse payment method preference by city."
    ),
    "refunds": (
        "Refund records linked to payments and orders for returned or partially refunded orders."
    ),
    "shipments": (
        "Shipment details per order including carrier (Delhivery, Blue Dart etc.), "
        "tracking number, and delivery timestamps."
    ),
    "shipment_tracking": (
        "Granular event-level tracking log per shipment (picked up → in transit → out for delivery → delivered)."
    ),
    "reviews": (
        "Customer product reviews with star rating (1-5), title, body, verified purchase flag, "
        "and helpful vote count. Join with products to get average rating per product."
    ),
    "review_images": "Photos attached to customer reviews.",
    "review_votes": (
        "Tracks which users voted a review as helpful or not helpful. One vote per user per review."
    ),
}


SAMPLE_QUERIES: dict[str, list[str]] = {
    "users": [
        "Find all active users",
        "Count users by city",
        "Get users who registered in the last 30 days",
        "Find users with unverified emails",
        "Which cities have the most customers",
        "Average order value by city",
        "Most popular payment method by city",
        "Revenue breakdown by customer city",
        "Find users from Mumbai or Delhi",
        "Join users with orders to get city-level sales data",
        "List customer names and emails",
        "Identify customers who purchased from multiple categories",
        "Find customers who placed more than 5 orders",
        "Total number of orders placed by each customer",
        "Customers who bought from both Electronics and Fashion",
        "Customers who bought items from the Electronics category",
        "Customers who placed orders in 2025 but not in 2026",
        "Find inactive customers who stopped ordering",
        "Customers who placed at least 3 orders in a year",
    ],
    "orders": [
        "Get total revenue by month",
        "Find all delivered orders for a specific user",
        "Count orders by status",
        "Calculate average order value",
        "Find orders above a certain grand total",
        "Which cities have the highest average order value",
        "Top 3 cities by total revenue or AOV",
        "Average order value grouped by city",
        "Order count and revenue per city",
        "Filter orders placed in the last 30 days by created_at",
        "Revenue from orders in the last 6 months",
        "Date filter on orders using created_at timestamp",
        "Join orders with order_items to get revenue within a date range",
        "Find customers who have spent more than 1000 in total",
        "Total amount spent per customer across all orders",
        "Customers with total spend above a threshold",
        "High value customers who spent more than a certain amount",
        "Count total number of orders placed by each customer",
        "Total orders per customer",
    ],
    "categories": [
        "List all product categories",
        "Find all subcategories under Electronics",
        "Products belonging to a specific category",
        "Customers who purchased from both Electronics and Home Decor",
        "Customers who bought items from multiple categories",
        "Filter products by category name like Electronics or Fashion",
        "Which category has the most sales",
        "Cross-category purchase analysis",
        "Identify customers who shopped across different categories",
        "Category-wise revenue breakdown",
        "Find category_id for Electronics or Home Decor",
    ],
    "products": [
        "List all active products in a category",
        "Find products by brand",
        "Get products sorted by base price",
        "Search products by name",
        "Find low stock products and their revenue",
        "Top 5 products by revenue in the last 30 days",
        "Products with current stock below threshold",
        "Revenue generated from low stock or out of stock products",
        "List product names with their stock levels and sales",
        "Which products are running low and how much have they earned",
        "Find products that belong to Electronics or Home Decor category",
        "Products purchased by customers across multiple categories",
        "Join products with categories to filter by category name",
        "Products with high ratings and most reviews",
        "Average review rating per product",
    ],
    "product_variants": [
        "Find variants with low stock quantity",
        "Get current stock level for a product",
        "Find out of stock variants (stock_qty = 0)",
        "Low stock products with less than 10 units remaining",
        "Calculate total revenue per variant from order items",
        "Top selling variants by revenue in last 30 days",
        "Stock level and revenue for each product variant",
        "Which variants are low stock and still generating revenue",
        "Bridge between products and order_items for revenue calculations",
    ],
    "order_items": [
        "Find best selling products by quantity",
        "Calculate revenue per product or variant",
        "Get all items in a specific order",
        "Total revenue generated by a product in last 30 days",
        "Revenue from low stock products",
        "Top 5 products by total sales revenue",
        "Sales quantity and revenue per variant in a date range",
        "Join order_items with orders to filter by date and sum revenue",
        "Revenue per product for orders placed in the last 30 days",
        "Sum of total_price per product within a time window",
        "Which customers purchased items from a specific category",
        "Link orders to products via order_items and product_variants",
        "Find all products a customer has purchased",
        "Items purchased by a customer across different categories",
        "Product-level revenue in the last 90 days",
        "Calculate total revenue at product level not order level",
        "Which products generated most revenue in a time period",
    ],
    "reviews": [
        "Get average rating per product",
        "Find all 5-star reviews",
        "List reviews for a specific product",
        "Find verified purchase reviews only",
        "Find customers who left a low rating review",
        "Get the body text of the lowest rated review per customer",
        "Customers with rating 1 or 2 star reviews",
        "Negative reviews with rating 2 or lower",
        "Review text and rating for unhappy customers",
        "Products with best ratings",
        "Most helpful reviews for a product",
    ],
    "payments": [
        "Count payments by method (UPI vs Card vs COD)",
        "Find failed payments",
        "Calculate total revenue from successful payments",
        "Get payment details for a specific order",
        "Most popular payment method by city",
        "Payment method preference per city or region",
        "Which cities prefer UPI over credit card",
    ],
    "shipments": [
        "Find orders that are still in transit",
        "Get average delivery time",
        "List shipments by carrier",
        "Find undelivered shipments older than 7 days",
        "Current delivery status of an order",
    ],
    "shipment_tracking": [
        "Get full tracking history for a shipment",
        "Find shipments currently out for delivery",
        "Tracking events for a specific order",
        "Current location and status of a package",
        "Find all shipments at a particular hub",
    ],
    "inventory_log": [
        "Audit trail for all stock movements of a variant",
        "Which variants had damage write-offs or corrections",
        "History of restock events for a specific SKU",
        "How many times was a variant restocked",
        "Stock adjustment history grouped by reason",
    ],
    "addresses": [
        "Find default delivery address for a user",
        "List all addresses in a specific state or pincode",
        "Users with addresses in Maharashtra",
    ],
    "refunds": [
        "Total refund amount in the last 30 days",
        "Orders that were refunded",
        "Find partial refunds",
        "Refund rate per payment method",
    ],
}


# =============================================================================
# KEYWORD SIGNALS
# Used for re-ranking after vector retrieval.
# Purpose: boost tables with keyword evidence, penalise noise tables with none.
# =============================================================================

TABLE_KEYWORD_SIGNALS: dict[str, list[str]] = {
    "orders": [
        "order", "revenue", "sales", "last 30", "30 days", "90 days", "6 months",
        "date", "month", "period", "placed", "grand total", "aov", "average order",
        "average order value", "time", "generated", "calculate", "last", "days",
        "spent", "spend", "total spend", "how much", "more than", "greater than",
        "above", "threshold", "number of orders", "total orders", "count orders",
        "inactive", "2025", "2026", "year", "highest", "average value",
    ],
    "users": [
        "user", "customer", "city", "region", "who", "account", "registered",
        "name", "email", "contact", "profile", "find customers", "customer name",
        "customer email", "names and emails", "identify customers", "list customers",
        "customers who", "customers that", "each city", "by city", "per city",
        "cities", "in each",
    ],
    "categories": [
        "category", "categories", "electronics", "home decor", "home & kitchen",
        "fashion", "sports", "beauty", "subcategory", "department",
        "both categories", "multiple categories", "purchased from", "bought from",
        "cross-category", "category name",
    ],
    "products": [
        "product", "item", "name", "catalog", "listing", "low stock", "stock",
        "electronics", "home decor", "fashion", "product category",
        "average review rating", "high rating products", "best rated",
    ],
    "product_variants": [
        "variant", "sku", "stock", "stock_qty", "low stock", "quantity",
        "out of stock", "stock level",
    ],
    "order_items": [
        "revenue", "sales", "total_price", "sold", "line item", "product revenue",
        "purchased", "bought", "items purchased", "items from", "category purchase",
        "product-level revenue", "revenue per product", "revenue generated",
        "total revenue generated", "90 days", "last 90", "calculate revenue",
    ],
    "payments": [
        "payment", "method", "upi", "card", "cod", "gateway", "paid", "transaction",
    ],
    "shipments": [
        "ship", "deliver", "carrier", "track", "transit", "courier",
        "shipping status", "delivery status", "dispatched", "in transit",
    ],
    "shipment_tracking": [
        "tracking event", "hub", "location update", "tracking history",
        "current location", "out for delivery",
    ],
    "reviews": [
        "review", "rating", "star", "feedback", "helpful", "low rating",
        "bad review", "negative review", "lowest rating", "review body",
        "review text", "unhappy customer", "dissatisfied", "average rating",
        "high rated", "best rated", "most reviews",
    ],
    "addresses": [
        "address", "pincode", "location", "state", "delivery address",
    ],
    "inventory_log": [
        "restock", "write-off", "write off", "adjustment", "audit log",
        "stock history", "correction", "stock movement", "stock change",
    ],
    "brands": ["brand", "manufacturer", "maker"],
    "refunds": ["refund", "return", "chargeback", "reversed"],
}

# Tables that should only appear when explicitly asked about
HARD_NOISE = {
    "inventory_log", "shipment_tracking", "user_sessions", "user_role_mapping",
    "review_images", "review_votes", "variant_attributes", "product_images",
    "attribute_values", "attributes",
}

# Tables relevant only for specific query types
SOFT_NOISE = {
    "brands", "refunds", "addresses", "order_status_history", "user_roles",
}


# =============================================================================
# SYSTEM COLUMNS
# Postgres information_schema columns that sometimes leak into results.
# =============================================================================

SYSTEM_COLUMNS = {
    "udt_catalog", "udt_schema", "udt_name", "attribute_name", "ordinal_position",
    "attribute_default", "is_nullable", "data_type", "character_maximum_length",
    "character_octet_length", "character_set_catalog", "character_set_schema",
    "character_set_name", "collation_catalog", "collation_schema", "collation_name",
    "numeric_precision", "numeric_precision_radix", "numeric_scale", "datetime_precision",
    "interval_type", "interval_precision", "attribute_udt_catalog", "attribute_udt_schema",
    "attribute_udt_name", "scope_catalog", "scope_schema", "scope_name",
    "maximum_cardinality", "dtd_identifier", "is_derived_reference_attribute",
}


# FK join summary with cardinality
# Cardinality rules:
#   junction tables  → many-to-many (each FK side is many-to-one)
#   all other tables → many-to-one  (child row → one parent row)
FK_CARDINALITY: dict[str, str] = {
    "user_role_mapping":    "many-to-many (junction table)",
    "variant_attributes":   "many-to-many (junction table)",
    "order_items":          "many-to-one",
    "orders":               "many-to-one",
    "addresses":            "many-to-one",
    "product_variants":     "many-to-one",
    "products":             "many-to-one",
    "reviews":              "many-to-one",
    "payments":             "many-to-one",
    "shipments":            "many-to-one",
    "refunds":              "many-to-one",
    "inventory_log":        "many-to-one",
    "shipment_tracking":    "many-to-one",
    "user_sessions":        "many-to-one",
    "order_status_history": "many-to-one",
    "product_images":       "many-to-one",
    "attribute_values":     "many-to-one",
    "categories":           "many-to-one",
}



# =============================================================================
# STAGE 1 — SCHEMA EXTRACTOR
# Pulls table columns, primary keys, and foreign keys from Postgres.
# The FK data is used later by fk_expand() to walk the join chain.
# =============================================================================

def schema_extractor() -> dict:
    """
    Extracts clean schema from all public tables in Postgres.
    Uses the module-level `engine` created from DATABASE_URL in .env.

    Returns:
        {
            "table_name": {
                "columns":      [(col, dtype, nullable), ...],
                "primary_keys": ["id", ...],
                "foreign_keys": [{"column": "variant_id", "references": "product_variants.id"}, ...]
            }
        }
    """
    schema: dict = {}

    with engine.connect() as conn:
        tables = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)).fetchall()

        for (table,) in tables:
            raw_columns = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = :table
                ORDER BY ordinal_position
            """), {"table": table}).fetchall()

            # Filter out any leaked information_schema system column names
            columns = [
                (col, dtype, nullable)
                for col, dtype, nullable in raw_columns
                if col not in SYSTEM_COLUMNS
            ]

            primary_keys = conn.execute(text("""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema    = kcu.table_schema
                WHERE tc.table_schema = 'public'
                  AND tc.table_name   = :table
                  AND tc.constraint_type = 'PRIMARY KEY'
            """), {"table": table}).fetchall()

            foreign_keys = conn.execute(text("""
                SELECT
                    kcu.column_name,
                    ccu.table_name  AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                   AND tc.table_schema    = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                   AND ccu.table_schema    = tc.table_schema
                WHERE tc.table_schema  = 'public'
                  AND tc.table_name    = :table
                  AND tc.constraint_type = 'FOREIGN KEY'
            """), {"table": table}).fetchall()

            schema[table] = {
                "columns":      columns,
                "primary_keys": [pk[0] for pk in primary_keys],
                "foreign_keys": [
                    {"column": fk[0], "references": f"{fk[1]}.{fk[2]}"}
                    for fk in foreign_keys
                ],
            }

    return schema


# =============================================================================
# STAGE 2 — DOCUMENT BUILDER
# Converts raw schema into enriched LangChain Documents for embedding.
# Rich content here directly improves retrieval quality — don't cut corners.
# =============================================================================

def build_documents(schema: dict) -> list[Document]:
    """
    Builds one Document per table with:
      - Business description (bridges natural language ↔ schema vocabulary)
      - All columns with types, nullability, PK/FK annotations
      - FK join hints
      - Sample natural language queries (most important for retrieval)

    The sample queries are the main reason RAG finds the right tables —
    the embedding model matches user queries against these, not raw DDL.
    """
    documents: list[Document] = []

    for table, info in schema.items():
        pk_list  = info["primary_keys"]
        fk_list  = info["foreign_keys"]
        col_list = info["columns"]

        pk_set     = set(pk_list)
        fk_col_set = {fk["column"] for fk in fk_list}

        # Column block with PK/FK annotations
        col_lines = []
        for col, dtype, nullable in col_list:
            tags = []
            if col in pk_set:
                tags.append("PRIMARY KEY")
            if col in fk_col_set:
                fk_ref = next(fk["references"] for fk in fk_list if fk["column"] == col)
                tags.append(f"FK → {fk_ref}")
            null_tag = "NOT NULL" if nullable == "NO" else "nullable"
            tag_str  = f"  [{', '.join(tags)}]" if tags else ""
            col_lines.append(f"  - {col} ({dtype}, {null_tag}){tag_str}")

        
        table_cardinality = FK_CARDINALITY.get(table, "many-to-one")

        fk_lines = []
        for fk in fk_list:
            ref        = fk["references"]
            ref_table  = ref.split(".")[0]
            if table_cardinality == "many-to-many (junction table)":
                cardinality_note = "many-to-many via this junction table"
            else:
                cardinality_note = f"many {table} → one {ref_table}"
            fk_lines.append(
                f"  - {table}.{fk['column']} → {ref}  [{cardinality_note}]"
            )

        # Sample queries
        queries     = SAMPLE_QUERIES.get(table, [])
        query_lines = [f"  - {q}" for q in queries]

        description = TABLE_DESCRIPTIONS.get(table, f"Table storing {table} data.")

        content_parts = [
            f"Table: {table}",
            f"Purpose: {description}",
            "",
            f"Columns ({len(col_list)} total):",
            *col_lines,
        ]

        if fk_lines:
            content_parts += ["", "Relationships (Foreign Keys):"] + fk_lines

        if query_lines:
            content_parts += ["", "Example queries this table can answer:"] + query_lines

        page_content = "\n".join(content_parts)

        metadata = {
            "table_name":   table,
            "primary_keys": ", ".join(pk_list),
            "foreign_keys": json.dumps([fk["references"] for fk in fk_list]),
            "column_names": ", ".join(col for col, _, _ in col_list),
            "column_count": len(col_list),
            "has_fk":       len(fk_list) > 0,
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"  Built {len(documents)} documents from {len(schema)} tables")
    return documents


# =============================================================================
# STAGE 3 — VECTOR STORE
# Embeds documents into ChromaDB using bge-small-en-v1.5.
#
# Why bge-small-en-v1.5 over all-MiniLM-L6-v2:
#   - BAAI/bge models are trained for asymmetric retrieval (short query →
#     long passage), exactly our use case
#   - Same model size and inference speed
#   - Meaningfully better recall on schema-vs-natural-language matching
# =============================================================================

def build_vector_store(documents: list[Document], persist_dir: str = CHROMA_DB_PATH):
    """Embeds documents and persists to ChromaDB."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"  Writing to ChromaDB at {persist_dir} ...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Clear existing docs to avoid duplicates on re-runs
    try:
        existing = vector_store.get()
        if existing["ids"]:
            vector_store.delete(ids=existing["ids"])
            print(f"  Cleared {len(existing['ids'])} existing docs")
    except Exception:
        pass

    vector_store.add_documents(documents)
    print(f"  ✓ {len(documents)} documents embedded and stored\n")
    return vector_store


# =============================================================================
# STAGE 4 — SMART RETRIEVER
# Vector retrieval + keyword signal re-ranking.
# Returns top k_return tables sorted by combined (vector + keyword) score.
#
# Note: this does NOT yet apply FK expansion. Call fk_expand() after this.
# =============================================================================

def smart_retrieve(
    query:       str,
    schema:      dict,
    persist_dir: str = CHROMA_DB_PATH,
    k_fetch:     int = 14,
    k_return:    int = 6,
) -> list[Document]:
    """
    Stage 4: Retrieve and re-rank candidate tables from ChromaDB.

    Args:
        query:       Natural language user query
        schema:      Output of schema_extractor() — needed by fk_expand() later
        persist_dir: ChromaDB persist directory
        k_fetch:     Broad retrieval count (cast a wide net)
        k_return:    Tables to return after re-ranking (pre FK-expansion)

    Returns:
        List of top k_return Document objects, sorted by relevance.
    """
    from langchain_chroma import Chroma

    # Use global embeddings instance
    if embeddings is None:
        raise RuntimeError("Embeddings not initialized. Call init_embeddings() at startup.")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Broad vector retrieval
    candidates  = vector_store.similarity_search_with_score(query, k=k_fetch)
    query_lower = query.lower()

    # Re-rank: vector similarity + keyword signal bonus + noise penalties
    scored = []
    for doc, vector_score in candidates:
        table        = doc.metadata["table_name"]
        signals      = TABLE_KEYWORD_SIGNALS.get(table, [])
        keyword_hits = sum(1 for kw in signals if kw in query_lower)

        # ChromaDB returns L2 distance — lower = more similar
        # Convert to similarity: higher = better
        base_score    = 1 - vector_score
        keyword_bonus = keyword_hits * 0.25
        combined      = base_score + keyword_bonus

        # Penalise noise tables when they have zero keyword signal
        if keyword_hits == 0:
            if table in HARD_NOISE:
                combined *= 0.60   # strong suppression
            elif table in SOFT_NOISE:
                combined *= 0.55   # moderate suppression

        scored.append((doc, combined, vector_score, keyword_hits))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = [doc for doc, _, _, _ in scored[:k_return]]

    # Debug table
    print(f"\n[smart_retrieve] Query: '{query}'")
    print(f"  {'Table':<25} {'VectorSim':>10} {'KWHits':>7} {'Combined':>10}  {'Selected':>8}")
    print(f"  {'-'*65}")
    for doc, combined, vscore, hits in scored:
        selected = "✅" if doc in results else ""
        print(f"  {doc.metadata['table_name']:<25} {1-vscore:>10.4f} {hits:>7} {combined:>10.4f}  {selected:>8}")

    return results


# =============================================================================
# STAGE 5 — FK EXPANSION
# Deterministic graph walk — no LLM, no ambiguity.
#
# This solves the core problem: RAG retrieves order_items + products but
# misses product_variants. FK expansion walks the chain and adds it.
#
#   order_items -[variant_id]-> product_variants -[product_id]-> products
#
# The schema dict already contains foreign_keys extracted in Stage 1, so
# no extra DB calls or config needed.
# =============================================================================

def fk_expand(
    retrieved_docs: list[Document],
    schema:         dict,
    depth:          int = 2,
) -> set[str]:
    """
    Stage 5: Walk the FK graph outward from retrieved tables.

    For each retrieved table, follow its foreign keys (and their foreign keys)
    up to `depth` hops. Any table discovered along the chain is added to the
    final set — even if the vector retriever never found it.

    Args:
        retrieved_docs: Documents returned by smart_retrieve()
        schema:         Output of schema_extractor()
        depth:          How many FK hops to follow (2 covers most JOIN chains)

    Returns:
        Set of all table names needed (retrieved + FK-expanded).

    Example with depth=2:
        Retrieved:  {order_items, products}
        Hop 1:      order_items → product_variants  (via variant_id FK)
                    order_items → orders             (via order_id FK)
        Hop 2:      product_variants → products      (already in set)
        Final:      {order_items, products, product_variants, orders}
    """
    expanded: set[str] = {doc.metadata["table_name"] for doc in retrieved_docs}
    queue = list(expanded)

    for hop in range(depth):
        next_queue: list[str] = []
        for table in queue:
            fk_list = schema.get(table, {}).get("foreign_keys", [])
            for fk in fk_list:
                # "product_variants.id" → "product_variants"
                ref_table = fk["references"].split(".")[0]
                if ref_table not in expanded:
                    expanded.add(ref_table)
                    next_queue.append(ref_table)
                    print(f"  [fk_expand] hop {hop+1}: {table} → {ref_table}  (via {fk['column']})")
        queue = next_queue

    return expanded


# =============================================================================
# STAGE 6 — SCHEMA CONTEXT BUILDER
# Formats the final expanded table set into a clean context string for the LLM.
# =============================================================================

def build_schema_context(table_names: set[str], schema: dict) -> str:
    """
    Stage 6: Build a clean schema context string from the expanded table set.

    Produces a CREATE TABLE-style block per table with FK annotations so the
    LLM knows exactly which columns to use for JOINs.

    Args:
        table_names: Set of table names (output of fk_expand)
        schema:      Output of schema_extractor()

    Returns:
        Multi-line string ready to paste into the SQL generation prompt.
    """
    lines = []
    for table in sorted(table_names):
        info = schema.get(table)
        if not info:
            continue

        description = TABLE_DESCRIPTIONS.get(table, "")
        lines.append(f"-- {description}" if description else "")
        lines.append(f"Table: {table}")

        col_list = info["columns"]
        fk_list  = info["foreign_keys"]
        pk_list  = info["primary_keys"]
        pk_set   = set(pk_list)
        fk_map   = {fk["column"]: fk["references"] for fk in fk_list}

        for col, dtype, nullable in col_list:
            annotations = []
            if col in pk_set:
                annotations.append("PK")
            if col in fk_map:
                annotations.append(f"FK → {fk_map[col]}")
            null_str  = "" if nullable == "NO" else " nullable"
            annot_str = f"  -- {', '.join(annotations)}" if annotations else ""
            lines.append(f"  {col} {dtype}{null_str}{annot_str}")

        if fk_list:
            lines.append("  -- JOIN hints:")
            for fk in fk_list:
                ref_table = fk['references'].split(".")[0]
                cardinality = FK_CARDINALITY.get(table, "many-to-one")
                if cardinality == "many-to-many (junction table)":
                    note = "many-to-many via junction"
                else:
                    note = f"many {table} → one {ref_table}"
                lines.append(f"  --   {table}.{fk['column']} → {fk['references']}  [{note}]")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN ENTRY POINT
# Call this single function from your chatbot / query handler.
# =============================================================================

def retrieve_and_expand(
    query:       str,
    schema:      dict,
    persist_dir: str = CHROMA_DB_PATH,
    k_fetch:     int = 14,
    k_return:    int = 6,
    fk_depth:    int = 2,
) -> tuple[str, set[str]]:
    """
    Full pipeline: retrieve → FK expand → build context.

    This is the single function to call from your chatbot.

    Args:
        query:       Natural language user question
        schema:      Output of schema_extractor() — call once and cache
        persist_dir: ChromaDB persist directory
        k_fetch:     Broad retrieval count for vector store
        k_return:    Tables returned after re-ranking (before FK expansion)
        fk_depth:    FK graph traversal depth (2 covers most chains)

    Returns:
        (schema_context_string, set_of_table_names)

    Usage:
        schema = schema_extractor(engine)          # once at startup
        context, tables = retrieve_and_expand(user_query, schema)
        sql = your_llm.generate(prompt(context, user_query))
    """
    print(f"\n{'='*65}")
    print(f"Query: {query}")
    print(f"{'='*65}")

    # Stage 4: Vector retrieval + keyword re-ranking
    retrieved_docs = smart_retrieve(query, schema, persist_dir, k_fetch, k_return)
    retrieved_names = {d.metadata["table_name"] for d in retrieved_docs}
    print(f"\n[retrieve_and_expand] RAG retrieved: {sorted(retrieved_names)}")

    # Stage 5: FK graph expansion
    expanded_names = fk_expand(retrieved_docs, schema, depth=fk_depth)
    new_tables = expanded_names - retrieved_names
    if new_tables:
        print(f"[retrieve_and_expand] FK expansion added: {sorted(new_tables)}")
    print(f"[retrieve_and_expand] Final tables: {sorted(expanded_names)}")

    # Stage 6: Build context string
    context = build_schema_context(expanded_names, schema)

    return context, expanded_names


# =============================================================================
# MAIN — runs the full build + test when executed directly
# =============================================================================

if __name__ == "__main__":
    print("\n── Step 1: Extracting schema from Postgres ──")
    schema = schema_extractor()
    print(f"  Found {len(schema)} tables: {', '.join(sorted(schema.keys()))}\n")

    print("── Step 2: Building RAG documents ──")
    documents = build_documents(schema)

    print("\n── Step 3: Embedding into ChromaDB ──")
    vector_store = build_vector_store(documents)

    print("\n── Step 4: Test retrieve_and_expand ──")

    test_queries = [
        "Which 3 cities have the highest average order value, and what is the most popular payment method in each of those cities?",
        "For products that are currently low stock, calculate the total revenue we have generated from them in the last 30 days. List the top 5 such products by revenue and their current stock level.",
        "What is the total revenue by month for the last 6 months?",
        "Which products have the best ratings and most reviews?",
        "Find all shipments that are still in transit and their current tracking status.",
        "List the names and emails of customers who have spent more than 5000 in total across all their orders.",
    ]

    for query in test_queries:
        context, tables = retrieve_and_expand(query, schema)
        print(f"\n  → Final tables passed to LLM: {sorted(tables)}")
        print(f"  → Context length: {len(context)} chars\n")

    print("\n✅ Schema RAG pipeline ready!\n")
    print("─── Usage in your chatbot ───────────────────────────────────")
    print("  from schema_rag_pipeline import schema_extractor, build_vector_store")
    print("  from schema_rag_pipeline import build_documents, retrieve_and_expand")
    print()
    print("  # At startup (once):")
    print("  schema = schema_extractor()            # engine is picked up from .env automatically")
    print("  docs   = build_documents(schema)")
    print("  build_vector_store(docs)   # only needed on first run or schema change")
    print()
    print("  # Per user query:")
    print("  context, tables = retrieve_and_expand(user_query, schema)")
    print("  sql = llm.generate(f'Schema:\\n{context}\\n\\nQuestion: {user_query}\\nSQL:')")