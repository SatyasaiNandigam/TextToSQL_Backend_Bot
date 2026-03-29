"""
Ground truth dataset for schema retriever evaluation.

30 queries across 6 categories, grounded in the real ecommerce DB schema.
Each entry specifies which tables MUST appear in the FK-expanded retrieval set (required_tables)
and which are acceptable collateral from FK expansion (optional_tables).

FK join graph (key paths):
  orders → order_items → product_variants → products → brands / categories
  orders → payments → refunds
  orders → shipments
  reviews → products / users / orders
  users → orders / reviews / addresses
"""

from dataclasses import dataclass, field


@dataclass
class GroundTruthEntry:
    query_id: str
    category: str
    query: str
    required_tables: set[str]
    optional_tables: set[str]
    reference: str  # gold NL description used as Ragas reference


GROUND_TRUTH: list[GroundTruthEntry] = [
    # ─────────────────────────────────────────────────────────────────────────
    # Category 1: Order Analytics
    # ─────────────────────────────────────────────────────────────────────────
    GroundTruthEntry(
        query_id="ORD-01",
        category="Order Analytics",
        query="What are the top 5 products by total revenue?",
        required_tables={"orders", "order_items", "product_variants", "products"},
        optional_tables={"users", "categories", "brands"},
        reference=(
            "To find top products by revenue, join orders with order_items on order_id, "
            "then join order_items with product_variants on variant_id, and product_variants "
            "with products on product_id. Sum total_price from order_items grouped by product name."
        ),
    ),
    GroundTruthEntry(
        query_id="ORD-02",
        category="Order Analytics",
        query="How many orders are in each status?",
        required_tables={"orders"},
        optional_tables={"users", "payments"},
        reference=(
            "Query the orders table and group by the status column "
            "(values: pending, confirmed, processing, shipped, delivered, cancelled, returned) "
            "to count orders in each state."
        ),
    ),
    GroundTruthEntry(
        query_id="ORD-03",
        category="Order Analytics",
        query="What is the average order value per customer?",
        required_tables={"orders", "users"},
        optional_tables={"order_items", "payments"},
        reference=(
            "Join orders with users on user_id. Group by user (first_name, last_name or email) "
            "and compute AVG(grand_total) from the orders table to get average order value per customer."
        ),
    ),
    GroundTruthEntry(
        query_id="ORD-04",
        category="Order Analytics",
        query="Which orders were cancelled and what was the total revenue lost?",
        required_tables={"orders"},
        optional_tables={"users", "payments", "order_items"},
        reference=(
            "Filter the orders table where status = 'cancelled'. "
            "Sum grand_total across those rows to get total revenue lost from cancellations."
        ),
    ),
    GroundTruthEntry(
        query_id="ORD-05",
        category="Order Analytics",
        query="Show the total number of items ordered per product category",
        required_tables={"order_items", "product_variants", "products", "categories"},
        optional_tables={"orders"},
        reference=(
            "Join order_items → product_variants on variant_id, product_variants → products on product_id, "
            "products → categories on category_id. Group by category name and sum quantity from order_items."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Category 2: Product & Inventory
    # ─────────────────────────────────────────────────────────────────────────
    GroundTruthEntry(
        query_id="PRD-01",
        category="Product & Inventory",
        query="Which products have zero stock across all their variants?",
        required_tables={"products", "product_variants"},
        optional_tables={"categories", "brands"},
        reference=(
            "Join products with product_variants on product_id. Group by product and check that "
            "SUM(stock_qty) = 0 or all variants have stock_qty = 0 to find fully out-of-stock products."
        ),
    ),
    GroundTruthEntry(
        query_id="PRD-02",
        category="Product & Inventory",
        query="What are the top 10 best-selling products by quantity sold?",
        required_tables={"order_items", "product_variants", "products"},
        optional_tables={"orders", "categories"},
        reference=(
            "Join order_items with product_variants on variant_id, then product_variants with products "
            "on product_id. Group by product name and sum quantity from order_items to rank by units sold."
        ),
    ),
    GroundTruthEntry(
        query_id="PRD-03",
        category="Product & Inventory",
        query="List all active products in the Electronics category with their variant prices",
        required_tables={"products", "categories", "product_variants"},
        optional_tables={"brands"},
        reference=(
            "Join products → categories on category_id (filter category name = 'Electronics') and "
            "products → product_variants on product_id (filter product status = 'active'). "
            "Return product name, variant SKU, and price."
        ),
    ),
    GroundTruthEntry(
        query_id="PRD-04",
        category="Product & Inventory",
        query="How many product variants does each product have?",
        required_tables={"products", "product_variants"},
        optional_tables=set(),
        reference=(
            "Join products with product_variants on product_id. "
            "Group by product name and count the number of variant rows per product."
        ),
    ),
    GroundTruthEntry(
        query_id="PRD-05",
        category="Product & Inventory",
        query="Which brands have the most products and what categories do they cover?",
        required_tables={"products", "brands", "categories"},
        optional_tables=set(),
        reference=(
            "Join products → brands on brand_id and products → categories on category_id. "
            "Group by brand name, count products, and list distinct category names per brand."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Category 3: Customer Behavior
    # ─────────────────────────────────────────────────────────────────────────
    GroundTruthEntry(
        query_id="CUS-01",
        category="Customer Behavior",
        query="Who are the top 10 customers by total spend?",
        required_tables={"users", "orders"},
        optional_tables={"order_items", "payments"},
        reference=(
            "Join users with orders on user_id. Group by customer (full_name or email) and "
            "sum grand_total from orders to rank customers by total spend descending."
        ),
    ),
    GroundTruthEntry(
        query_id="CUS-02",
        category="Customer Behavior",
        query="Which customers have placed more than 3 orders?",
        required_tables={"users", "orders"},
        optional_tables=set(),
        reference=(
            "Join users with orders on user_id. Group by user and count order rows. "
            "Filter groups having COUNT(order_id) > 3."
        ),
    ),
    GroundTruthEntry(
        query_id="CUS-03",
        category="Customer Behavior",
        query="Show customers who have never placed an order",
        required_tables={"users", "orders"},
        optional_tables=set(),
        reference=(
            "LEFT JOIN users with orders on user_id. Filter where order_id IS NULL "
            "to find users who have no associated orders."
        ),
    ),
    GroundTruthEntry(
        query_id="CUS-04",
        category="Customer Behavior",
        query="What is the distribution of orders by customer city?",
        required_tables={"users", "orders"},
        optional_tables=set(),
        reference=(
            "Join orders with users on user_id. Group by users.city and count orders "
            "to see how many orders originate from each city."
        ),
    ),
    GroundTruthEntry(
        query_id="CUS-05",
        category="Customer Behavior",
        query="Which customers have both placed orders and written reviews?",
        required_tables={"users", "orders", "reviews"},
        optional_tables={"products"},
        reference=(
            "Find users who appear in both orders (via user_id) and reviews (via user_id). "
            "Use INTERSECT or JOIN users with both orders and reviews tables on user_id."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Category 4: Payment & Revenue
    # ─────────────────────────────────────────────────────────────────────────
    GroundTruthEntry(
        query_id="PAY-01",
        category="Payment & Revenue",
        query="What is the total revenue collected by each payment method?",
        required_tables={"orders", "payments"},
        optional_tables=set(),
        reference=(
            "Join payments with orders on order_id. Filter payments where status = 'success'. "
            "Group by payments.method and sum amount to get revenue per payment method."
        ),
    ),
    GroundTruthEntry(
        query_id="PAY-02",
        category="Payment & Revenue",
        query="How many payment transactions failed and what was the total failed amount?",
        required_tables={"payments"},
        optional_tables={"orders"},
        reference=(
            "Filter the payments table where status = 'failed'. "
            "Count the rows and sum amount to get the total value of failed transactions."
        ),
    ),
    GroundTruthEntry(
        query_id="PAY-03",
        category="Payment & Revenue",
        query="What is the total refund amount issued grouped by refund reason?",
        required_tables={"payments", "refunds"},
        optional_tables={"orders"},
        reference=(
            "Join refunds with payments on payment_id. Group by refunds.reason and "
            "sum refunds.amount to get total refunded per reason category."
        ),
    ),
    GroundTruthEntry(
        query_id="PAY-04",
        category="Payment & Revenue",
        query="Show month-wise revenue trend for the current year",
        required_tables={"orders"},
        optional_tables={"payments"},
        reference=(
            "Query the orders table. Extract month from created_at, filter for the current year, "
            "group by month, and sum grand_total to see monthly revenue trend."
        ),
    ),
    GroundTruthEntry(
        query_id="PAY-05",
        category="Payment & Revenue",
        query="Which orders still have a pending payment?",
        required_tables={"orders", "payments"},
        optional_tables={"users"},
        reference=(
            "Join orders with payments on order_id. Filter where payments.status = 'pending' "
            "to identify orders awaiting payment confirmation."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Category 5: Shipping & Fulfillment
    # ─────────────────────────────────────────────────────────────────────────
    GroundTruthEntry(
        query_id="SHP-01",
        category="Shipping & Fulfillment",
        query="What is the average delivery time in days from shipped to delivered?",
        required_tables={"shipments"},
        optional_tables={"orders"},
        reference=(
            "Query the shipments table where status = 'delivered'. "
            "Compute AVG(delivered_at - shipped_at) in days to get mean delivery duration."
        ),
    ),
    GroundTruthEntry(
        query_id="SHP-02",
        category="Shipping & Fulfillment",
        query="Which orders have been shipped but not yet delivered?",
        required_tables={"orders", "shipments"},
        optional_tables={"users"},
        reference=(
            "Join orders with shipments on order_id. Filter where shipments.status = 'in_transit' "
            "or orders.status = 'shipped' to find orders currently in transit."
        ),
    ),
    GroundTruthEntry(
        query_id="SHP-03",
        category="Shipping & Fulfillment",
        query="What is the total shipping charge collected by each carrier?",
        required_tables={"shipments"},
        optional_tables={"orders"},
        reference=(
            "Query the shipments table. Group by carrier and sum shipping_charge "
            "to get total charges collected per carrier."
        ),
    ),
    GroundTruthEntry(
        query_id="SHP-04",
        category="Shipping & Fulfillment",
        query="Show the count of shipments in each current status",
        required_tables={"shipments"},
        optional_tables={"orders"},
        reference=(
            "Query the shipments table. Group by status "
            "(in_transit, delivered, returned) and count rows per status."
        ),
    ),
    GroundTruthEntry(
        query_id="SHP-05",
        category="Shipping & Fulfillment",
        query="Which products have been shipped the most times?",
        required_tables={"shipments", "orders", "order_items", "product_variants", "products"},
        optional_tables=set(),
        reference=(
            "Join shipments → orders on order_id, orders → order_items on order_id, "
            "order_items → product_variants on variant_id, product_variants → products on product_id. "
            "Group by product name and count shipment rows to rank most-shipped products."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Category 6: Reviews & Ratings
    # ─────────────────────────────────────────────────────────────────────────
    GroundTruthEntry(
        query_id="REV-01",
        category="Reviews & Ratings",
        query="What is the average product rating by category?",
        required_tables={"reviews", "products", "categories"},
        optional_tables=set(),
        reference=(
            "Join reviews → products on product_id, products → categories on category_id. "
            "Group by category name and compute AVG(reviews.rating)."
        ),
    ),
    GroundTruthEntry(
        query_id="REV-02",
        category="Reviews & Ratings",
        query="Which products have received the most low ratings (1 or 2 stars)?",
        required_tables={"reviews", "products"},
        optional_tables=set(),
        reference=(
            "Join reviews with products on product_id. Filter where reviews.rating <= 2. "
            "Group by product name and count low-rating reviews descending."
        ),
    ),
    GroundTruthEntry(
        query_id="REV-03",
        category="Reviews & Ratings",
        query="How many verified purchase reviews does each product have?",
        required_tables={"reviews", "products"},
        optional_tables={"order_items", "orders"},
        reference=(
            "Join reviews with products on product_id. Filter where is_verified_purchase = true. "
            "Group by product name and count verified reviews."
        ),
    ),
    GroundTruthEntry(
        query_id="REV-04",
        category="Reviews & Ratings",
        query="What percentage of customers have written at least one review?",
        required_tables={"users", "reviews"},
        optional_tables=set(),
        reference=(
            "Count distinct user_id in reviews divided by total user count from users table, "
            "multiplied by 100 to get the percentage of reviewing customers."
        ),
    ),
    GroundTruthEntry(
        query_id="REV-05",
        category="Reviews & Ratings",
        query="What is the average rating per brand?",
        required_tables={"reviews", "products", "brands"},
        optional_tables=set(),
        reference=(
            "Join reviews → products on product_id, products → brands on brand_id. "
            "Group by brand name and compute AVG(reviews.rating) to rank brands by customer satisfaction."
        ),
    ),
]


def get_ground_truth() -> list[GroundTruthEntry]:
    return GROUND_TRUTH
