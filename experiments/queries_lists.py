q_list = [
    'What is the total revenue by city in the last 30 days?',
    'How many orders were placed in each city this month?',
    'What are the top 10 products by total units sold of all time?',
    'Which 5 customers have placed the most orders in the last 6 months?',
    'What is the average order value per city in 2025?',
    'What are the top 5 products by revenue in the Electronics category in 2025?',
    'Which brands have generated more than ₹1,00,000 in total revenue across all time?',
    'How many distinct customers purchased from the Home & Kitchen category in the last 6 months?'
]

complex_q_list = [
    'Identify customers who purchased from the Electronics category in 2025 but have NOT placed any order at all in 2026.',
    'Which products were sold in 2024 but have had zero orders in 2025?',
    'List customers who placed at least one order in 2024 but never left a review on any product',
    'Which customers have a total spend above the average customer spend in 2025?',
    'Which product variants have a stock quantity below the average stock across all variants in their category?',
    'Which cities have an average order value higher than the overall average order value in the last 6 months?'
]

q_list_tables_needed = [
    'users,orders',
    'users,orders',
    'order_items, product_variants, products, orders',
    'orders,users',
    'orders,users'
]

q_list_standard_sql = [
    """
        SELECT 
            u.city,
            SUM(o.grand_total) AS total_revenue
        FROM orders o
        JOIN users u 
            ON o.user_id = u.user_id
        WHERE 
            o.created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY 
            u.city
        ORDER BY 
            total_revenue DESC;
    """,
    
    """
        SELECT 
            u.city,
            COUNT(o.order_id) AS total_orders
        FROM orders o
        JOIN users u 
            ON o.user_id = u.user_id
        WHERE 
            DATE_TRUNC('month', o.created_at) = DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 
            u.city
        ORDER BY 
            total_orders DESC;
    
    """,
    
    """
        SELECT 
            p.product_id,
            p.name,
            SUM(oi.quantity) AS total_units_sold
        FROM order_items oi
        JOIN product_variants pv 
            ON oi.variant_id = pv.variant_id
        JOIN products p 
            ON pv.product_id = p.product_id
        JOIN orders o 
            ON oi.order_id = o.order_id
        WHERE 
            o.status = 'delivered'
        GROUP BY 
            p.product_id, p.name
        ORDER BY 
            total_units_sold DESC
        LIMIT 10;
    
    """,
    
    """
        SELECT 
            u.user_id,
            u.full_name,
            COUNT(o.order_id) AS total_orders
        FROM orders o
        JOIN users u 
            ON o.user_id = u.user_id
        WHERE 
            o.created_at >= CURRENT_DATE - INTERVAL '6 months'
            AND o.status = 'delivered'  -- optional but recommended
        GROUP BY 
            u.user_id, u.full_name
        ORDER BY 
            total_orders DESC
        LIMIT 5;
    
    """,
    
    """
        SELECT 
            u.city,
            AVG(o.grand_total) AS avg_order_value
        FROM orders o
        JOIN users u 
            ON o.user_id = u.user_id
        WHERE 
            EXTRACT(YEAR FROM o.created_at) = 2025
            AND o.status = 'delivered'  -- optional but recommended
        GROUP BY 
            u.city
        ORDER BY 
            avg_order_value DESC;
    
    """ 
    
]