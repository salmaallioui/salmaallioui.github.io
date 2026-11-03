-- Perfil diario de la tabla: una fila por día con las métricas que vigilan los checks.
SELECT
    order_date                                                   AS day,
    COUNT(*)                                                     AS row_count,
    COUNT(*) - COUNT(DISTINCT order_id)                          AS duplicate_ids,
    1.0 * SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END)
        / COUNT(*)                                               AS null_rate_customer_id,
    SUM(CASE WHEN amount_eur < 0 OR amount_eur > 5000 THEN 1 ELSE 0 END)
                                                                 AS out_of_range_amount,
    ROUND(SUM(amount_eur), 2)                                    AS revenue_eur
FROM orders
WHERE order_date BETWEEN :start AND :end
GROUP BY order_date
ORDER BY order_date;
