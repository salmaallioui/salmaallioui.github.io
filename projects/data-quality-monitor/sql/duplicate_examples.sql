-- Ejemplos de order_id duplicados para el día indicado (para diagnóstico en el informe).
SELECT order_id, COUNT(*) AS n
FROM orders
WHERE order_date = :day
GROUP BY order_id
HAVING COUNT(*) > 1
ORDER BY n DESC, order_id
LIMIT 5;
