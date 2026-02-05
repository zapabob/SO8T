import sqlite3
conn = sqlite3.connect('so8t_memory.db')
cursor = conn.cursor()
print('--- MODEL METRICS ---')
cursor.execute("SELECT metric_type, metric_value, status, timestamp FROM model_metrics")
for row in cursor.fetchall():
    print(row)
conn.close()
