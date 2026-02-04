# -*- coding: utf-8 -*-
import sqlite3
conn = sqlite3.connect('so8t_memory.db')
cur = conn.cursor()
print('TABLES:')
for (name,) in cur.execute('select name from sqlite_master where type=\'table\' order by name'):
    print('-', name)
print('\nLATEST_DOC_LOGS:')
rows = cur.execute('select topic, length(content), created_at, updated_at from knowledge_base where source_type=\'document\' order by updated_at desc limit 20').fetchall()
for row in rows:
    print(row)
print('\nSAMPLE_LOG_SNIPPETS:')
rows = cur.execute('select topic, substr(content,1,300) from knowledge_base where source_type=\'document\' and (topic like \'%ムーンショット%\' or topic like \'%moonshot%\' or topic like \'%再開%\' or topic like \'%startup%\') order by updated_at desc limit 5').fetchall()
for topic, snippet in rows:
    print('\n---', topic, '---')
    print(snippet.replace('\\n',' '))
conn.close()
