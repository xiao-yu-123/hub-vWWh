import sqlite3
# sqlite单机关系型数据库

# 连接到Chinook数据库
conn = sqlite3.connect('chinook.db')  # 数据库文件，包含多张表
# 创建一个游标对象
cursor = conn.cursor()

# 获取数据库中所有表的名称
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

'''数据库解析'''
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData
import pandas as pd
from openai import OpenAI
import json
from typing import List, Tuple, Union, Optional, Dict, Any
import re

class DBParser:
    '''DBParser 数据库解析'''

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_db_schema(self) -> Dict[str, Any]:
        """
        提取 Chinook 数据库的完整结构信息：
        - 所有业务表名（排除 sqlite 内部表）
        - 每个表的列名、数据类型、主键、外键关系
        """
        # 获取所有表名（排除 sqlite_ 开头的系统表）
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        table1 = [row[0] for row in self.cursor.fetchall()]

        schema_info = {"tables": {}}
        foreign_keys = []

        for table in table1:
            # 列信息
            self.cursor.execute(f"PRAGMA table_info({table})")
            columns = self.cursor.fetchall()
            col_list = []
            pk_cols = []
            for col in columns:
                col_dict = {
                    "name": col[1],
                    "type": col[2],
                    "notnull": col[3],
                    "pk": col[5]
                }
                col_list.append(col_dict)
                if col[5] == 1:
                    pk_cols.append(col[1])

            # 外键信息
            self.cursor.execute(f"PRAGMA foreign_key_list({table})")
            fks = self.cursor.fetchall()
            for fk in fks:
                foreign_keys.append({
                    "table": table,
                    "from": fk[3],  # 本表列名
                    "to_table": fk[2],  # 引用表
                    "to_col": fk[4]  # 引用列
                })

            schema_info["tables"][table] = {
                "columns": col_list,
                "primary_keys": pk_cols,
                "foreign_keys": fks
            }

        # self.conn.close()
        schema_info["foreign_key_relations"] = foreign_keys
        return schema_info

    # ==================== 2. 构造 Schema 描述文本 ====================
    def format_schema_for_llm(self, schema: Dict[str, Any]) -> str:
        """将数据库结构格式化为 LLM 易读的文本"""
        lines = []
        lines.append("数据库 Schema 信息：\n")
        for table_name, info in schema["tables"].items():
            lines.append(f"表名: {table_name}")
            col_desc = []
            for col in info["columns"]:
                pk_mark = " (主键)" if col["pk"] else ""
                col_desc.append(f"  - {col['name']} ({col['type']}){pk_mark}")
            lines.extend(col_desc)

            # 外键关系
            fk_list = [fk for fk in schema["foreign_key_relations"] if fk["table"] == table_name]
            if fk_list:
                fk_desc = "外键: " + ", ".join([f"{fk['from']} -> {fk['to_table']}.{fk['to_col']}" for fk in fk_list])
                lines.append(f"  {fk_desc}")
            lines.append("")  # 空行分隔
        return "\n".join(lines)

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        cursor.execute(sql)
        results = cursor.fetchall()
        return results

# 调用大模型
client = OpenAI(
    api_key="sk-9bf45d961ac64f75a3b6a64c7fd08817",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class SQLAgent:
    # 初始化
    def __init__(self, parser):
        self.parser = parser

    # 构建提示词
    def _build_prompt(self, question):
        schema = self.parser.get_db_schema()
        schema_text = self.parser.format_schema_for_llm(schema)
        prompt = f"""你是一个 SQL 专家。请根据用户的问题，生成一个可以在 SQLite 数据库中执行的 SELECT 查询语句。
只输出 SQL 代码，不要有任何额外解释。

数据库 Schema 如下：
{schema_text}

用户问题：{question}
SQL：
        """
        return prompt

    def ask_llm(self, prompt, model="qwen-plus", temperature=0.1):
        """调用大模型，prompt为字符串"""
        messages = [{"role": "user", "content": prompt}]
        # print(json.dumps(messages, indent=4, ensure_ascii=False))
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            sql = completion.choices[0].message.content.strip()
            # 清理可能的 markdown 代码块标记
            sql = re.sub(r"```sql\n?|```", "", sql).strip()
            return sql
        except Exception as e:
            print('调用大模型失败:', e)
            return None
    # 智能体入口

    def format_answer(self, question: str, sql: str, results: List[tuple]) -> str:
        """将 SQL 执行结果组织成自然语言回答（也可再次调用 LLM 润色）"""
        # 简单格式化：如果结果只有一个数值，直接回答；否则展示表格
        if len(results) == 0:
            return "查询结果为空。"
        if len(results) == 1 and len(results[0]) == 1:
            value = results[0][0]
            # 针对三个特定问题做友好输出
            if "多少张表" in question:
                return f"数据库中总共有 {value} 张表。"
            if "员工表" in question and "记录" in question:
                return f"员工表中共有 {value} 条记录。"
            if "客户个数和员工个数" in question:
                # 假设 SQL 返回两列 (customer_count, employee_count)
                if len(results[0]) == 2:
                    return f"客户个数为 {results[0][0]}，员工个数为 {results[0][1]}。"
            return f"结果为 {value}。"
        # 通用返回：将结果拼接为文本
        return f"查询结果：\n" + "\n".join(str(row) for row in results)

    def action(self, question):
        # 生成sql
        prompt = self._build_prompt(question)
        sql = self.ask_llm(prompt)
        result = self.parser.execute_sql(sql)
        res = self.format_answer(question, sql, result)

        return res


if __name__ == '__main__':
    parser = DBParser("chinook.db")
    agent = SQLAgent(parser)
    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？",
    ]
    for q in questions:
        print(f"问题：{q}\n")
        print(f"回答：{agent.action(q)}\n")
