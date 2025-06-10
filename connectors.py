# connectors.py
import json
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as _connection
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import WriteOptions

from config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DATABASE, POSTGRES_USER,
    POSTGRES_PASSWORD, POSTGRES_CONFIGURED, INFLUXDB_URL, INFLUXDB_TOKEN,
    INFLUXDB_ORG, INFLUXDB_BUCKET, INFLUXDB_CONFIGURED
)

logger = logging.getLogger(__name__)

class PostgresConnector:
    """
    Manages connections and queries to a PostgreSQL database.
    Provides methods for schema description and executing SQL queries.
    """
    def __init__(self):
        self.conn_params = {
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "database": POSTGRES_DATABASE,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
        }
        self._is_connected = False
        self.connection: Optional[_connection] = None

        if not POSTGRES_CONFIGURED:
            logger.error("PostgreSQL configuration is incomplete or disabled. Skipping connection attempt.")
            return

        try:
            logger.info(f"Attempting to connect to PostgreSQL at {self.conn_params.get('host')}:{self.conn_params.get('port')}...")
            self.connection = psycopg2.connect(**self.conn_params)
            self.connection.autocommit = True
            self._is_connected = True
            logger.info("Connected to PostgreSQL successfully.")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}", exc_info=True)
            self.connection = None
            self._is_connected = False

    def is_connected(self) -> bool:
        """Checks if the connection to PostgreSQL is active."""
        if self.connection is None:
            return False
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return self._is_connected
        except Exception:
            self._is_connected = False
            return False

    def get_schema_description(self) -> str:
        """
        Retrieves and formats the schema description for predefined tables of interest.
        Returns a JSON string of the schema or an error message.
        """
        if not self.is_connected():
            return "PostgreSQL database not connected, schema unavailable."

        tables_of_interest_desc = {
            "public.mrp_workorder": "Work Orders (Details about manufacturing work orders)",
            "public.mrp_production": "Production Orders (Manufacturing production orders)",
            "public.product_product": "Product Variants (Specific product variants)",
            "public.product_template": "Product Templates (General product information)",
            "public.crm_lead": "CRM Leads/Opportunities (Sales leads and opportunities)",
            "public.res_partner": "Partners (Customer, vendor, and contact information)",
            "cobot_data.rtde_logs": "Cobot RTDE Logs (Historical sensor data from cobot, including joints, forces, temperatures, and modes)"
        }

        # Explicitly define columns for cobot_data.rtde_logs as it's a custom table
        rtde_logs_columns_definition = [
            {"column_name": "timestamp", "data_type": "timestamp with time zone"},
            {"column_name": "joint1", "data_type": "double precision"}, {"column_name": "joint2", "data_type": "double precision"},
            {"column_name": "joint3", "data_type": "double precision"}, {"column_name": "joint4", "data_type": "double precision"},
            {"column_name": "joint5", "data_type": "double precision"}, {"column_name": "joint6", "data_type": "double precision"},
            {"column_name": "fx", "data_type": "double precision"}, {"column_name": "fy", "data_type": "double precision"},
            {"column_name": "fz", "data_type": "double precision"}, {"column_name": "tx", "data_type": "double precision"},
            {"column_name": "ty", "data_type": "double precision"}, {"column_name": "tz", "data_type": "double precision"},
            {"column_name": "temp1", "data_type": "double precision"}, {"column_name": "temp2", "data_type": "double precision"},
            {"column_name": "temp3", "data_type": "double precision"}, {"column_name": "temp4", "data_type": "double precision"},
            {"column_name": "temp5", "data_type": "double precision"}, {"column_name": "temp6", "data_type": "double precision"},
            {"column_name": "robot_mode", "data_type": "integer"}, {"column_name": "safety_mode", "data_type": "integer"}
        ]

        final_schema_representation = {}
        try:
            with self.connection.cursor() as cursor:
                for full_table_name_str, friendly_description in tables_of_interest_desc.items():
                    schema_name, table_name_simple = full_table_name_str.split('.')
                    
                    current_table_columns = []
                    if full_table_name_str == "cobot_data.rtde_logs":
                        current_table_columns = rtde_logs_columns_definition
                    else:
                        cursor.execute(sql.SQL("""
                            SELECT column_name, data_type 
                            FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s;
                        """), [schema_name, table_name_simple])
                        db_columns = cursor.fetchall()
                        
                        if not db_columns:
                            logger.warning(f"Table {full_table_name_str} not found or has no columns in information_schema. Will be excluded from schema description.")
                            continue
                        for col in db_columns:
                            current_table_columns.append({"column_name": col[0], "data_type": col[1]})
                    
                    final_schema_representation[full_table_name_str] = {
                        "description": friendly_description,
                        "columns": current_table_columns
                    }
            
            if not final_schema_representation:
                return "No schema information could be retrieved for the specified tables of interest. Ensure tables exist and are accessible."
            return json.dumps(final_schema_representation, indent=2)

        except Exception as e:
            logger.error(f"Error fetching PostgreSQL schema for specified tables: {e}", exc_info=True)
            return f"Error fetching PostgreSQL schema: {e}"
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a read-only SQL SELECT query against the PostgreSQL database.
        Returns a list of dictionaries, where each dictionary represents a row.
        """
        if not self.is_connected():
            logger.error("PostgreSQL connection is not available to execute query.")
            return [{"error": "PostgreSQL connection not available."}]

        if not query.strip().upper().startswith("SELECT"):
            logger.warning(f"Attempted to execute non-SELECT query: {query[:100]}...")
            return [{"error": "Only read-only SELECT queries are allowed."}]

        logger.info(f"Executing PostgreSQL query: {query[:200]}...")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    logger.info(f"PostgreSQL query executed successfully. Returned {len(results)} rows.")
                    logger.debug(f"PostgreSQL query full output: {results}")
                    return results
                else:
                    logger.info("PostgreSQL SELECT query executed but returned no description (possibly empty result set).")
                    return []

        except Exception as e:
            logger.error(f"Error executing PostgreSQL query: {e}", exc_info=True)
            return [{"error": str(e)}]

    def get_recent_work_orders(self, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Retrieves the most recent work orders."""
        work_orders_table = sql.Identifier("mrp_workorder")
        production_table = sql.Identifier("mrp_production")
        product_variant_table = sql.Identifier("product_product")
        products_table = sql.Identifier("product_template")

        query = sql.SQL("""
            SELECT w.name AS work_order, p.name AS product, w.state, w.date_planned_start
            FROM {work_orders_table} w
            JOIN {production_table} pr ON w.production_id = pr.id
            JOIN {product_variant_table} pp ON w.product_id = pp.id
            JOIN {products_table} p ON pp.product_tmpl_id = p.id
            ORDER BY w.date_planned_start DESC
            LIMIT %s;
        """).format(
            work_orders_table=work_orders_table,
            production_table=production_table,
            product_variant_table=product_variant_table,
            products_table=products_table
        )
        return self.execute_query(query.as_string(self.connection), (limit,))

    def get_recent_opportunities_with_orders(self, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Retrieves recent CRM opportunities with associated production orders."""
        crm_table = sql.Identifier("crm_lead")
        partner_table = sql.Identifier("res_partner")
        production_table = sql.Identifier("mrp_production")

        query = sql.SQL("""
            SELECT crm.name AS opportunity, res.name AS customer, pr.name AS production_order
            FROM {crm_table} crm
            JOIN {partner_table} res ON crm.partner_id = res.id
            LEFT JOIN {production_table} pr ON crm.name = pr.origin
            ORDER BY crm.create_date DESC
            LIMIT %s;
        """).format(
            crm_table=crm_table,
            partner_table=partner_table,
            production_table=production_table
        )
        return self.execute_query(query.as_string(self.connection), (limit,))

    def get_recent_customers(self, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Fetches information about the most recently added customers."""
        partner_table = sql.Identifier("res_partner")

        query = sql.SQL("""
            SELECT name, email, phone
            FROM {partner_table}
            ORDER BY create_date DESC
            LIMIT %s;
        """).format(partner_table=partner_table)
        return self.execute_query(query.as_string(self.connection), (limit,))

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Returns a list of tool descriptions for PostgreSQL operations."""
        tools = []
        if self.is_connected():
            tools.append({
                "name": "postgres_query",
                "description": (
                    "Use this tool to execute read-only SQL queries against the PostgreSQL database "
                    "containing Odoo ERP data (work orders, production, CRM, customers, etc.). "
                    "Input should be a valid SQL SELECT query string. "
                    "Only use SELECT queries. Do not attempt INSERT, UPDATE, DELETE, or DDL commands. "
                    "If you need schema information, request the 'get_postgres_schema' tool, then form your query based on that."
                ),
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The SQL SELECT query string."}}, "required": ["query"]}
            })
            tools.append({
                "name": "get_recent_work_orders",
                "description": (
                    "Retrieves the most recent work orders from the PostgreSQL database. "
                    "This tool is useful for getting an overview of manufacturing tasks. "
                    "It takes an optional integer `limit` parameter to specify the number of recent orders to retrieve (default is 5). "
                    "Example: 'Show me the last 10 work orders' or 'What are the recent work orders?'"
                ),
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "The maximum number of work orders to retrieve."}}}
            })
            tools.append({
                "name": "get_recent_opportunities_with_orders",
                "description": (
                    "Retrieves recent CRM opportunities along with associated production orders from PostgreSQL. "
                    "Useful for tracking sales pipeline and manufacturing linkage. "
                    "Takes an optional integer `limit` parameter (default is 5)."
                    "Example: 'What are our latest sales opportunities?' or 'Show me recent opportunities and their production orders.'"
                ),
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "The maximum number of opportunities to retrieve."}}}
            })
            tools.append({
                "name": "get_recent_customers",
                "description": (
                    "Fetches information about the most recently added customers from PostgreSQL. "
                    "Useful for reviewing new client acquisitions. "
                    "Takes an optional integer `limit` parameter (default is 5)."
                    "Example: 'Who are our newest customers?' or 'List the 5 most recent customers.'"
                ),
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "The maximum number of customers to retrieve."}}}
            })
            tools.append({
                "name": "get_postgres_schema",
                "description": (
                    "Retrieves a detailed description of the PostgreSQL database schema (tables and columns). "
                    "Use this when the user asks about database structure or when you need to form a complex SQL query. "
                    "This tool takes no parameters."
                ),
                "parameters": {"type": "object", "properties": {}}
            })

        return tools

    def get_tool_function(self, tool_name: str):
        """Returns the function object for a given tool name."""
        if not self.is_connected():
            return None
        if tool_name == "postgres_query":
            return self.execute_query
        if tool_name == "get_recent_work_orders":
            return self.get_recent_work_orders
        if tool_name == "get_recent_opportunities_with_orders":
            return self.get_recent_opportunities_with_orders
        if tool_name == "get_recent_customers":
            return self.get_recent_customers
        if tool_name == "get_postgres_schema":
            return self.get_schema_description
        return None

class InfluxDBConnector:
    """
    Manages connections and queries to an InfluxDB database.
    Provides methods for schema description and executing Flux queries.
    """
    def __init__(self):
        self.config = {
            "url": INFLUXDB_URL,
            "token": INFLUXDB_TOKEN,
            "org": INFLUXDB_ORG,
            "bucket": INFLUXDB_BUCKET
        }
        self._is_connected = False
        self.client: Optional[InfluxDBClient] = None
        self.query_api = None

        if not INFLUXDB_CONFIGURED:
            logger.error("InfluxDB configuration is incomplete or disabled. Skipping connection attempt.")
            return

        try:
            logger.info(f"Attempting to connect to InfluxDB at {self.config.get('url')}...")
            self.client = InfluxDBClient(
                url=self.config["url"],
                token=self.config["token"],
                org=self.config["org"]
            )
            if not self.client.ping():
                raise ConnectionError("InfluxDB ping failed. Server might be down or unreachable.")

            self.query_api = self.client.query_api()
            self._is_connected = True
            logger.info("Connected to InfluxDB successfully.")
        except Exception as e:
            logger.error(f"InfluxDB connection failed: {e}", exc_info=True)
            self.client = None
            self.query_api = None
            self._is_connected = False

    def is_connected(self) -> bool:
        """Checks if the connection to InfluxDB is active."""
        return self._is_connected

    def get_schema_description(self) -> str:
        """
        Retrieves and formats the schema description for InfluxDB.
        Returns a JSON string of the schema or an error message.
        """
        if not self.is_connected():
            return "InfluxDB database not connected, schema unavailable."
        
        org_name = self.config.get("org")
        bucket_name = self.config.get("bucket")
        if not org_name or not bucket_name:
            return "InfluxDB organization or bucket not configured, schema unavailable."

        schema_info = {"buckets": {}}
        try:
            flux_query_measurements = f'''
            import "influxdata/influxdb/schema"
            schema.measurements(bucket: "{bucket_name}")
            '''
            tables_measurements = self.query_api.query(query=flux_query_measurements, org=org_name)
            
            measurements = [record.get_value() for table in tables_measurements for record in table.records]
            
            schema_info["buckets"][bucket_name] = {"measurements": {}}
            for measurement in measurements:
                # Corrected queries for fields and tags per measurement
                fields_query = f'''
                import "influxdata/influxdb/schema"
                schema.measurementFieldKeys(bucket: "{bucket_name}", measurement: "{measurement}")
                '''
                tags_query = f'''
                import "influxdata/influxdb/schema"
                schema.measurementTagKeys(bucket: "{bucket_name}", measurement: "{measurement}")
                '''
                
                fields_tables = self.query_api.query(query=fields_query, org=org_name)
                tags_tables = self.query_api.query(query=tags_query, org=org_name)
                
                fields = [record.get_value() for table in fields_tables for record in table.records]
                tags = [record.get_value() for table in tags_tables for record in table.records]
                
                schema_info["buckets"][bucket_name]["measurements"][measurement] = {
                    "fields": fields,
                    "tags": tags
                }
            
            if "cobot_telemetry" in schema_info["buckets"][bucket_name]["measurements"]:
                 schema_info["buckets"][bucket_name]["measurements"]["cobot_telemetry"]["description"] = "Real-time data from the cobot, including joint angles, forces, temperatures, and robot/safety modes."
            else:
                 logger.warning(f"cobot_telemetry measurement not found in bucket {bucket_name} during schema description.")

            return json.dumps(schema_info, indent=2)

        except Exception as e:
            logger.error(f"Error fetching InfluxDB schema: {e}", exc_info=True)
            return f"Error fetching schema: {e}"

    def execute_flux_query(self, flux_query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a Flux query against the InfluxDB database.
        Returns a list of dictionaries, where each dictionary represents a record.
        """
        if not self.is_connected() or self.query_api is None:
            logger.error("InfluxDB connection or query API not available to execute query.")
            return [{"error": "InfluxDB connection not available."}]

        bucket_name = self.config.get("bucket")
        org_name = self.config.get("org")

        if not bucket_name or not org_name:
            logger.error("InfluxDB bucket or organization name is missing in config.")
            return [{"error": "InfluxDB bucket or organization is not configured."}]

        # Prepend default bucket and range if not present in the query
        if not re.search(r'from\s*\(\s*bucket:\s*["\']?.*?["\']?\s*\)', flux_query, re.IGNORECASE):
            default_prefix = f'from(bucket: "{bucket_name}") |> range(start: -15m)'
            if flux_query.strip().startswith('|'):
                flux_query = f'{default_prefix} {flux_query.strip()}'
                logger.warning(f"Prepended default bucket/range to Flux query: {flux_query[:200]}...")
            else:
                flux_query = f'{default_prefix}\n{flux_query.strip()}'
                logger.warning(f"Prepended default bucket/range to Flux query: {flux_query[:200]}...")

        logger.info(f"Executing InfluxDB Flux query: {flux_query[:200]}...")

        try:
            tables = self.query_api.query(query=flux_query, org=org_name)

            if not tables:
                logger.info("InfluxDB query returned no results.")
                return []

            results_list: List[Dict[str, Any]] = []
            for table in tables:
                for record in table.records:
                    results_list.append(record.values)

            logger.info(f"InfluxDB Flux query executed successfully. Returned {len(results_list)} records.")
            logger.debug(f"InfluxDB Flux query full output: {results_list}")
            return results_list

        except InfluxDBError as e:
            logger.error(f"InfluxDB API Error executing Flux query: {e}", exc_info=True)
            error_message = getattr(e, 'message', str(e))
            return [{"error": f"InfluxDB query failed: {error_message}"}]
        except Exception as e:
            logger.error(f"An unexpected error occurred during query execution: {e}", exc_info=True)
            return [{"error": f"An unexpected error occurred: {str(e)}"}]

    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        """Retrieves the latest telemetry data for key cobot metrics from InfluxDB."""
        if not self.is_connected() or self.query_api is None:
            logger.error("InfluxDB connection or query API not available to get latest telemetry.")
            return {"error": "Not connected to InfluxDB"}

        bucket_name = self.config.get("bucket")
        org_name = self.config.get("org")

        if not bucket_name or not org_name:
            logger.error("InfluxDB bucket or organization name is missing in config for latest telemetry.")
            return {"error": "InfluxDB bucket or organization is not configured."}

        flux_query = f'''
        from(bucket: "{bucket_name}")
          |> range(start: -5m)
          |> filter(fn: (r) => r._measurement == "cobot_telemetry")
          |> filter(fn: (r) =>
              r._field == "joint1" or
              r._field == "fx" or
              r._field == "temp1" or
              r._field == "robot_mode" or
              r._field == "safety_mode"
          )
          |> last()
          |> yield(name: "last_values")
        '''

        logger.info("Executing InfluxDB query for latest telemetry...")

        try:
            result = self.query_api.query(org=org_name, query=flux_query)

            if not result:
                logger.info("InfluxDB latest telemetry query returned no results.")
                return {"info": "No recent telemetry available"}

            latest_data: Dict[str, Any] = {}
            for table in result:
                for record in table.records:
                    field = record.get_field()
                    value = record.get_value()

                    if field is not None:
                        latest_data[field] = value

            if not latest_data:
                logger.info("Processed InfluxDB latest telemetry results but no fields were found.")
                return {"info": "No recent telemetry available with specified fields"}

            logger.info(f"InfluxDB latest telemetry query executed successfully. Found data for fields: {list(latest_data.keys())}")
            logger.debug(f"InfluxDB latest telemetry data: {latest_data}")
            return latest_data

        except InfluxDBError as e:
            logger.error(f"InfluxDB API Error getting latest telemetry: {e}", exc_info=True)
            error_message = getattr(e, 'message', str(e))
            return {"error": f"InfluxDB query failed: {error_message}"}
        except Exception as e:
            logger.error(f"An unexpected error getting latest InfluxDB telemetry: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Returns a list of tool descriptions for InfluxDB operations."""
        tools = []
        if self.is_connected():
            tools.append({
                "name": "influxdb_query",
                "description": (
                    "Use this tool to execute Flux queries against the InfluxDB database "
                    "containing real-time Cobot telemetry data. Input should be a valid Flux query string. "
                    "This is useful for getting specific time series data, aggregations, or filtered results. "
                    "Example query: `from(bucket: \"your_cobot_bucket\") |> range(start: -15m) |> filter(fn: (r) => r[\"_measurement\"] == \"cobot_sensor\" and r[\"_field\"] == \"temperature\") |> last()`"
                    "If your query doesn't specify a bucket or range, the default configured bucket and a range of the last 15 minutes will be used automatically."
                    "Be specific about the time range and fields you are interested in."
                    "If you need schema information, request the 'get_influxdb_schema' tool, then form your query based on that."
                ),
                "parameters": {"type": "object", "properties": {"flux_query": {"type": "string", "description": "The Flux query string."}}, "required": ["flux_query"]}
            })
            tools.append({
                "name": "get_latest_telemetry",
                "description": (
                    "Retrieves the latest telemetry data for key cobot metrics (joint1, fx, temp1, robot_mode, safety_mode) "
                    "from the InfluxDB database. This is ideal for quick checks on current robot status. "
                    "This tool does not take any parameters."
                    "Example: 'What's the latest robot telemetry?' or 'Give me the current cobot status.'"
                ),
                "parameters": {"type": "object", "properties": {}}
            })
            tools.append({
                "name": "get_influxdb_schema",
                "description": (
                    "Retrieves a detailed description of the InfluxDB database schema (buckets, measurements, fields, and tags). "
                    "Use this when the user asks about database structure or when you need to form a complex Flux query. "
                    "This tool takes no parameters."
                ),
                "parameters": {"type": "object", "properties": {}}
            })

        return tools

    def get_tool_function(self, tool_name: str):
        """Returns the function object for a given tool name."""
        if not self.is_connected():
            return None

        if tool_name == "influxdb_query":
            return self.execute_flux_query
        if tool_name == "get_latest_telemetry":
            return self.get_latest_telemetry
        if tool_name == "get_influxdb_schema":
            return self.get_schema_description
        return None