"""
PyArrow: Format Conversions and Plain English Pipeline Definitions
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as csv
import pyarrow.json as json
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.flight as flight
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
import io
import re

# ============================================================================
# PYARROW FORMAT CONVERSION CAPABILITIES
# ============================================================================

print("=" *80)
print("PYARROW FORMAT CONVERSION CAPABILITIES")
print("=" *80)

print("""
PyArrow is EXTREMELY effective for format conversions because:

1. **Zero-Copy Operations**: Arrow's columnar format allows converting between 
   formats without copying data when possible

2. **Unified Memory Format**: All conversions go through Arrow's standard format,
   making it a universal translator

3. **Streaming Support**: Can process files larger than memory

4. **Rich Type System**: Preserves data types across conversions

5. **Performance**: Written in C++ with optimized implementations
""")

# ============================================================================
# SUPPORTED FORMATS
# ============================================================================


class FormatConverter:
    """Demonstrates PyArrow's format conversion capabilities"""

    @staticmethod
    def show_supported_formats():
        formats = {
            'Input Formats': [
                'CSV - Comma/Tab/Pipe separated values',
                'JSON - JSON files and line-delimited JSON',
                'Parquet - Columnar storage format',
                'Feather/Arrow IPC - Native Arrow format',
                'ORC - Optimized Row Columnar',
                'Pandas DataFrame - Direct integration',
                'NumPy arrays - Zero-copy when possible',
                'Python lists/dicts - Native Python objects',
                'Database results - Via ADBC or other connectors',
                'Apache Kafka - Via streaming APIs',
                'Flight - Arrow Flight Protocol',
            ],
            'Output Formats': [
                'All input formats',
                'Plus: Excel (via Pandas)',
                'Plus: HDF5 (via Pandas)',
                'Plus: Stata/SAS/SPSS (via Pandas)',
                'Plus: SQL databases (via ADBC)',
                'Plus: Cloud storage (S3, GCS, Azure)',
            ]
        }
        return formats


# Create sample data for demonstrations
sample_data = pa.table({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000.0, 60000.0, 75000.0, 55000.0, 70000.0],
    'department': ['Sales', 'Engineering', 'Sales', 'HR', 'Engineering'],
    'joined_date': pd.date_range('2020-01-01', periods=5, freq='3M').tolist(),
    'is_active': [True, True, False, True, True],
    'scores': [[85, 92], [78, 88], [91, 95], [82, 87], [89, 94]],  # Nested array
})

print("\nSample Table Schema:")
print(sample_data.schema)

# ============================================================================
# FORMAT CONVERSION EXAMPLES
# ============================================================================

print("\n" + "=" *80)
print("FORMAT CONVERSION EXAMPLES")
print("=" *80)

# 1. CSV Conversions
print("\n1. CSV CONVERSIONS:")
print("-" * 40)

# Write to CSV
csv_buffer = io.BytesIO()
csv.write_csv(sample_data, csv_buffer)
print(f"CSV size: {csv_buffer.tell()} bytes")

# Read back with options
csv_buffer.seek(0)
csv_options = csv.ReadOptions(column_names=['id', 'name', 'age', 'salary', 'dept', 'date', 'active', 'scores'])
convert_options = csv.ConvertOptions(
    column_types={'salary': pa.float32(), 'age': pa.int32()},
    strings_can_be_null=True,
    include_columns=['id', 'name', 'salary']
)
table_from_csv = csv.read_csv(csv_buffer, read_options=csv_options, convert_options=convert_options)
print(f"Filtered CSV columns: {table_from_csv.column_names}")

# 2. Parquet Conversions
print("\n2. PARQUET CONVERSIONS:")
print("-" * 40)

# Write to Parquet with compression
parquet_buffer = io.BytesIO()
pq.write_table(sample_data, parquet_buffer, compression='snappy')
parquet_size = parquet_buffer.tell()
print(f"Parquet size (snappy): {parquet_size} bytes")

# Compare with different compressions
compressions = ['none', 'snappy', 'gzip', 'brotli', 'zstd']
for comp in compressions:
    buffer = io.BytesIO()
    pq.write_table(sample_data, buffer, compression=comp)
    print(f"  {comp}: {buffer.tell()} bytes")

# 3. JSON Conversions
print("\n3. JSON CONVERSIONS:")
print("-" * 40)

# Convert to JSON
json_buffer = io.BytesIO()
json.write_json(sample_data, json_buffer)
print(f"JSON size: {json_buffer.tell()} bytes")

# Read JSON with schema
json_buffer.seek(0)
schema = pa.schema([
    ('id', pa.int64()),
    ('name', pa.string()),
    ('salary', pa.float64())
])
table_from_json = json.read_json(json_buffer, parse_options=json.ParseOptions(explicit_schema=schema))

# 4. Feather/Arrow IPC
print("\n4. FEATHER/ARROW IPC:")
print("-" * 40)

# Write to Feather (Arrow IPC format)
feather_buffer = io.BytesIO()
with pa.ipc.new_file(feather_buffer, sample_data.schema) as writer:
    writer.write_table(sample_data)
print(f"Feather size: {feather_buffer.tell()} bytes")

# 5. Pandas Integration
print("\n5. PANDAS INTEGRATION:")
print("-" * 40)

# To Pandas (zero-copy for many types)
df = sample_data.to_pandas()
print(f"DataFrame shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum()} bytes")

# From Pandas (with type preservation)
table_from_pandas = pa.Table.from_pandas(df, preserve_index=False)
print(f"Round-trip successful: {table_from_pandas.equals(sample_data)}")

# ============================================================================
# PLAIN ENGLISH PIPELINE DEFINITION
# ============================================================================


class PlainEnglishPipeline:
    """
    Converts plain English descriptions to PyArrow operations
    """

    def __init__(self):
        self.patterns = self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for plain English commands"""
        return [
            # Data loading
            (r'load (\w+) file "([^"]+)"', self._load_file),
            (r'read (\w+) from "([^"]+)"', self._load_file),

            # Filtering
            (r'filter where (\w+) is greater than ([\d.]+)', self._filter_gt),
            (r'filter where (\w+) equals "([^"]+)"', self._filter_eq),
            (r'keep only rows where (\w+) is not null', self._filter_not_null),
            (r'remove rows where (\w+) is null', self._drop_null),

            # Selection
            (r'select columns? ([\w\s,]+)', self._select_columns),
            (r'drop columns? ([\w\s,]+)', self._drop_columns),
            (r'keep only ([\w\s,]+) columns?', self._select_columns),

            # Transformation
            (r'rename (\w+) to (\w+)', self._rename_column),
            (r'convert (\w+) to (\w+)', self._convert_type),
            (r'uppercase (\w+)', self._uppercase),
            (r'trim whitespace from (\w+)', self._trim),
            (r'round (\w+) to (\d+) decimal places?', self._round),

            # Calculation
            (r'add new column (\w+) as (\w+) plus (\w+)', self._add_columns),
            (r'calculate (\w+) as (\w+) times ([\d.]+)', self._multiply),
            (r'compute average of (\w+) as (\w+)', self._compute_avg),

            # Aggregation
            (r'group by ([\w\s,]+) and sum (\w+)', self._group_sum),
            (r'count unique values in (\w+)', self._count_unique),

            # Sorting
            (r'sort by (\w+) ascending', self._sort_asc),
            (r'sort by (\w+) descending', self._sort_desc),
            (r'order by (\w+)', self._sort_asc),

            # Output
            (r'save as (\w+) file "([^"]+)"', self._save_file),
            (r'export to (\w+) "([^"]+)"', self._save_file),
            (r'write to (\w+) file "([^"]+)"', self._save_file),
        ]

    def parse(self, command: str, table: pa.Table) -> pa.Table:
        """Parse plain English command and apply to table"""
        command = command.lower().strip()

        for pattern, handler in self.patterns:
            match = re.match(pattern, command)
            if match:
                return handler(table, *match.groups())

        raise ValueError(f"Could not understand: '{command}'")

    # Handler methods
    def _filter_gt(self, table, column, value):
        return table.filter(pc.greater(table[column], float(value)))

    def _filter_eq(self, table, column, value):
        return table.filter(pc.equal(table[column], value))

    def _filter_not_null(self, table, column):
        return table.filter(pc.is_valid(table[column]))

    def _drop_null(self, table, column):
        return table.filter(pc.is_valid(table[column]))

    def _select_columns(self, table, columns_str):
        columns = [c.strip() for c in columns_str.split(',')]
        return table.select(columns)

    def _drop_columns(self, table, columns_str):
        columns = [c.strip() for c in columns_str.split(',')]
        remaining = [c for c in table.column_names if c not in columns]
        return table.select(remaining)

    def _rename_column(self, table, old_name, new_name):
        names = [new_name if n == old_name else n for n in table.column_names]
        return table.rename_columns(names)

    def _convert_type(self, table, column, type_name):
        type_map = {
            'integer': pa.int64(),
            'float': pa.float64(),
            'string': pa.string(),
            'date': pa.date32(),
        }
        new_column = pc.cast(table[column], type_map.get(type_name, pa.string()))
        return table.set_column(
            table.column_names.index(column),
            column,
            new_column
        )

    def _uppercase(self, table, column):
        new_column = pc.utf8_upper(table[column])
        return table.set_column(
            table.column_names.index(column),
            column,
            new_column
        )

    def _trim(self, table, column):
        new_column = pc.utf8_trim(table[column])
        return table.set_column(
            table.column_names.index(column),
            column,
            new_column
        )

    def _round(self, table, column, places):
        new_column = pc.round(table[column], int(places))
        return table.set_column(
            table.column_names.index(column),
            column,
            new_column
        )

    def _add_columns(self, table, new_name, col1, col2):
        new_column = pc.add(table[col1], table[col2])
        return table.append_column(new_name, new_column)

    def _multiply(self, table, new_name, column, factor):
        new_column = pc.multiply(table[column], float(factor))
        return table.append_column(new_name, new_column)

    def _sort_asc(self, table, column):
        indices = pc.sort_indices(table[column])
        return table.take(indices)

    def _sort_desc(self, table, column):
        indices = pc.sort_indices(table[column], sort_keys=[('', 'descending')])
        return table.take(indices)

    def _load_file(self, table, format_type, filename):
        # Simplified - would implement actual file loading
        print(f"Would load {format_type} file from {filename}")
        return table

    def _save_file(self, table, format_type, filename):
        # Simplified - would implement actual file saving
        print(f"Would save as {format_type} to {filename}")
        return table

    def _count_unique(self, table, column):
        unique_values = pc.unique(table[column])
        print(f"Unique values in {column}: {len(unique_values)}")
        return table

    def _compute_avg(self, table, column, new_name):
        avg = pc.mean(table[column])
        # Would add as new column with repeated value
        return table

    def _group_sum(self, table, group_cols, sum_col):
        # Simplified - would use group_by
        print(f"Would group by {group_cols} and sum {sum_col}")
        return table


# ============================================================================
# PLAIN ENGLISH PIPELINE EXAMPLES
# ============================================================================

print("\n" + "=" *80)
print("PLAIN ENGLISH PIPELINE EXAMPLES")
print("=" *80)

pipeline = PlainEnglishPipeline()

# Example pipeline definitions
pipelines = {
    "Data Cleaning Pipeline": [
        "filter where age is greater than 25",
        "remove rows where department is null",
        "trim whitespace from name",
        "uppercase department",
        "rename department to dept",
        "select columns id, name, age, salary, dept",
        "sort by salary descending",
    ],

    "Financial Analysis Pipeline": [
        "filter where salary is greater than 50000",
        "add new column total_comp as salary plus salary",
        "calculate tax as salary times 0.25",
        "round tax to 2 decimal places",
        "keep only name, salary, tax, total_comp columns",
        "sort by total_comp descending",
    ],

    "Data Export Pipeline": [
        "filter where is_active equals \"True\"",
        "convert joined_date to string",
        "drop columns scores",
        "save as csv file \"active_employees.csv\"",
        "save as parquet file \"active_employees.parquet\"",
    ]
}

print("Plain English Pipeline Definitions:\n")

for pipeline_name, commands in pipelines.items():
    print(f"{pipeline_name}:")
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    print()

# Execute a pipeline
print("Executing Data Cleaning Pipeline:")
print("-" * 40)

result = sample_data
for command in pipelines["Data Cleaning Pipeline"]:
    print(f"Step: {command}")
    try:
        result = pipeline.parse(command, result)
        print(f"  → Result has {len(result)} rows, {len(result.column_names)} columns")
    except Exception as e:
        print(f"  → Error: {e}")

# ============================================================================
# ADVANCED CONVERSION PATTERNS
# ============================================================================

print("\n" + "=" *80)
print("ADVANCED CONVERSION PATTERNS")
print("=" *80)


class AdvancedConverter:
    """Advanced format conversion patterns"""

    @staticmethod
    def streaming_conversion(input_file: str, output_file: str):
        """Convert large files using streaming"""
        print(f"\nStreaming conversion: {input_file} → {output_file}")

        # Example: CSV to Parquet streaming
        # This would actually work with real files
        """
        # Define schema for consistent typing
        schema = pa.schema([
            ('id', pa.int64()),
            ('name', pa.string()),
            ('value', pa.float64())
        ])
        
        # Stream read and write
        with pq.ParquetWriter(output_file, schema) as writer:
            # Read CSV in chunks
            for chunk in pd.read_csv(input_file, chunksize=10000):
                table = pa.Table.from_pandas(chunk, schema=schema)
                writer.write_table(table)
        """
        print("  → Converts in chunks, handles files larger than memory")

    @staticmethod
    def multi_file_dataset():
        """Work with partitioned datasets"""
        print("\nMulti-file dataset patterns:")

        # PyArrow can treat multiple files as one dataset
        """
        # Read partitioned parquet dataset
        dataset = ds.dataset('path/to/partitioned/data', format='parquet')
        
        # Filter across all files efficiently
        filtered = dataset.filter(pc.greater(ds.field('year'), 2020))
        
        # Convert entire dataset to different format
        ds.write_dataset(
            filtered, 
            'output/path',
            format='csv',
            partitioning=['year', 'month']
        )
        """
        print("  → Handles partitioned data transparently")
        print("  → Pushes filters down to file level")
        print("  → Parallel processing of files")

    @staticmethod
    def format_optimization():
        """Optimize formats for specific use cases"""
        print("\nFormat optimization strategies:")

        strategies = {
            "For Analytics": {
                "format": "Parquet",
                "compression": "snappy",
                "reason": "Columnar format, fast queries, good compression"
            },
            "For Streaming": {
                "format": "Arrow IPC/Feather",
                "compression": "lz4",
                "reason": "Zero-copy reads, low latency"
            },
            "For Archival": {
                "format": "Parquet",
                "compression": "zstd",
                "reason": "Best compression ratio, long-term storage"
            },
            "For Data Exchange": {
                "format": "CSV/JSON",
                "compression": "gzip",
                "reason": "Universal compatibility"
            },
            "For Real-time": {
                "format": "Arrow Flight",
                "compression": "none",
                "reason": "Minimal latency, network optimized"
            }
        }

        for use_case, config in strategies.items():
            print(f"\n  {use_case}:")
            print(f"    Format: {config['format']}")
            print(f"    Compression: {config['compression']}")
            print(f"    Reason: {config['reason']}")


# ============================================================================
# PIPELINE COMPOSITION PATTERNS
# ============================================================================

print("\n" + "=" *80)
print("PIPELINE COMPOSITION PATTERNS")
print("=" *80)


class PipelineComposer:
    """Compose complex pipelines from simple operations"""

    def __init__(self):
        self.operations = []

    def add_operation(self, description: str, func):
        """Add operation to pipeline"""
        self.operations.append({
            'description': description,
            'function': func
        })
        return self

    def execute(self, table: pa.Table) -> pa.Table:
        """Execute all operations in sequence"""
        result = table
        for op in self.operations:
            print(f"Executing: {op['description']}")
            result = op['function'](result)
        return result

    # Fluent API for common operations
    def load_csv(self, filename: str):
        return self.add_operation(
            f"Load CSV from {filename}",
            lambda t: csv.read_csv(filename)
        )

    def filter_nulls(self, column: str):
        return self.add_operation(
            f"Remove null values in {column}",
            lambda t: t.filter(pc.is_valid(t[column]))
        )

    def add_calculated_column(self, name: str, expression: str):
        return self.add_operation(
            f"Add column {name} = {expression}",
            lambda t: t  # Would implement expression evaluation
        )

    def save_parquet(self, filename: str):
        return self.add_operation(
            f"Save as Parquet to {filename}",
            lambda t: pq.write_table(t, filename) or t
        )


# Example of fluent pipeline
print("Fluent Pipeline Example:")
print("-" * 40)

pipeline = (PipelineComposer()
            .add_operation("Start with sample data", lambda t: sample_data)
            .filter_nulls("name")
            .add_calculated_column("annual_bonus", "salary * 0.15")
            .save_parquet("output.parquet")
            )

print("\nPipeline steps:")
for i, op in enumerate(pipeline.operations, 1):
    print(f"  {i}. {op['description']}")


# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

print("\n" + "=" *80)
print("FORMAT CONVERSION PERFORMANCE")
print("=" *80)

print("""
PyArrow Format Conversion Performance:

1. **CSV → Parquet**: 
   - 5-10x faster than pandas
   - Streaming support for large files
   - Type inference or explicit schema

2. **Parquet → Arrow**:
   - Zero-copy in many cases
   - Lazy loading of columns
   - Predicate pushdown

3. **JSON → Arrow**:
   - Optimized C++ parser
   - Schema inference or validation
   - Handles nested data efficiently

4. **Pandas ↔ Arrow**:
   - Zero-copy for many types
   - Preserves pandas metadata
   - Handles nullable types properly

5. **Database → Arrow**:
   - Direct columnar transfer with ADBC
   - Batch fetching
   - Type mapping optimization

Real-world benchmarks:
- 1GB CSV → Parquet: ~3 seconds
- 1GB Parquet → Pandas: ~0.5 seconds  
- 1M rows Pandas → Arrow: ~0.01 seconds (zero-copy)
- 1GB JSON → Arrow Table: ~5 seconds
""")
