# XVIEW - Visualization Layer

## Overview

XVIEW provides a unified visualization abstraction for both console and server-side rendering of data from Repos. It leverages Rich for console output and Plotly for web-based charts, with pandas as the data manipulation engine.

## Architecture Plan

### Core Design Principles

1. **Unified Interface** - Single API for both console and web visualizations
2. **Cache-First** - Leverage InMemoryRepo for fast pandas DataFrame operations
3. **Schema-Aware** - Use Inspector to understand data types for appropriate visualizations
4. **Streaming Support** - Handle large datasets with pagination and chunking
5. **Ultrathin Design** - Minimal abstraction, delegate to specialized libraries

### Component Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      Repo       │────▶│  InMemoryCache  │────▶│     Pandas      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                ┌─────────────────────────┴─────────────────────────┐
                                │                  XViewFactory                     │
                                ├──────────────────────────────────────────────────┤
                                │  • Creates appropriate view instances            │
                                │  • Selects correct adapter for data source      │
                                │  • Manages view lifecycle                        │
                                └────────────────────┬─────────────────────────────┘
                                                     │
                                ┌────────────────────┴─────────────────────────────┐
                                │                 ViewAdapter                      │
                                ├──────────────────────────────────────────────────┤
                                │  • RepoAdapter: Repo → DataFrame                 │
                                │  • DataFrameAdapter: DataFrame → View Format    │
                                │  • DictAdapter: Dict/List → DataFrame           │
                                └────────────────────┬─────────────────────────────┘
                                                     │
                                ┌────────────────────┴─────────────────────────────┐
                                │                     XView                         │
                                ├───────────────────────┬───────────────────────────┤
                                │    Console Output     │    Server-Side Output     │
                                ├───────────────────────┼───────────────────────────┤
                                │  • Rich Tables        │  • Plotly Charts          │
                                │  • Rich Trees         │  • HTML Tables            │
                                │  • Progress Bars      │  • Export (PNG/SVG/PDF)  │
                                │  • Pivot Tables       │  • Interactive Dashboards │
                                └───────────────────────┴───────────────────────────┘
```

## Planned Components

### 1. XViewFactory

```python
class XViewFactory:
    """Factory for creating views with appropriate adapters"""
    
    def __init__(self):
        self._view_registry = {}
        self._adapter_registry = {}
        
    def register_view(self, view_type: str, view_class: Type[ViewRenderer]):
        """Register a view renderer"""
        self._view_registry[view_type] = view_class
        
    def register_adapter(self, data_type: Type, adapter_class: Type[ViewAdapter]):
        """Register a data adapter"""
        self._adapter_registry[data_type] = adapter_class
        
    async def create_view(
        self,
        view_type: str,
        data: Union[Repo, pd.DataFrame, Dict, List],
        x: Optional[List[str]] = None,
        y: Optional[List[str]] = None,
        c: Optional[List[str]] = None,
        **kwargs
    ) -> ViewOutput:
        """Create and render a view with automatic adapter selection"""
        # Select appropriate adapter
        adapter = self._get_adapter(type(data))
        
        # Transform data
        prepared_data = await adapter.prepare(data, x=x, y=y, c=c)
        
        # Create and render view
        view = self._view_registry[view_type]()
        return await view.render(prepared_data, x=x, y=y, c=c, **kwargs)
        
    def _get_adapter(self, data_type: Type) -> ViewAdapter:
        """Get appropriate adapter for data type"""
        for dtype, adapter_class in self._adapter_registry.items():
            if issubclass(data_type, dtype):
                return adapter_class()
        raise ValueError(f"No adapter found for {data_type}")
```

### 2. ViewAdapter Interface

```python
class ViewAdapter(ABC):
    """Base adapter for data transformation"""
    
    @abstractmethod
    async def prepare(
        self,
        data: Any,
        x: Optional[List[str]] = None,
        y: Optional[List[str]] = None,
        c: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Prepare data for visualization"""
        pass
        
class RepoAdapter(ViewAdapter):
    """Adapter for Repo data sources"""
    
    async def prepare(
        self,
        repo: Repo,
        x: Optional[List[str]] = None,
        y: Optional[List[str]] = None,
        c: Optional[List[str]] = None
    ) -> pd.DataFrame:
        # Use InMemoryCache if available
        if hasattr(repo, 'cache') and repo.cache:
            df = await repo.cache.to_dataframe()
        else:
            # Fetch data with column optimization
            columns = list(filter(None, (x or []) + (y or []) + (c or [])))
            data = await repo.find(projection=columns if columns else None)
            df = pd.DataFrame(data)
        
        return df
        
class DataFrameAdapter(ViewAdapter):
    """Pass-through adapter for DataFrames"""
    
    async def prepare(
        self,
        df: pd.DataFrame,
        x: Optional[List[str]] = None,
        y: Optional[List[str]] = None,
        c: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return df
        
class DictAdapter(ViewAdapter):
    """Adapter for dict/list data"""
    
    async def prepare(
        self,
        data: Union[Dict, List],
        x: Optional[List[str]] = None,
        y: Optional[List[str]] = None,
        c: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if isinstance(data, dict):
            # Single record
            df = pd.DataFrame([data])
        else:
            # List of records
            df = pd.DataFrame(data)
        return df
```

### 3. Base Visualization Interface

```python
class ViewRenderer(ABC):
    """Base class for all view renderers with x,y,c support"""
    
    @abstractmethod
    async def render(
        self,
        data: pd.DataFrame,
        x: Optional[List[str]] = None,  # Dimensional/categorical fields
        y: Optional[List[str]] = None,  # Value/metric fields
        c: Optional[List[str]] = None,  # Color/grouping fields
        **kwargs  # Additional view-specific options
    ) -> Union[ConsoleOutput, WebOutput]:
        pass
```

### 4. Console Visualization (Rich Library)

#### 4.1 TableView
- **Purpose**: Display tabular data with formatting
- **Parameters**:
  - `x`: Columns to display as dimensions/identifiers (shown first)
  - `y`: Columns to display as metrics/values (shown after x columns)
  - `c`: Column to use for row coloring/highlighting based on categories
- **Features**:
  - Column sorting and filtering
  - Pagination for large datasets
  - Column width auto-adjustment
  - Conditional formatting (highlight cells based on values)
  - Export to CSV/Excel
- **Implementation**:
```python
class TableView(ViewRenderer):
    async def render(
        self,
        data: pd.DataFrame,
        x: Optional[List[str]] = None,  # Dimensional columns
        y: Optional[List[str]] = None,  # Metric columns
        c: Optional[List[str]] = None,  # Color/grouping column
        **kwargs  # sort_by, limit, color_map, highlight_rules, etc.
    ) -> ConsoleOutput:
        # If x,y not specified, show all columns
        # Order: x columns first, then y columns
        # Apply coloring based on c column values
        pass
```

#### 4.2 PivotView
- **Purpose**: Console-based pivot table operations
- **Parameters**:
  - `x`: Columns to use as row dimensions (index)
  - `y`: Columns to aggregate as values
  - `c`: Columns to use as column dimensions (columns in pivot)
- **Features**:
  - Multi-level grouping
  - Aggregation functions (sum, avg, count, min, max)
  - Row/column totals
  - Percentage calculations
  - Drill-down capabilities
- **Implementation**:
```python
class PivotView(ViewRenderer):
    async def render(
        self,
        data: pd.DataFrame,
        x: Optional[List[str]] = None,  # Row dimensions
        y: Optional[List[str]] = None,  # Values to aggregate
        c: Optional[List[str]] = None,  # Column dimensions
        **kwargs  # aggfunc, margins, fill_value, normalize, etc.
    ) -> ConsoleOutput:
        # Use pandas pivot_table:
        # pd.pivot_table(data, index=x, values=y, columns=c, **kwargs)
        pass
```

#### 4.3 TreeView
- **Purpose**: Hierarchical data display
- **Features**:
  - Expandable/collapsible nodes
  - Search within tree
  - Lazy loading for large trees
  - Custom node formatting

#### 4.4 ProgressView
- **Purpose**: Track long-running operations
- **Features**:
  - Multiple progress bars
  - ETA calculations
  - Nested progress tracking
  - Integration with Repo operations

#### 4.5 StatsView
- **Purpose**: Statistical summaries
- **Features**:
  - Distribution histograms
  - Box plots in ASCII
  - Correlation matrices
  - Summary statistics

### 5. Server-Side Visualization

#### 5.1 Chart Interface
```python
class ChartRenderer(ViewRenderer):
    """Base for all chart renderers"""
    
    async def render(
        self,
        data: pd.DataFrame,
        x: Optional[List[str]] = None,  # X-axis fields
        y: Optional[List[str]] = None,  # Y-axis fields  
        c: Optional[List[str]] = None,  # Color/category fields
        **kwargs  # title, mode, orientation, etc.
    ) -> PlotlyOutput:
        pass
```

#### 5.2 Supported Chart Types

##### Bar Chart
- **Parameters**: x (categories), y (values), c (color grouping)
- **Features**: 
  - Grouped bars (side-by-side comparison)
  - Stacked bars (part-to-whole with totals)
  - 100% stacked (percentage distribution)
  - Horizontal/vertical orientation
  - Error bars support
  - Custom bar ordering
- **Use Cases**: Category comparisons, distributions, part-to-whole analysis

##### Line Chart
- **Parameters**: x (time/sequence), y (values), c (series)
- **Features**: Multiple series, annotations, trend lines
- **Use Cases**: Time series, trends, comparisons

##### Pie Chart
- **Parameters**: x (labels), y (values)
- **Features**: Donut variant, percentages, exploded slices
- **Use Cases**: Part-to-whole relationships

##### Gantt Chart
- **Parameters**: x (task names), y (start/end dates), c (categories)
- **Features**: Dependencies, milestones, resource allocation
- **Use Cases**: Project timelines, scheduling

##### Sankey Diagram
- **Parameters**: x (source nodes), y (target nodes), c (flow values)
- **Features**: Multi-level flows, custom colors
- **Use Cases**: Flow analysis, conversions, relationships

#### 5.3 PivotEngine
- **Purpose**: Server-side pivot operations
- **Features**:
  - Multi-dimensional pivoting
  - Custom aggregation functions
  - Calculated fields
  - Export to Excel with formatting
  - Pivot caching for performance

### 6. Data Pipeline

#### 6.1 ViewDataAdapter
```python
class ViewDataAdapter:
    """Converts Repo data to pandas DataFrame efficiently"""
    
    async def to_dataframe(
        self,
        repo: Repo,
        query: Optional[Query] = None,
        cache_key: Optional[str] = None
    ) -> pd.DataFrame:
        # Leverage InMemoryCache for repeated operations
        pass
```

#### 6.2 Cache Strategy
- **InMemoryCache**: Fast pandas operations
- **Cache Keys**: Based on query + repo signature
- **TTL**: Configurable time-to-live
- **Size Limits**: Automatic eviction of old data

#### 6.3 Aggregation Engine
- **Operations**: groupby, pivot, melt, merge
- **Performance**: Chunked processing for large data
- **Memory**: Streaming aggregations when needed

## Integration Plan

### With Repo
- Direct access to Repo.cache for InMemoryCache
- Query optimization before DataFrame conversion
- Lazy loading for large datasets
- Schema awareness from Repo metadata

### With Inspector
- Use schema info for appropriate chart selection
- Data type validation before visualization
- Automatic format detection (dates, numbers, categories)

### With Models
- Read UI widget hints from model definitions
- Default visualization based on field types
- Permission-aware rendering

## Usage Examples (Planned)

### Factory Initialization
```python
# Initialize factory and register components
factory = XViewFactory()

# Register views
factory.register_view("table", TableView)
factory.register_view("pivot", PivotView)
factory.register_view("bar", BarChart)
factory.register_view("line", LineChart)
factory.register_view("gantt", GanttChart)

# Register adapters
factory.register_adapter(Repo, RepoAdapter)
factory.register_adapter(pd.DataFrame, DataFrameAdapter)
factory.register_adapter(dict, DictAdapter)
factory.register_adapter(list, DictAdapter)
```

### Console Table
```python
# Simple table view with factory
await factory.create_view(
    "table",
    user_repo,
    x=["user_id", "username", "email"],  # Identifiers
    y=["created_at", "last_login"],      # Metrics
    limit=50,
    sort_by="created_at"
)

# Table with color grouping
await factory.create_view(
    "table",
    transaction_repo,
    x=["transaction_id", "user_id"],
    y=["amount", "fee", "net_amount"],
    c=["status"],  # Color by status
    filter={"amount": {">": 1000}},
    color_map={"success": "green", "failed": "red", "pending": "yellow"}
)
```

### Pivot Table
```python
# Sales pivot with x,y,c parameters
await factory.create_view(
    "pivot",
    sales_repo,
    x=["region", "product"],         # Row dimensions
    y=["amount", "quantity"],         # Values to aggregate
    c=["year", "quarter"],           # Column dimensions
    aggfunc={"amount": "sum", "quantity": "avg"},
    margins=True,
    margins_name="Total"
)

# Employee stats pivot
await factory.create_view(
    "pivot",
    employee_repo,
    x=["department"],                 # Rows
    y=["salary", "bonus"],           # Values
    c=["role", "location"],          # Columns
    aggfunc={"salary": ["mean", "median"], "bonus": "sum"},
    fill_value=0
)
```

### Charts
```python
# Bar chart - grouped
await factory.create_view(
    "bar",
    sales_data,
    x=["product"],
    y=["revenue", "cost"],
    c=["region"],
    title="Sales by Product and Region",
    bar_mode="group"  # side-by-side bars
)

# Bar chart - stacked
await factory.create_view(
    "bar", 
    sales_data,
    x=["product"],
    y=["q1_sales", "q2_sales", "q3_sales", "q4_sales"],
    c=["quarter"],
    title="Quarterly Sales by Product",
    bar_mode="stack"  # stacked bars showing total
)

# Line chart with multiple series
await factory.create_view(
    "line",
    timeseries_data,
    x=["date"],
    y=["value", "forecast"],
    c=["category"],  # Multiple lines by category
    title="Trends Over Time",
    show_legend=True
)

# Gantt chart
await factory.create_view(
    "gantt",
    project_data,
    x=["task_name"],
    y=["start_date", "end_date"],
    c=["team"],
    title="Project Timeline",
    show_dependencies=True
)
```

### Working with Different Data Sources
```python
# From Repo
await factory.create_view("table", user_repo, x=["name"], y=["score"])

# From DataFrame
df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95, 87]})
await factory.create_view("table", df, x=["name"], y=["score"])

# From Dict/List
data = [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]
await factory.create_view("table", data, x=["name"], y=["score"])
```

## Performance Considerations

1. **Caching Strategy**
   - Cache pandas DataFrames for repeated visualizations
   - Incremental updates for real-time data
   - Memory limits with LRU eviction

2. **Large Dataset Handling**
   - Pagination for console output
   - Data sampling for initial views
   - Progressive loading for web charts

3. **Async Operations**
   - All rendering methods are async
   - Concurrent data fetching
   - Non-blocking UI updates

## Security Considerations

1. **Data Access**
   - Respect Repo permissions
   - Filter data based on user context
   - Audit visualization access

2. **Export Controls**
   - Configurable export permissions
   - Watermarking for sensitive data
   - Format restrictions

## Future Enhancements

1. **Real-time Updates**
   - WebSocket support for live charts
   - Streaming console updates
   - Delta-based refreshes

2. **Advanced Analytics**
   - ML-based chart recommendations
   - Automated insight detection
   - Anomaly highlighting

3. **Export Formats**
   - PDF reports with charts
   - PowerPoint generation
   - Interactive HTML dashboards

## Dependencies

- **Console**: Rich (for terminal UI)
- **Charts**: Plotly (for web charts)
- **Data**: Pandas (for data manipulation)
- **Export**: XlsxWriter (for Excel export)
- **Cache**: Built on Repo's InMemoryCache

## Testing Strategy

1. **Unit Tests**
   - Each renderer tested independently
   - Mock data for consistent results
   - Performance benchmarks

2. **Integration Tests**
   - Full pipeline from Repo to output
   - Large dataset handling
   - Cache behavior validation

3. **Visual Tests**
   - Screenshot comparison for charts
   - Console output validation
   - Cross-platform rendering