"""
Brick Schema Utilities
Provides Brick-specific operations analogous to SPINACH's Wikidata utilities.
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
import json
from functools import lru_cache


# Brick Schema Namespaces
BRICK = Namespace("https://brickschema.org/schema/Brick#")
BLDG = Namespace("bldg-59#")
REF = Namespace("https://brickschema.org/schema/Brick/ref#")

# Standard prefixes for Brick SPARQL queries
BRICK_PREFIXES = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX bldg: <bldg-59#>
PREFIX ref: <https://brickschema.org/schema/Brick/ref#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


class SparqlExecutionStatus(Enum):
    """Status codes for SPARQL execution"""
    SUCCESS = "success"
    EMPTY_RESULT = "empty_result"
    SYNTAX_ERROR = "syntax_error"
    TIMED_OUT = "timed_out"
    OTHER_ERROR = "other_error"

    def get_message(self) -> str:
        messages = {
            self.SUCCESS: "Query executed successfully",
            self.EMPTY_RESULT: "Query returned no results",
            self.SYNTAX_ERROR: "Query has syntax error",
            self.TIMED_OUT: "Query execution timed out",
            self.OTHER_ERROR: "Query execution encountered an error"
        }
        return messages.get(self, "Unknown status")


class BrickGraph:
    """
    Manages the Brick schema graph and provides query capabilities.
    Singleton pattern to maintain one graph instance.
    """
    _instance = None
    _graph = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BrickGraph, cls).__new__(cls)
        return cls._instance

    def initialize(self, ttl_file: str = None, graph: Graph = None):
        """
        Initialize the Brick graph from TTL file or existing graph.

        Args:
            ttl_file: Path to Turtle file containing Brick schema
            graph: Existing rdflib Graph object
        """
        if self._graph is not None:
            print("Warning: BrickGraph already initialized. Skipping.")
            return

        if graph is not None:
            self._graph = graph
        elif ttl_file:
            self._graph = Graph()
            self._graph.parse(ttl_file, format="turtle")
            print(f"Loaded {len(self._graph)} triples from {ttl_file}")
        else:
            self._graph = Graph()
            print("Initialized empty Brick graph")

        # Bind common namespaces
        self._graph.bind("brick", BRICK)
        self._graph.bind("bldg", BLDG)
        self._graph.bind("ref", REF)

    def get_graph(self) -> Graph:
        """Get the underlying RDFLib graph"""
        if self._graph is None:
            raise RuntimeError("BrickGraph not initialized. Call initialize() first.")
        return self._graph

    def add_timeseries_data(self, csv_file: str, max_rows: int = 100):
        """
        Load timeseries data from CSV into the graph.

        Args:
            csv_file: Path to CSV file
            max_rows: Maximum rows to load (loads LAST N rows to get recent data)
        """
        import pandas as pd

        # Load last N rows to get most recent data
        df_full = pd.read_csv(csv_file)
        df = df_full.tail(max_rows)
        print(f"Loading last {len(df)} rows from {csv_file} (total: {len(df_full)} rows)")
        print(f"Date range: {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}")

        for idx, row in df.iterrows():
            timestamp_str = row['Datetime']

            # Convert timestamp from "12/31/2018 23:59" to ISO format for proper sorting
            from datetime import datetime
            dt = datetime.strptime(timestamp_str, "%m/%d/%Y %H:%M")
            timestamp_iso = dt.strftime("%Y-%m-%dT%H:%M:%S")

            for column_name in df.columns:
                if column_name == 'Datetime':
                    continue

                value = row[column_name]
                if pd.isna(value):
                    continue

                obs_uri = BLDG[f"obs_{column_name}_{idx}"]
                sensor_uri = BLDG[column_name]

                self._graph.add((sensor_uri, REF.hasObservation, obs_uri))
                self._graph.add((obs_uri, REF.hasTimestamp, Literal(timestamp_iso, datatype=XSD.dateTime)))
                self._graph.add((obs_uri, REF.hasValue, Literal(float(value), datatype=XSD.float)))

        print(f"Added timeseries data. Total triples: {len(self._graph)}")

    def save_graph(self, output_file: str = "brick_graph_cache.ttl"):
        """
        Save the graph to a file for quick reloading.

        Args:
            output_file: Path to save the graph (default: brick_graph_cache.ttl)
        """
        if self._graph is None:
            raise RuntimeError("BrickGraph not initialized. Nothing to save.")

        print(f"💾 Saving graph with {len(self._graph)} triples to {output_file}...")
        self._graph.serialize(destination=output_file, format='turtle')
        print(f"✅ Graph saved successfully!")

    def load_from_cache(self, cache_file: str = "brick_graph_cache.ttl") -> bool:
        """
        Load graph from a cached file instead of rebuilding from CSV.

        Args:
            cache_file: Path to cached graph file

        Returns:
            True if loaded successfully, False otherwise
        """
        import os

        if self._graph is not None:
            print("Warning: BrickGraph already initialized. Skipping cache load.")
            return True

        if not os.path.exists(cache_file):
            print(f"❌ Cache file {cache_file} not found. Will need to build graph from scratch.")
            return False

        print(f"📂 Loading cached graph from {cache_file}...")
        self._graph = Graph()
        self._graph.parse(cache_file, format="turtle")

        # Bind common namespaces
        self._graph.bind("brick", BRICK)
        self._graph.bind("bldg", BLDG)
        self._graph.bind("ref", REF)

        print(f"✅ Loaded {len(self._graph)} triples from cache!")
        return True

    @classmethod
    def reset_singleton(cls):
        """Reset the graph singleton (useful for rebuilding cache)"""
        if cls._instance is not None:
            cls._instance._graph = None
            cls._graph = None
            print("🔄 Graph singleton reset")

    def reset(self):
        """Reset the graph (useful for rebuilding cache)"""
        BrickGraph._graph = None
        self._graph = None
        print("🔄 Graph reset")

    def get_date_range(self) -> tuple:
        """
        Get the date range of the timeseries data in the graph.

        Returns:
            Tuple of (min_date, max_date) as strings in YYYY-MM-DD format, or (None, None) if no data
        """
        if self._graph is None:
            return (None, None)

        query = f"""
        {BRICK_PREFIXES}
        SELECT (MIN(?timestamp) AS ?min_date) (MAX(?timestamp) AS ?max_date)
        WHERE {{
            ?obs ref:hasTimestamp ?timestamp .
        }}
        """

        try:
            results = list(self._graph.query(query))
            if results and results[0]:
                min_date = results[0][0]
                max_date = results[0][1]

                if min_date and max_date:
                    # Extract just the date part (YYYY-MM-DD)
                    min_str = str(min_date)[:10] if len(str(min_date)) >= 10 else str(min_date)
                    max_str = str(max_date)[:10] if len(str(max_date)) >= 10 else str(max_date)
                    return (min_str, max_str)
        except Exception as e:
            print(f"Warning: Could not determine date range: {e}")

        return (None, None)


def execute_sparql(query: str, return_status: bool = False, timeout: int = 30):
    """
    Execute a SPARQL query on the Brick graph.

    Args:
        query: SPARQL query string
        return_status: If True, return (results, status) tuple
        timeout: Query timeout in seconds

    Returns:
        Query results or (results, status) tuple if return_status=True
    """
    try:
        graph = BrickGraph().get_graph()

        # Add prefixes if not present
        if "PREFIX" not in query.upper():
            query = BRICK_PREFIXES + query

        results = graph.query(query)

        # Convert results to list of dicts
        result_list = []
        for row in results:
            row_dict = {}
            for var in results.vars:
                value = row[var]
                if value is not None:
                    row_dict[str(var)] = {"value": str(value)}
            if row_dict:
                result_list.append(row_dict)

        status = SparqlExecutionStatus.SUCCESS if result_list else SparqlExecutionStatus.EMPTY_RESULT

        if return_status:
            return result_list, status
        return result_list

    except Exception as e:
        error_msg = str(e).lower()
        if "syntax" in error_msg or "parse" in error_msg:
            status = SparqlExecutionStatus.SYNTAX_ERROR
        elif "timeout" in error_msg:
            status = SparqlExecutionStatus.TIMED_OUT
        else:
            status = SparqlExecutionStatus.OTHER_ERROR

        print(f"SPARQL execution error: {e}")

        if return_status:
            return [], status
        return []


@lru_cache(maxsize=1000)
def search_brick(search_term: str, limit: int = 8, search_type: str = "all") -> List[Dict]:
    """
    Search for Brick entities (sensors, equipment, points) matching the search term.

    Args:
        search_term: Term to search for
        limit: Maximum number of results
        search_type: 'sensor', 'equipment', 'point', or 'all'

    Returns:
        List of dicts with 'id', 'label', 'type', 'description'
    """
    graph = BrickGraph().get_graph()
    results = []

    search_lower = search_term.lower()

    # Query for entities matching the search term
    query = f"""
    SELECT DISTINCT ?entity ?type ?label
    WHERE {{
        ?entity rdf:type ?type .
        OPTIONAL {{ ?entity rdfs:label ?label }}
        FILTER(CONTAINS(LCASE(STR(?entity)), "{search_lower}") ||
               CONTAINS(LCASE(STR(?type)), "{search_lower}") ||
               CONTAINS(LCASE(STR(?label)), "{search_lower}"))
    }}
    LIMIT {limit}
    """

    try:
        qresults = graph.query(BRICK_PREFIXES + query)

        for row in qresults:
            entity_uri = str(row.entity)
            entity_type = str(row.type)
            label = str(row.label) if row.label else entity_uri.split('#')[-1]

            # Extract local name
            local_name = entity_uri.split('#')[-1]

            results.append({
                "id": local_name,
                "uri": entity_uri,
                "label": label,
                "type": entity_type.split('#')[-1],
                "description": f"A {entity_type.split('#')[-1]} in the building"
            })

    except Exception as e:
        print(f"Search error: {e}")

    return results[:limit]


@lru_cache(maxsize=1000)
def get_brick_entity(entity_id: str, compact: bool = True) -> Dict:
    """
    Get all outgoing edges (properties and relationships) for a Brick entity.

    Args:
        entity_id: Entity identifier (e.g., 'RM_TEMP', 'FCU_OAT')
        compact: If True, return simplified format

    Returns:
        Dict with entity properties and relationships
    """
    graph = BrickGraph().get_graph()

    # Construct entity URI
    if entity_id.startswith("http"):
        entity_uri = URIRef(entity_id)
    else:
        entity_uri = BLDG[entity_id]

    # Get all outgoing edges
    properties = {}

    for pred, obj in graph.predicate_objects(subject=entity_uri):
        pred_name = str(pred).split('#')[-1]
        obj_str = str(obj)

        if pred_name not in properties:
            properties[pred_name] = []

        # Format object based on type
        if isinstance(obj, Literal):
            properties[pred_name].append({
                "value": obj_str,
                "datatype": str(obj.datatype) if obj.datatype else "string"
            })
        else:
            obj_local = obj_str.split('#')[-1]
            properties[pred_name].append({
                "value": obj_local,
                "uri": obj_str,
                "type": "uri"
            })

    # Get entity type
    entity_types = []
    for obj in graph.objects(subject=entity_uri, predicate=RDF.type):
        entity_types.append(str(obj).split('#')[-1])

    result = {
        "entity": entity_id,
        "uri": str(entity_uri),
        "types": entity_types,
        "properties": properties
    }

    return result


def get_property_examples(property_name: str, limit: int = 5) -> List[Tuple[str, str, str]]:
    """
    Get examples of how a property is used in the Brick graph.

    Args:
        property_name: Property name (e.g., 'hasObservation', 'feeds')
        limit: Number of examples to return

    Returns:
        List of (subject, property, object) tuples
    """
    graph = BrickGraph().get_graph()
    examples = []

    # Construct property URI
    if property_name.startswith("http"):
        prop_uri = URIRef(property_name)
    elif ":" in property_name:
        # Already has namespace prefix
        prop_uri = property_name
    else:
        # Try REF namespace first, then BRICK
        prop_uri = REF[property_name]

    query = f"""
    SELECT ?subject ?object
    WHERE {{
        ?subject <{prop_uri}> ?object .
    }}
    LIMIT {limit}
    """

    try:
        results = graph.query(query)

        for row in results:
            subj = str(row.subject).split('#')[-1]
            obj = str(row.object).split('#')[-1]
            examples.append((subj, property_name, obj))

    except Exception as e:
        print(f"Error getting property examples: {e}")

    return examples


def get_all_sensor_types() -> List[str]:
    """
    Get all sensor types available in the Brick schema.

    Returns:
        List of sensor type names
    """
    graph = BrickGraph().get_graph()

    query = """
    SELECT DISTINCT ?sensorType
    WHERE {
        ?sensor rdf:type ?sensorType .
        FILTER(CONTAINS(STR(?sensorType), "Sensor") || CONTAINS(STR(?sensorType), "Point"))
    }
    """

    sensor_types = []
    results = graph.query(BRICK_PREFIXES + query)

    for row in results:
        sensor_type = str(row.sensorType).split('#')[-1]
        sensor_types.append(sensor_type)

    return sorted(sensor_types)


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results for display to LLM.

    Args:
        results: List of search result dicts

    Returns:
        Formatted string
    """
    if not results:
        return "No results found."

    lines = []
    for item in results:
        label = item.get("label", item.get("id", ""))
        id_ = item.get("id", "")
        type_ = item.get("type", "")
        desc = item.get("description", "")

        lines.append(f"{label} ({id_}): {desc}")
        if type_:
            lines.append(f"  Type: {type_}")

    return "\n".join(lines)


def format_entity_info(entity_info: Dict) -> str:
    """
    Format entity information for display to LLM.

    Args:
        entity_info: Entity info dict from get_brick_entity()

    Returns:
        Formatted string
    """
    lines = [f"Entity: {entity_info['entity']}"]

    if entity_info.get('types'):
        lines.append(f"Types: {', '.join(entity_info['types'])}")

    lines.append("\nProperties:")
    for prop_name, values in entity_info.get('properties', {}).items():
        lines.append(f"  {prop_name}:")
        for val in values[:3]:  # Limit to first 3 values
            if val.get('type') == 'uri':
                lines.append(f"    -> {val['value']}")
            else:
                lines.append(f"    = {val['value']}")

    return "\n".join(lines)
