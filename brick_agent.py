"""
Brick Agent
An iterative framework for converting natural language questions to Brick SPARQL queries.
Uses a part-to-whole approach to build queries incrementally.
"""

import sys
import io
import time

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        pass  # stdout already configured or not available

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

from brick_utils import (
    BrickGraph,
    execute_sparql,
    search_brick,
    get_brick_entity,
    get_property_examples,
    format_search_results,
    format_entity_info,
    SparqlExecutionStatus,
    BRICK_PREFIXES
)
from brick_decomposer import BrickQueryDecomposer, QueryDecomposition
from brick_temporal_handler import BrickTemporalHandler, TemporalApplication



@dataclass
class BrickAction:
    """
    Represents a single action in the reasoning chain.
    Follows the Thought-Action-Observation pattern.
    """
    thought: str
    action_name: str
    action_argument: str
    observation: Optional[str] = None
    observation_raw: Optional[any] = None

    POSSIBLE_ACTIONS = [
        "search_brick",           # Search for entities/sensors
        "get_brick_entity",       # Explore entity properties
        "get_property_examples",  # See how properties are used
        "execute_sparql",         # Test SPARQL query
        "stop"                    # Finalize answer
    ]

    def __post_init__(self):
        if self.action_name not in self.POSSIBLE_ACTIONS:
            raise ValueError(f"Invalid action: {self.action_name}. Must be one of {self.POSSIBLE_ACTIONS}")

    def to_string(self, include_observation: bool = True) -> str:
        """Format action for display"""
        result = f"Thought: {self.thought}\n"
        result += f"Action: {self.action_name}({self.action_argument})\n"
        if include_observation and self.observation:
            result += f"Observation: {self.observation}\n"
        return result

    def __eq__(self, other):
        """Check if two actions are duplicates"""
        if not isinstance(other, BrickAction):
            return False
        return (self.action_name == other.action_name and
                self.action_argument == other.action_argument)

    def __hash__(self):
        return hash((self.action_name, self.action_argument))


@dataclass
class BrickSparqlQuery:
    """Represents a SPARQL query with its execution results"""
    sparql: str
    execution_result: Optional[List[Dict]] = None
    execution_status: Optional[SparqlExecutionStatus] = None

    def execute(self):
        """Execute the SPARQL query"""
        self.execution_result, self.execution_status = execute_sparql(
            self.sparql,
            return_status=True
        )

    def has_results(self) -> bool:
        """Check if query returned results"""
        return bool(self.execution_result)

    def results_as_table(self, max_rows: int = 10) -> str:
        """Format results as a text table"""
        if not self.execution_result:
            return "No results"

        # Convert to DataFrame for nice formatting
        formatted_data = []
        for row in self.execution_result:
            formatted_row = {}
            for key, val in row.items():
                formatted_row[key] = val.get('value', str(val))
            formatted_data.append(formatted_row)

        df = pd.DataFrame(formatted_data)

        if len(df) > max_rows:
            # Show first and last rows
            half = max_rows // 2
            top = df.head(half)
            bottom = df.tail(max_rows - half)
            result = top.to_string(index=False) + f"\n... ({len(df) - max_rows} rows omitted) ...\n" + bottom.to_string(index=False, header=False)
        else:
            result = df.to_string(index=False)

        return result


@dataclass
class AgentState:
    """Maintains the state of the Brick agent"""
    question: str
    actions: List[BrickAction] = field(default_factory=list)
    generated_sparqls: List[BrickSparqlQuery] = field(default_factory=list)
    final_sparql: Optional[BrickSparqlQuery] = None
    max_iterations: int = 15
    decomposition: Optional[QueryDecomposition] = None  # Query decomposition results

    def get_action_history(self, last_n: int = 10, include_observation: bool = True) -> str:
        """Get formatted action history for context"""
        recent_actions = self.actions[-last_n:] if len(self.actions) > last_n else self.actions

        history = []
        for i, action in enumerate(recent_actions):
            # Skip observations for search/get actions in the middle
            should_include_obs = include_observation
            if i < len(recent_actions) - 2 and action.action_name in ["search_brick", "get_brick_entity"]:
                should_include_obs = False

            history.append(action.to_string(include_observation=should_include_obs))

        return "\n".join(history)

    def is_duplicate_action(self, action: BrickAction) -> bool:
        """Check if action is a duplicate of recent actions"""
        return action in self.actions[-5:]

    def should_stop(self) -> bool:
        """Check if agent should stop (max iterations or stop action)"""
        if len(self.actions) >= self.max_iterations:
            return True
        if self.actions and self.actions[-1].action_name == "stop":
            return True
        return False


class BrickAgent:
    """
    Main agent class that implements the iterative approach for Brick SPARQL generation.
    """

    def __init__(self, engine: str = "gemini-flash", use_decomposer: bool = True, use_temporal_handler: bool = True, use_fewshot: bool = False, ttl_schema_file: str = None, gcp_project_id: str = None, gcp_location: str = "us-central1"):
        """
        Initialize the agent.

        Args:
            engine: LLM engine to use (e.g., 'gemini-flash', 'gemini-pro', 'gemini-flash-lite')
            use_decomposer: If True, use query decomposer as preprocessing step
            use_temporal_handler: If True, apply temporal constraints separately after SPARQL generation
            use_fewshot: If True, include few-shot examples in the controller prompt
            ttl_schema_file: Path to .ttl schema file to include in prompt (optional)
            gcp_project_id: Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            gcp_location: Google Cloud location (default: us-central1)
        """
        import os

        self.engine = engine
        self.brick_graph = BrickGraph()
        self.use_decomposer = use_decomposer
        self.use_temporal_handler = use_temporal_handler
        self.use_fewshot = use_fewshot
        self.ttl_schema_file = ttl_schema_file
        self.ttl_schema_content = None

        # Get GCP project ID from parameter or environment variable
        self.gcp_project_id = gcp_project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.gcp_location = gcp_location

        if not self.gcp_project_id:
            print("⚠️  WARNING: No GCP project ID specified. Set GOOGLE_CLOUD_PROJECT environment variable or pass gcp_project_id parameter.")

        # Initialize decomposer
        if use_decomposer:
            self.decomposer = BrickQueryDecomposer(project=self.gcp_project_id)
            print("✅ Query decomposer initialized")
        else:
            self.decomposer = None
            print("⚠️ Running without query decomposer")

        # Initialize temporal handler
        if use_temporal_handler:
            self.temporal_handler = BrickTemporalHandler()
            print("✅ Temporal handler initialized")
        else:
            self.temporal_handler = None
            print("⚠️ Running without temporal handler")

        # Load few-shot examples if enabled
        if use_fewshot:
            try:
                with open("prompts/brick_controller_fewshot_examples.txt", "r", encoding="utf-8") as f:
                    self.fewshot_examples = f.read()
                print("✅ Few-shot examples loaded")
            except Exception as e:
                print(f"⚠️ Failed to load few-shot examples: {e}")
                self.fewshot_examples = None
        else:
            self.fewshot_examples = None
            print("⚠️ Running without few-shot examples")

        # Load TTL schema content if provided
        if ttl_schema_file:
            try:
                with open(ttl_schema_file, "r", encoding="utf-8") as f:
                    self.ttl_schema_content = f.read()
                print(f"✅ TTL schema loaded from {ttl_schema_file} ({len(self.ttl_schema_content)} chars)")
            except Exception as e:
                print(f"⚠️ Failed to load TTL schema: {e}")
                self.ttl_schema_content = None
        else:
            print("⚠️ Running without TTL schema in prompt")

        # Create timestamped log file for this agent instance
        from datetime import datetime
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f"prompt_logs_{timestamp}.jsonl")
        print(f"📝 LLM logs will be saved to: {self.log_file}")

        # Dataset metadata (populated during initialization)
        self.dataset_date_range = (None, None)
        self.dataset_year = None

    def initialize_graph(self, ttl_file: str = None, csv_file: str = None, max_csv_rows: int = 100, use_cache: bool = True):
        """
        Initialize the Brick graph with data.

        Args:
            ttl_file: Path to Turtle file with Brick schema
            csv_file: Path to CSV file with timeseries data
            max_csv_rows: Maximum rows to load from CSV
            use_cache: If True, try to load from cache first (default: True)
        """
        import os

        # Generate cache filename based on parameters
        cache_file = f"brick_graph_cache_{max_csv_rows}rows.ttl"

        # Try to load from cache if enabled
        if use_cache and os.path.exists(cache_file):
            if self.brick_graph.load_from_cache(cache_file):
                print(f"⚡ Loaded graph from cache in seconds (skipped CSV processing)!")
                # Get date range after loading
                self._extract_date_metadata()
                return

        # Cache not available or disabled, build from scratch
        print("🔨 Building graph from scratch...")
        self.brick_graph.initialize(ttl_file=ttl_file)

        if csv_file:
            self.brick_graph.add_timeseries_data(csv_file, max_rows=max_csv_rows)

        # Always save to cache after building (even if use_cache was False for forcing rebuild)
        print(f"\n💾 Saving graph to cache for faster loading next time...")
        self.brick_graph.save_graph(cache_file)
        print(f"💡 Next time, graph will load instantly from {cache_file}!")

        # Get date range after building
        self._extract_date_metadata()

    def _extract_date_metadata(self):
        """Extract and store date metadata from the dataset"""
        # Hardcoded for LBNL FCU dataset (2018)
        self.dataset_year = "2018"
        self.dataset_date_range = ("2018-01-01", "2018-12-31")

        print(f"📅 Dataset year: {self.dataset_year}")

        # Update temporal handler with the default year
        if self.temporal_handler:
            self.temporal_handler.default_year = self.dataset_year

    def controller(self, state: AgentState) -> BrickAction:
        """
        The controller decides the next action based on the question and history.
        This is the "brain" of the agent.

        Args:
            state: Current agent state

        Returns:
            Next action to take
        """

        try:
            # Use Vertex AI directly instead of chainlite to avoid async issues
            from vertexai.generative_models import GenerativeModel
            import vertexai
            from datetime import datetime
            import time

            # Rate limiting: Add delay to avoid 429 errors
            time.sleep(6.5)  # 6.5 second delay between API calls (max 9 calls/min for 10/min quota)

            # Initialize Vertex AI
            if not self.gcp_project_id:
                raise ValueError("GCP project ID not configured. Set GOOGLE_CLOUD_PROJECT environment variable or pass gcp_project_id to BrickAgent constructor.")
            vertexai.init(project=self.gcp_project_id, location=self.gcp_location)

            # Build prompt from template
            action_history_str = "\n\n".join([
                action.to_string(include_observation=True) for action in state.actions
            ])

            # Read prompt template
            with open("prompts/brick_controller.prompt", "r", encoding="utf-8") as f:
                template = f.read()

            # Add TTL schema content if available
            ttl_schema_str = ""
            if self.ttl_schema_content:
                ttl_schema_str = f"""
COMPLETE BRICK SCHEMA (Turtle/TTL Format):
------------------------------------
The following is the complete Brick schema for this building in Turtle format.
Use this to understand the exact structure, entities, relationships, and available sensors.

{self.ttl_schema_content}
------------------------------------

"""

            # Add temporal patterns guide if there's temporal constraint
            temporal_guide_str = ""
            using_temporal_guide = False
            if state.decomposition and state.decomposition.temporal.has_constraint:
                try:
                    with open("prompts/brick_temporal_patterns.txt", "r", encoding="utf-8") as f:
                        temporal_guide_str = "\n" + f.read() + "\n"
                        using_temporal_guide = True
                        print(f"[INFO] Using temporal patterns guide for {state.decomposition.temporal.type} constraint")
                except FileNotFoundError:
                    print("[WARNING] brick_temporal_patterns.txt not found, skipping temporal guide")

            # Add decomposition info to prompt if available
            decomposition_str = ""
            if state.decomposition:
                decomp = state.decomposition
                temporal_note = ""
                if decomp.temporal.has_constraint:
                    temporal_note = "\n  Follow the temporal patterns guide above to generate correct temporal SPARQL."

                decomposition_str = f"""
Query Decomposition (use this to guide your actions):
- Sensors needed ({decomp.sensor_count}): {', '.join([s.sensor_id for s in decomp.sensors])}
- Temporal constraint: {decomp.temporal.type}{temporal_note}
- Aggregation: {', '.join(decomp.aggregation.operations) if decomp.aggregation.required else 'None'}
- Intent: {decomp.query_intent}

"""

            # Simple template rendering (replace variables)
            prompt = template.replace("{{ question }}", state.question)

            # Store flag for execute_action to know if temporal guide is being used
            self._using_temporal_guide = using_temporal_guide

            # Insert TTL schema, temporal guide and decomposition info
            insert_str = ""
            if ttl_schema_str:
                insert_str += ttl_schema_str
            if temporal_guide_str:
                insert_str += temporal_guide_str
            if decomposition_str:
                insert_str += decomposition_str

            # Insert before action history
            if insert_str:
                # Add after the question, before action history
                prompt = prompt.replace("{% if action_history %}", insert_str + "\n{% if action_history %}")

            prompt = prompt.replace("{% if action_history %}", "")
            prompt = prompt.replace("{% for i in range(0, action_history|length) %}", "")
            prompt = prompt.replace("{{ action_history[i] }}", action_history_str)
            prompt = prompt.replace("{% endfor %}", "")
            prompt = prompt.replace("{% endif %}", "")
            prompt = prompt.replace("{% for i in range(0, conversation_history|length) %}", "")
            prompt = prompt.replace("{% endfor %}", "")

            # Call Gemini
            model = GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            time.sleep(60)  # 1-minute cooldown after each Vertex AI API call to avoid quota limits

            # Handle multiple content parts (LLM sometimes duplicates output)
            try:
                controller_output = response.text
            except ValueError as e:
                if "Multiple content parts" in str(e):
                    # Extract text from first part only
                    print("[WARNING] LLM returned duplicate content parts, using first part only")
                    controller_output = response.candidates[0].content.parts[0].text
                else:
                    raise

            # Log to timestamped prompt_logs file
            self._log_to_jsonl(self.log_file, {
                "timestamp": datetime.now().isoformat(),
                "model": self.engine,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "response": controller_output
            })

            # Parse the response into BrickAction
            # Improved parsing to handle multiple formats and edge cases
            import re

            # Clean up the response - remove markdown code blocks
            cleaned_output = controller_output

            # Remove markdown code blocks (```sparql ... ```)
            cleaned_output = re.sub(r'```sparql\s*', '', cleaned_output)
            cleaned_output = re.sub(r'```', '', cleaned_output)

            # Stop at "Observation:" or "Output one" if present (LLM echoing prompt)
            stop_markers = ['Observation:', 'Output one "Thought"', 'Output one Thought']
            for marker in stop_markers:
                if marker in cleaned_output:
                    cleaned_output = cleaned_output.split(marker)[0]

            # Extract FIRST Thought and Action only (ignore multiple pairs)
            thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', cleaned_output, re.DOTALL)

            if thought_match:
                thought = thought_match.group(1).strip()
            else:
                # No explicit thought - check if output starts with Action directly
                if cleaned_output.strip().startswith('Action:'):
                    thought = "Proceeding with action"
                else:
                    # Try to extract any text before "Action:" as implicit thought
                    action_pos = cleaned_output.find('Action:')
                    if action_pos > 0:
                        thought = cleaned_output[:action_pos].strip()
                        if not thought:
                            thought = "Proceeding with action"
                    else:
                        raise ValueError(f"Could not find Thought or Action in output: {controller_output[:200]}...")
            # Try to extract action name and argument
            # Pattern 1: Action: action_name(argument)
            action_with_parens = re.search(r'Action:\s*(\w+)\s*\(', cleaned_output)

            if action_with_parens:
                action_name = action_with_parens.group(1).strip()

                # Find the argument by matching parentheses
                action_start = action_with_parens.end() - 1  # Position of opening (
                argument = self._extract_balanced_parens(cleaned_output[action_start:])

                if argument is None:
                    raise ValueError("Could not find matching parentheses for action argument")

                return BrickAction(
                    thought=thought,
                    action_name=action_name,
                    action_argument=argument.strip()
                )

            # Pattern 2: Action: just the SPARQL query (multiline, no action_name wrapper)
            action_content_match = re.search(r'Action:\s*(.+?)(?=\n\n|$)', cleaned_output, re.DOTALL)

            if action_content_match:
                action_content = action_content_match.group(1).strip()

                # If it starts with PREFIX, it's a SPARQL query
                if action_content.startswith('PREFIX') or action_content.startswith('SELECT'):
                    return BrickAction(
                        thought=thought,
                        action_name="execute_sparql",
                        action_argument=action_content
                    )

            # Could not parse
            raise ValueError(f"Could not parse LLM output. Output: {controller_output[:500]}...")

        except Exception as e:
            print(f"[ERROR] LLM controller failed: {e}")
            raise

    def _validate_and_fix_sparql(self, sparql: str) -> str:
        """
        Validate and auto-correct common SPARQL syntax errors.

        Args:
            sparql: SPARQL query string

        Returns:
            Corrected SPARQL query
        """
        import re

        original = sparql
        fixed = sparql.strip()

        # Fix 0a: Auto-prepend PREFIX declarations if missing
        # Check if query starts with SELECT but has no PREFIX declarations
        if not re.search(r'^\s*PREFIX\s+', fixed, re.IGNORECASE):
            print(f"[INFO] Query missing PREFIX declarations - auto-prepending standard prefixes")
            fixed = BRICK_PREFIXES.strip() + "\n\n" + fixed

        # Fix 0b: Fix incorrect PREFIX URIs (LLM sometimes generates wrong namespaces)
        # Common mistakes:
        # - BrickRef# instead of Brick/ref#
        # - Missing rdf: and rdfs: prefixes
        correct_prefixes = {
            'brick': 'https://brickschema.org/schema/Brick#',
            'bldg': 'bldg-59#',
            'ref': 'https://brickschema.org/schema/Brick/ref#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
        }

        for prefix_name, correct_uri in correct_prefixes.items():
            # Match any PREFIX declaration for this prefix name and replace with correct URI
            pattern = rf'PREFIX\s+{prefix_name}:\s+<[^>]+>'
            replacement = f'PREFIX {prefix_name}: <{correct_uri}>'
            if re.search(pattern, fixed, re.IGNORECASE):
                old_match = re.search(pattern, fixed, re.IGNORECASE).group(0)
                if correct_uri not in old_match:
                    print(f"[INFO] Correcting PREFIX {prefix_name}: from {old_match} to {replacement}")
                fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

        # Fix 0c: Missing angle brackets in PREFIX declarations
        # Pattern: PREFIX name: http... (without < >)
        # Replace with: PREFIX name: <http...>
        fixed = re.sub(
            r'PREFIX\s+(\w+):\s+(https?://[^\s<>]+)',
            r'PREFIX \1: <\2>',
            fixed,
            flags=re.IGNORECASE
        )

        # Fix 0d: Remove timezone 'Z' from timestamps
        # Pattern: "2018-12-28T14:00:00Z"^^xsd:dateTime
        # Should be: "2018-12-28T14:00:00"^^xsd:dateTime
        # The Z suffix causes comparison issues with xsd:dateTime
        before_z_fix = fixed
        fixed = re.sub(
            r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z"',
            r'"\1"',
            fixed
        )
        if fixed != before_z_fix:
            print(f"[INFO] Removed timezone 'Z' from timestamp literals (required for xsd:dateTime)")

        # Fix 1: DISABLED - Modern LLMs generate correct ORDER BY syntax
        # (This regex was causing issues by incorrectly "fixing" already-correct queries)
        # If you encounter ORDER BY DESC(?var without ), the LLM should be retrained
        pass

        # Fix 2: Add LIMIT if using ORDER BY DESC (for "latest" queries) and no LIMIT exists
        if re.search(r'ORDER\s+BY\s+DESC', fixed, re.IGNORECASE) and \
           not re.search(r'LIMIT\s+\d+', fixed, re.IGNORECASE):
            fixed += ' LIMIT 1'
            print(f"[INFO] Auto-added 'LIMIT 1' to query with ORDER BY DESC")

        # Fix 3: DISABLED - No longer auto-adding LIMIT 10
        # Users want full result sets without automatic limiting

        if fixed != original:
            print(f"[INFO] SPARQL auto-corrected:")
            print(f"  Before: {original[:100]}...")
            print(f"  After:  {fixed[:100]}...")

        return fixed

    def _extract_balanced_parens(self, text: str) -> Optional[str]:
        """
        Extract content between balanced parentheses.
        Handles nested parentheses like in SPARQL: DESC(?timestamp)

        Args:
            text: String starting with opening parenthesis

        Returns:
            Content between balanced parentheses (without outer parens), or None if unbalanced
        """
        if not text or text[0] != '(':
            return None

        depth = 0
        for i, char in enumerate(text):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    # Found matching closing paren
                    return text[1:i]  # Return content without outer parens

        return None  # Unbalanced parentheses

    def _log_to_jsonl(self, filename: str, data: dict):
        """Log data to a JSONL file"""
        import json
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"📝 Logged to {filename}")

    def _build_controller_prompt(self, context: Dict) -> str:
        """Build the prompt for the controller LLM"""
        # Build dataset context
        dataset_context = ""
        if self.dataset_year:
            dataset_context = f"""
IMPORTANT - Dataset Information:
- This dataset contains data from {self.dataset_date_range[0]} to {self.dataset_date_range[1]}
- If a date is mentioned WITHOUT a year (e.g., "November 15th", "June 1st"), assume year {self.dataset_year}
- Example: "November 15th at 2pm" should be interpreted as "{self.dataset_year}-11-15T14:00:00Z"
"""

        prompt = f"""Your task is to write a Brick SPARQL query to answer the given question. Follow a step-by-step process:

1. Start by constructing very simple fragments of the SPARQL query.
2. Execute each fragment to verify its correctness. Adjust as needed based on observations.
3. Confirm all your assumptions about the Brick schema structure before proceeding.
4. Gradually build the complete SPARQL query by adding one piece at a time.
5. Do NOT repeat the same action, as the results will be the same.
6. Continue until you find the answer.
{dataset_context}
REQUIRED SPARQL Prefixes and Schema:
------------------------------------
All SPARQL queries MUST start with these PREFIX definitions:
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX bldg: <bldg-59#>
PREFIX ref: <https://brickschema.org/schema/Brick/ref#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

Brick Schema Pattern:
- Sensors are in the bldg: namespace (e.g., bldg:RM_TEMP, bldg:FCU_OAT)
- Sensors have observations: ?sensor ref:hasObservation ?obs
- Observations have timestamps: ?obs ref:hasTimestamp ?timestamp
- Observations have values: ?obs ref:hasValue ?value
- Timestamps are xsd:dateTime type (format: "2018-12-28T14:00:00"^^xsd:dateTime)
  CRITICAL: Do NOT add timezone 'Z' suffix. Use "2018-12-28T14:00:00" NOT "2018-12-28T14:00:00Z"

Form exactly one "Thought" and perform exactly one "Action", then wait for the "Observation".

CRITICAL: Every SPARQL query you generate MUST begin with the PREFIX declarations above. Never omit them.

Possible actions:
- search_brick(string): Search for Brick entities (sensors, equipment) matching the string
- get_brick_entity(entity_id): Get all properties and relationships of a Brick entity
- get_property_examples(property_name): See examples of how a property is used
- execute_sparql(SPARQL): Run a SPARQL query and get results. IMPORTANT: SPARQL MUST start with PREFIX declarations (see REQUIRED SPARQL Prefixes above)
- stop(): Mark the last SPARQL query as final answer and end
"""

        # Add few-shot examples if enabled
        if self.fewshot_examples:
            prompt += f"""
Here are examples of Text-to-SPARQL conversions for reference:

{self.fewshot_examples}

"""

        prompt += f"""User Question: {context['question']}

{context['action_history']}

Output one "Thought" and one "Action":
"""
        return prompt

    def execute_action(self, action: BrickAction) -> BrickAction:
        """
        Execute an action and populate its observation.

        Args:
            action: Action to execute

        Returns:
            Action with observation filled
        """
        try:
            if action.action_name == "search_brick":
                results = search_brick(action.action_argument)
                action.observation_raw = results
                action.observation = format_search_results(results)

            elif action.action_name == "get_brick_entity":
                entity_info = get_brick_entity(action.action_argument)
                action.observation_raw = entity_info
                action.observation = format_entity_info(entity_info)

            elif action.action_name == "get_property_examples":
                examples = get_property_examples(action.action_argument)
                action.observation_raw = examples
                if examples:
                    lines = [f"{subj} -- {prop} --> {obj}" for subj, prop, obj in examples]
                    action.observation = "\n".join(lines)
                else:
                    action.observation = "No examples found"

            elif action.action_name == "execute_sparql":
                # Validate and fix SPARQL before execution
                corrected_sparql = self._validate_and_fix_sparql(action.action_argument)

                # Apply temporal constraints if handler is enabled and decomposition available
                # Skip if temporal guide was used (LLM already generated temporal constraints)
                if self.use_temporal_handler and self.temporal_handler and hasattr(self, '_current_state'):
                    if self._current_state.decomposition and self._current_state.decomposition.temporal.has_constraint:
                        # Check if temporal guide was used (set in controller)
                        if hasattr(self, '_using_temporal_guide') and self._using_temporal_guide:
                            print(f"[INFO] Skipping temporal handler (LLM used temporal patterns guide)")
                        else:
                            print(f"[INFO] Applying temporal constraint: {self._current_state.decomposition.temporal.type}")
                            temporal_result = self.temporal_handler.apply_temporal_constraint(
                                corrected_sparql,
                                decomposition=self._current_state.decomposition
                            )
                            if temporal_result.was_modified:
                                print(f"[INFO] SPARQL modified with temporal pattern: {temporal_result.pattern_applied}")
                                corrected_sparql = temporal_result.modified_sparql

                sparql_query = BrickSparqlQuery(sparql=corrected_sparql)

                try:
                    sparql_query.execute()
                except Exception as exec_error:
                    # Catch execution errors and provide helpful feedback
                    error_msg = str(exec_error).lower()
                    if "prefix" in error_msg or "namespace" in error_msg:
                        action.observation = f"PREFIX ERROR: {str(exec_error)}\n\nMake sure all PREFIX declarations are properly formatted with angle brackets.\nRequired prefixes: brick:, bldg:, ref:, xsd:"
                    else:
                        action.observation = f"SPARQL Execution Error: {str(exec_error)}"
                    return action, None

                action.observation_raw = sparql_query

                if sparql_query.has_results():
                    action.observation = sparql_query.results_as_table()
                else:
                    action.observation = f"Query returned no results. Status: {sparql_query.execution_status.value}"

                # Store the SPARQL query
                return action, sparql_query

            elif action.action_name == "stop":
                action.observation = "Stopping execution"

            else:
                action.observation = f"Unknown action: {action.action_name}"

        except Exception as e:
            action.observation = f"Error executing action: {str(e)}"

        return action, None

    def run(self, question: str, verbose: bool = True) -> Tuple[AgentState, Optional[BrickSparqlQuery]]:
        """
        Run the agent on a question.

        Args:
            question: Natural language question
            verbose: If True, print progress

        Returns:
            (final_state, final_sparql) tuple
        """
        state = AgentState(question=question)

        # Store state reference for temporal handler access
        self._current_state = state

        if verbose:
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"{'='*80}\n")

        # Step 1: Decompose the query (if enabled and needed)
        if self.use_decomposer and self.decomposer:
            # Check if decomposition is actually needed
            should_decompose = self.decomposer.should_decompose(question)

            if should_decompose:
                if verbose:
                    print("🔍 Step 1: Decomposing query (complex query detected)...\n")

                decomposition = self.decomposer.decompose(question, verbose=verbose)
                state.decomposition = decomposition

                if verbose:
                    print()  # Add spacing after decomposition output
            else:
                if verbose:
                    print("⚡ Step 1: Skipping decomposition (simple query)...\n")

        # Step 2: Iterative SPARQL generation
        if verbose and self.use_decomposer:
            if state.decomposition:
                print("🔧 Step 2: Iterative SPARQL generation (using decomposition)...\n")
            else:
                print("🔧 Step 1: Iterative SPARQL generation...\n")

        while not state.should_stop():
            # Get next action from controller
            action = self.controller(state)

            # Check for duplicates
            if state.is_duplicate_action(action):
                if verbose:
                    print(f"[WARNING] Duplicate action detected: {action.action_name}({action.action_argument})")
                    print("   Skipping...\n")
                # Force stop if repeating
                action = BrickAction(
                    thought="Detected duplicate action, stopping",
                    action_name="stop",
                    action_argument=""
                )

            # Execute action
            action, sparql_query = self.execute_action(action)

            # Store action
            state.actions.append(action)

            # Store SPARQL if generated
            if sparql_query:
                state.generated_sparqls.append(sparql_query)

            # Print progress
            if verbose:
                print(action.to_string(include_observation=True))
                print()

        # Set final SPARQL
        if state.generated_sparqls:
            state.final_sparql = state.generated_sparqls[-1]

        if verbose:
            print(f"{'='*80}")
            print(f"Completed in {len(state.actions)} actions")

            # Show decomposition summary
            if state.decomposition:
                print(f"\n📊 Decomposition Summary:")
                print(f"  - Sensors: {', '.join([s.sensor_id for s in state.decomposition.sensors])}")
                print(f"  - Temporal: {state.decomposition.temporal.type}")
                print(f"  - Aggregation: {', '.join(state.decomposition.aggregation.operations) if state.decomposition.aggregation.required else 'None'}")

            if state.final_sparql:
                print(f"\nFinal SPARQL:\n{state.final_sparql.sparql}")
                print(f"\nFinal Results:\n{state.final_sparql.results_as_table()}")
            print(f"{'='*80}\n")

        return state, state.final_sparql


# Simple helper function for quick testing
def ask_brick(question: str, ttl_file: str = None, csv_file: str = None, verbose: bool = True):
    """
    Quick helper to ask a question about Brick data.

    Args:
        question: Natural language question
        ttl_file: Path to Brick TTL file
        csv_file: Path to timeseries CSV file
        verbose: Print progress

    Returns:
        Final SPARQL query results
    """
    agent = BrickAgent()
    agent.initialize_graph(ttl_file=ttl_file, csv_file=csv_file)

    state, final_sparql = agent.run(question, verbose=verbose)

    if final_sparql and final_sparql.has_results():
        return final_sparql.execution_result
    return None


if __name__ == "__main__":
    # Interactive session with Brick Agent
    print("Brick Agent - Iterative framework for Brick SPARQL generation")
    print("="*80)

    # Initialize with your data files
    agent = BrickAgent(
        engine="gemini-flash",
        ttl_schema_file="LBNL_FDD_Data_Sets_FCU_ttl.ttl"  # Include TTL schema in prompt
    )

    # Load Brick schema and timeseries data
    print("\nLoading Brick schema and timeseries data...")
    agent.initialize_graph(
        ttl_file="LBNL_FDD_Data_Sets_FCU_ttl.ttl",
        csv_file="LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv",
        max_csv_rows=1000  # Load last 1000 rows (most recent data from Dec 2018)
    )
    print("Data loaded successfully!\n")

    # Interactive loop
    print("Ask questions about the building data.")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")

    while True:
        # Get question from user
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting session...")
            break

        # Check for exit commands
        if question.lower() in ['exit', 'quit', 'q', '']:
            print("\nExiting session...")
            break

        # Process the question
        state, final_sparql = agent.run(question, verbose=True)

        # Optionally show summary
        print("\n" + "-"*80)
        print("Ready for next question!\n")
