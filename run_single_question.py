"""
Evaluate a single question and save detailed logs
Useful for debugging and testing individual queries
"""

import sys
import io
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from brick_agent import BrickAgent

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate a single question with detailed logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_single_question.py "What is the latest room temperature?"
  python run_single_question.py "Show me room temperature over time" --fewshot
  python run_single_question.py "Get fan speed on Nov 15th at 2pm" --fewshot
  python run_single_question.py "Compare discharge and return temperatures" --fewshot --output logs/my_test.json
        """
    )

    parser.add_argument(
        "question",
        help="Natural language question to evaluate"
    )

    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="Enable few-shot examples in the controller prompt"
    )

    parser.add_argument(
        "--no-decomposer",
        action="store_true",
        help="Disable query decomposer"
    )

    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable temporal handler"
    )

    parser.add_argument(
        "--engine",
        default="gemini-flash",
        help="LLM engine to use (default: gemini-flash)"
    )

    parser.add_argument(
        "--output",
        help="Output log file path (default: auto-generated in logs/)"
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=525600,
        help="Maximum CSV rows to load (default: 525600)"
    )

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*80)
    print("SINGLE QUESTION EVALUATION")
    print("="*80)
    print(f"\nQuestion: {args.question}")
    print(f"\nConfiguration:")
    print(f"  Engine:            {args.engine}")
    print(f"  Few-shot:          {'✅ Enabled' if args.fewshot else '❌ Disabled'}")
    print(f"  Decomposer:        {'✅ Enabled' if not args.no_decomposer else '❌ Disabled'}")
    print(f"  Temporal Handler:  {'✅ Enabled' if not args.no_temporal else '❌ Disabled'}")
    print(f"  Max CSV Rows:      {args.max_rows:,}")
    print("="*80 + "\n")

    # Initialize agent
    print("\n[1/3] Initializing Brick Agent...")
    print("-"*80)
    agent = BrickAgent(
        engine=args.engine,
        use_decomposer=not args.no_decomposer,
        use_temporal_handler=not args.no_temporal,
        use_fewshot=args.fewshot
    )
    print("-"*80)
    print("✅ Agent initialized!\n")

    print("[2/3] Loading Brick schema and timeseries data...")
    print("-"*80)
    print(f"Loading {args.max_rows:,} rows...")
    agent.initialize_graph(
        ttl_file="LBNL_FDD_Data_Sets_FCU_ttl.ttl",
        csv_file="LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv",
        max_csv_rows=args.max_rows,
        use_cache=False
    )
    print("-"*80)
    print("✅ Data loaded successfully!\n")

    # Run the question
    print("\n[3/3] Processing question...")
    print("="*80)

    start_time = time.time()

    try:
        # Monkey-patch the execute action to show real-time progress
        original_execute_action = agent.execute_action
        iteration_count = [0]  # Use list to allow modification in nested function

        def logged_execute_action(state, action):
            iteration_count[0] += 1
            print(f"\n--- Iteration {iteration_count[0]} ---")
            print(f"💭 Thought: {action.thought}")
            print(f"⚡ Action: {action.action_name}({action.action_argument[:100]}{'...' if len(action.action_argument) > 100 else ''})")

            result = original_execute_action(state, action)

            # Show observation
            if action.observation:
                obs_preview = action.observation[:200] + "..." if len(action.observation) > 200 else action.observation
                print(f"👁️  Observation: {obs_preview}")

            return result

        agent.execute_action = logged_execute_action

        state, final_sparql = agent.run(args.question, verbose=False)  # Set verbose=False since we're doing custom logging
        execution_time = time.time() - start_time

        # Prepare detailed log
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": args.question,
            "configuration": {
                "engine": args.engine,
                "fewshot": args.fewshot,
                "decomposer": not args.no_decomposer,
                "temporal_handler": not args.no_temporal,
                "max_csv_rows": args.max_rows
            },
            "execution_time_seconds": round(execution_time, 2),
            "num_iterations": len(state.actions),
            "status": "unknown",
            "actions": [],
            "generated_sparqls": [],
            "final_sparql": None,
            "final_results": None,
            "num_results": 0
        }

        # Capture all actions
        for action in state.actions:
            action_log = {
                "thought": action.thought,
                "action_name": action.action_name,
                "action_argument": action.action_argument,
                "observation": action.observation
            }
            log_data["actions"].append(action_log)

        for sparql_query in state.generated_sparqls:
            # Determine syntax status
            syntax_status = "unknown"
            if sparql_query.execution_status:
                from brick_utils import SparqlExecutionStatus
                if sparql_query.execution_status in [SparqlExecutionStatus.SUCCESS, SparqlExecutionStatus.EMPTY_RESULT]:
                    syntax_status = "good"
                elif sparql_query.execution_status == SparqlExecutionStatus.SYNTAX_ERROR:
                    syntax_status = "error"
                else:
                    syntax_status = "unknown"
            
            sparql_log = {
                "sparql": sparql_query.sparql,
                "has_results": sparql_query.has_results(),
                "execution_status": sparql_query.execution_status.value if sparql_query.execution_status else None,
                "syntax_status": syntax_status,
                "num_results": len(sparql_query.execution_result) if sparql_query.execution_result else 0
            }
            log_data["generated_sparqls"].append(sparql_log)

        # Capture final results
        if final_sparql:
            log_data["final_sparql"] = final_sparql.sparql
            log_data["num_results"] = len(final_sparql.execution_result) if final_sparql.execution_result else 0

            if final_sparql.execution_result:
                # Store all results
                log_data["final_results"] = final_sparql.execution_result
                log_data["status"] = "success"

                print("\n" + "="*80)
                print(f"✅ SUCCESS")
                print("="*80)
                print(f"\n📊 Summary:")
                print(f"  Execution time:  {execution_time:.2f}s")
                print(f"  Iterations:      {len(state.actions)}")
                print(f"  SPARQL queries:  {len(state.generated_sparqls)}")
                print(f"  Results found:   {log_data['num_results']}")

                # Show all generated queries if more than 1
                if len(state.generated_sparqls) > 1:
                    print(f"\n📝 Generated SPARQL Queries ({len(state.generated_sparqls)} total):")
                    print("-"*80)
                    for i, sq in enumerate(state.generated_sparqls, 1):
                        status_icon = "✅" if sq.has_results() else "❌"
                        print(f"\n{i}. {status_icon} Query (returned {len(sq.execution_result) if sq.execution_result else 0} results)")
                        print(sq.sparql[:150] + "..." if len(sq.sparql) > 150 else sq.sparql)
                    print("-"*80)

                print(f"\n📊 Final SPARQL Query:")
                print("-"*80)
                print(final_sparql.sparql)
                print("-"*80)

                print(f"\n📋 Results (showing first 10 rows):")
                print("-"*80)
                for i, result in enumerate(final_sparql.execution_result[:10], 1):
                    print(f"{i}. {result}")
                if len(final_sparql.execution_result) > 10:
                    print(f"... and {len(final_sparql.execution_result) - 10} more rows")
                print("-"*80)
            else:
                log_data["status"] = "no_results"
                print("\n" + "="*80)
                print(f"⚠️  NO RESULTS")
                print("="*80)
                print(f"\n📊 Summary:")
                print(f"  Execution time:  {execution_time:.2f}s")
                print(f"  Iterations:      {len(state.actions)}")
                print(f"  SPARQL queries:  {len(state.generated_sparqls)}")
                print(f"  Results found:   0")

                # Show all generated queries if more than 1
                if len(state.generated_sparqls) > 1:
                    print(f"\n📝 Generated SPARQL Queries ({len(state.generated_sparqls)} total):")
                    print("-"*80)
                    for i, sq in enumerate(state.generated_sparqls, 1):
                        status_icon = "✅" if sq.has_results() else "❌"
                        print(f"\n{i}. {status_icon} Query (returned {len(sq.execution_result) if sq.execution_result else 0} results)")
                        print(sq.sparql[:150] + "..." if len(sq.sparql) > 150 else sq.sparql)
                    print("-"*80)

                print(f"\n📊 Final SPARQL Query:")
                print("-"*80)
                print(final_sparql.sparql)
                print("-"*80)
        else:
            log_data["status"] = "no_sparql"
            print("\n" + "="*80)
            print(f"⚠️  NO SPARQL GENERATED")
            print("="*80)
            print(f"\n📊 Summary:")
            print(f"  Execution time:  {execution_time:.2f}s")
            print(f"  Iterations:      {len(state.actions)}")
            print(f"  SPARQL queries:  0")

    except Exception as e:
        execution_time = time.time() - start_time
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": args.question,
            "configuration": {
                "engine": args.engine,
                "fewshot": args.fewshot,
                "decomposer": not args.no_decomposer,
                "temporal_handler": not args.no_temporal
            },
            "execution_time_seconds": round(execution_time, 2),
            "status": "error",
            "error": str(e),
            "error_traceback": sys.exc_info()
        }

        print("\n" + "="*80)
        print(f"❌ ERROR")
        print("="*80)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()

    # Save log file
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate filename
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create safe filename from question
        safe_question = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in args.question)
        safe_question = safe_question.replace(' ', '_')[:50]  # Limit length
        output_path = log_dir / f"single_question_{safe_question}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print(f"💾 Log saved to: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
