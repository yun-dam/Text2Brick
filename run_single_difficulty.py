"""
Run evaluation for a single difficulty level
Useful for testing or running partial evaluations
"""

import sys
import argparse
from run_evaluation import EvaluationRunner
from brick_agent import BrickAgent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run evaluation for a single difficulty level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_single_difficulty.py easy
  python run_single_difficulty.py easy --fewshot
  python run_single_difficulty.py easy --skip 3
  python run_single_difficulty.py easy --skip 1,3,5
  python run_single_difficulty.py easy --only 1,2,3,4,5
  python run_single_difficulty.py easy --no-temporal
  python run_single_difficulty.py advanced --fewshot --skip 2 --no-temporal
        """
    )

    parser.add_argument(
        "difficulty",
        choices=["easy", "advanced", "expert"],
        help="Difficulty level to evaluate"
    )

    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="Enable few-shot examples in the controller prompt"
    )

    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated list of question numbers to skip (e.g., '3' or '1,3,5')"
    )

    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated list of question numbers to run (e.g., '1,2,3,4,5')"
    )

    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable temporal pattern handling (no temporal guides or handlers)"
    )

    args = parser.parse_args()
    difficulty = args.difficulty.lower()

    # Map difficulty to question file
    question_files = {
        "easy": "evaluation_questions.txt",
        "advanced": "evaluation_questions_advanced.txt",
        "expert": "evaluation_questions_expert.txt"
    }

    print(f"\n{'='*80}")
    print(f"Running {difficulty.upper()} difficulty evaluation")
    if args.fewshot:
        print(f"Mode: WITH FEW-SHOT EXAMPLES")
    else:
        print(f"Mode: WITHOUT FEW-SHOT EXAMPLES")
    if args.no_temporal:
        print(f"Temporal: DISABLED (no temporal patterns)")
    else:
        print(f"Temporal: ENABLED (using temporal patterns)")
    if args.only:
        print(f"Running only questions: {args.only}")
    elif args.skip:
        print(f"Skipping questions: {args.skip}")
    print(f"{'='*80}\n")

    # Create evaluation runner with subfolder for fewshot mode
    output_subdir = f"evaluation_results/{difficulty}/fewshot" if args.fewshot else f"evaluation_results/{difficulty}/no_fewshot"
    runner = EvaluationRunner(output_dir=output_subdir)

    # Initialize agent
    print("[1/3] Initializing Brick Agent...")
    agent = BrickAgent(
        engine="gemini-flash",
        use_fewshot=args.fewshot,
        use_temporal_handler=not args.no_temporal,  # Disable if --no-temporal flag is set
        ttl_schema_file="LBNL_FDD_Data_Sets_FCU_ttl.ttl"  # Include TTL schema in prompt
    )

    print("[2/3] Loading Brick schema and timeseries data...")
    agent.initialize_graph(
        ttl_file="LBNL_FDD_Data_Sets_FCU_ttl.ttl",
        csv_file="LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv",
        max_csv_rows=168  # Load all hourly data (8760 hours in a year)
    )
    print("✅ Data loaded successfully!")

    # Parse questions
    print(f"[3/3] Parsing {difficulty} questions...")
    all_questions = runner.parse_question_file(question_files[difficulty])

    # Validate that --skip and --only are not used together
    if args.skip and args.only:
        print("ERROR: Cannot use both --skip and --only flags together")
        sys.exit(1)

    # Filter questions based on --skip or --only
    if args.only:
        only_numbers = [int(x.strip()) for x in args.only.split(',') if x.strip()]
        print(f"  Running only questions: {only_numbers}")
        # Question numbers are 1-indexed, list indices are 0-indexed
        questions = [q for i, q in enumerate(all_questions, start=1) if i in only_numbers]
    elif args.skip:
        skip_numbers = [int(x.strip()) for x in args.skip.split(',') if x.strip()]
        print(f"  Skipping questions: {skip_numbers}")
        # Question numbers are 1-indexed, list indices are 0-indexed
        questions = [q for i, q in enumerate(all_questions, start=1) if i not in skip_numbers]
    else:
        questions = all_questions

    print(f"  Found {len(questions)} questions (after filtering)")

    # Run evaluation
    results = runner.run_evaluation_set(agent, questions, difficulty)

    # Save results
    runner.save_results({difficulty: results})

    # Print summary
    print(f"\n{'='*80}")
    print(f"{difficulty.upper()} EVALUATION COMPLETE")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nTotal Questions: {len(results)}")
    print(f"Success:         {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"No Results:      {sum(1 for r in results if r['status'] == 'no_results')}")
    print(f"Errors:          {sum(1 for r in results if r['status'] == 'error')}")
    print(f"Avg Time:        {sum(r['execution_time_seconds'] for r in results) / len(results):.2f}s")
    print(f"Avg Iterations:  {sum(r['num_iterations'] for r in results) / len(results):.1f}")
    print(f"\n📊 Results saved to: {runner.result_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
