"""
Evaluation Script for Brick Text-to-SPARQL Framework
Runs all evaluation questions through the Brick Agent and logs results in JSON format.
"""

import json
from datetime import datetime
from pathlib import Path
import time
from typing import List, Dict, Optional
from brick_agent import BrickAgent
import traceback


class EvaluationRunner:
    """Manages the evaluation process for all question sets"""

    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the evaluation runner.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped result file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_file = self.output_dir / f"evaluation_results_{timestamp}.json"

        print(f"📊 Evaluation results will be saved to: {self.result_file}")

    def parse_question_file(self, file_path: str) -> List[str]:
        """
        Parse questions from a text file.

        Args:
            file_path: Path to question file

        Returns:
            List of questions
        """
        questions = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip empty lines, headers, and notes
            if not line or line.startswith('=') or line.startswith('-') or \
               line.startswith('#') or 'Notes:' in line or \
               'Question' in line and ':' not in line.split('Question')[-1][:20]:
                continue

            # Extract question text (remove numbering)
            # Handle formats like "1. Question text" or "11. Question text"
            if line and line[0].isdigit():
                # Find the first period after the number
                period_idx = line.find('.')
                if period_idx > 0 and period_idx < 5:  # Reasonable position for numbering
                    question = line[period_idx+1:].strip()
                    if question:  # Only add non-empty questions
                        questions.append(question)

        return questions

    def run_question(self, agent: BrickAgent, question: str, question_id: int,
                     difficulty: str) -> Dict:
        """
        Run a single question through the agent and capture all logs.

        Args:
            agent: Initialized BrickAgent
            question: Question text
            question_id: Question number
            difficulty: Difficulty level (easy/advanced/expert)

        Returns:
            Dictionary with complete evaluation logs
        """
        print(f"\n{'='*80}")
        print(f"[{difficulty.upper()}] Question {question_id}: {question}")
        print(f"{'='*80}")

        start_time = time.time()

        result = {
            "question_id": question_id,
            "difficulty": difficulty,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "execution_time_seconds": 0,
            "num_iterations": 0,
            "actions": [],
            "generated_sparqls": [],
            "final_sparql": None,
            "final_results": None,
            "num_results": 0,
            "error": None
        }

        try:
            # Run the agent
            state, final_sparql = agent.run(question, verbose=True)

            # Calculate execution time
            execution_time = time.time() - start_time
            result["execution_time_seconds"] = round(execution_time, 2)
            result["num_iterations"] = len(state.actions)

            # Capture all actions (thought-action-observation history)
            for action in state.actions:
                action_log = {
                    "thought": action.thought,
                    "action_name": action.action_name,
                    "action_argument": action.action_argument,
                    "observation": action.observation
                }
                result["actions"].append(action_log)

            # Capture all generated SPARQL queries
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
                result["generated_sparqls"].append(sparql_log)

            # Capture final SPARQL and results
            if final_sparql:
                result["final_sparql"] = final_sparql.sparql
                result["num_results"] = len(final_sparql.execution_result) if final_sparql.execution_result else 0

                # Store final results (limit to first 100 rows to avoid huge JSON files)
                if final_sparql.execution_result:
                    result["final_results"] = final_sparql.execution_result[:100]
                    result["status"] = "success"
                    print(f"\n✅ SUCCESS - {result['num_results']} results in {execution_time:.2f}s")
                else:
                    result["status"] = "no_results"
                    print(f"\n⚠️  NO RESULTS - completed in {execution_time:.2f}s")
            else:
                result["status"] = "no_sparql"
                print(f"\n⚠️  NO SPARQL GENERATED - completed in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            result["execution_time_seconds"] = round(execution_time, 2)
            result["status"] = "error"
            result["error"] = str(e)
            result["error_traceback"] = traceback.format_exc()
            print(f"\n❌ ERROR: {e}")
            print(traceback.format_exc())

        return result

    def run_evaluation_set(self, agent: BrickAgent, questions: List[str],
                          difficulty: str) -> List[Dict]:
        """
        Run all questions in a difficulty set.

        Args:
            agent: Initialized BrickAgent
            questions: List of questions
            difficulty: Difficulty level

        Returns:
            List of evaluation results
        """
        results = []

        print(f"\n{'#'*80}")
        print(f"# Starting {difficulty.upper()} difficulty evaluation")
        print(f"# Total questions: {len(questions)}")
        print(f"{'#'*80}\n")

        for i, question in enumerate(questions, 1):
            result = self.run_question(agent, question, i, difficulty)
            results.append(result)

            # Save intermediate results after each question
            self.save_results({difficulty: results})

            # Print progress
            success_count = sum(1 for r in results if r["status"] == "success")
            print(f"\n[Progress] {i}/{len(questions)} questions | Success rate: {success_count}/{i} ({success_count/i*100:.1f}%)")

            # Cooldown between questions to manage quota
            if i < len(questions):  # Don't wait after last question
                cooldown = 60  # 1 minute (60 seconds) cooldown between questions
                print(f"⏸️  Cooldown: waiting {cooldown}s ({cooldown//60} minute) before next question...")
                time.sleep(cooldown)

        return results

    def save_results(self, all_results: Dict):
        """
        Save results to JSON file.

        Args:
            all_results: Dictionary mapping difficulty -> list of results
        """
        # Calculate summary statistics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": sum(len(results) for results in all_results.values()),
            "difficulty_breakdown": {}
        }

        for difficulty, results in all_results.items():
            summary["difficulty_breakdown"][difficulty] = {
                "total": len(results),
                "success": sum(1 for r in results if r["status"] == "success"),
                "no_results": sum(1 for r in results if r["status"] == "no_results"),
                "no_sparql": sum(1 for r in results if r["status"] == "no_sparql"),
                "error": sum(1 for r in results if r["status"] == "error"),
                "avg_execution_time": round(sum(r["execution_time_seconds"] for r in results) / len(results), 2) if results else 0,
                "avg_iterations": round(sum(r["num_iterations"] for r in results) / len(results), 2) if results else 0
            }

        # Combine summary and detailed results
        output = {
            "summary": summary,
            "results": all_results
        }

        # Save to JSON
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved results to: {self.result_file}")

    def run_full_evaluation(self, ttl_file: str, csv_file: str, max_csv_rows: int = 60000):
        """
        Run the full evaluation across all difficulty levels.

        Args:
            ttl_file: Path to Brick TTL file
            csv_file: Path to timeseries CSV file
            max_csv_rows: Maximum CSV rows to load
        """
        print("\n" + "="*80)
        print("BRICK TEXT-TO-SPARQL EVALUATION")
        print("="*80)

        # Initialize agent
        print("\n[1/4] Initializing Brick Agent...")
        agent = BrickAgent(engine="gemini-flash")

        print("\n[2/4] Loading Brick schema and timeseries data...")
        agent.initialize_graph(
            ttl_file=ttl_file,
            csv_file=csv_file,
            max_csv_rows=max_csv_rows,
            use_cache=False  # Rebuild graph from scratch each time
        )
        print("✅ Data loaded successfully!")

        # Parse all question files
        print("\n[3/4] Parsing question files...")
        question_files = {
            "easy": "evaluation_questions.txt",
            "advanced": "evaluation_questions_advanced.txt",
            "expert": "evaluation_questions_expert.txt"
        }

        all_questions = {}
        for difficulty, file_path in question_files.items():
            questions = self.parse_question_file(file_path)
            all_questions[difficulty] = questions
            print(f"  - {difficulty.capitalize()}: {len(questions)} questions")

        # Run evaluation for each difficulty level
        print("\n[4/4] Running evaluation...")
        all_results = {}

        for difficulty in ["easy", "advanced", "expert"]:
            questions = all_questions[difficulty]
            results = self.run_evaluation_set(agent, questions, difficulty)
            all_results[difficulty] = results

        # Save final results
        self.save_results(all_results)

        # Print final summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)

        for difficulty, results in all_results.items():
            success_count = sum(1 for r in results if r["status"] == "success")
            print(f"\n{difficulty.upper()}:")
            print(f"  Total: {len(results)}")
            print(f"  Success: {success_count} ({success_count/len(results)*100:.1f}%)")
            print(f"  No Results: {sum(1 for r in results if r['status'] == 'no_results')}")
            print(f"  Errors: {sum(1 for r in results if r['status'] == 'error')}")
            print(f"  Avg Time: {sum(r['execution_time_seconds'] for r in results) / len(results):.2f}s")
            print(f"  Avg Iterations: {sum(r['num_iterations'] for r in results) / len(results):.1f}")

        print(f"\n📊 Detailed results saved to: {self.result_file}")
        print("="*80)


def main():
    """Main entry point for evaluation"""
    # Create evaluation runner
    runner = EvaluationRunner(output_dir="evaluation_results")

    # Run full evaluation
    runner.run_full_evaluation(
        ttl_file="LBNL_FDD_Data_Sets_FCU_ttl.ttl",
        csv_file="LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv",
        max_csv_rows=11000
    )


if __name__ == "__main__":
    main()
