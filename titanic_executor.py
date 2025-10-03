"""
Titanic Questions Executor - Automated execution of tool sequences for Titanic dataset analysis.
"""

import json
import sys
import os
from typing import Dict, Any, List
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all tools
from tools.data_tools import read_csv_file, get_column_info, get_data_summary, preview_data
from tools.operations_tools import filter_data, aggregate_data, perform_math_operations
from tools.manipulation_tools import create_dummy_variables, handle_missing_values
from tools.ml_tools import train_random_forest_model, train_regression_model, evaluate_model
from tools.titanic_specific_tools import (
    calculate_survival_rate_by_group,
    get_statistics_for_profile,
    calculate_survival_probability_by_features,
    get_fare_estimate_by_profile,
    count_passengers_by_criteria
)


class TitanicExecutor:
    """Execute tool sequences for Titanic analysis questions."""

    def __init__(self, data_file_path: str = "data/titanic.csv"):
        self.data_file_path = data_file_path
        self.results = {}

        # Map tool names to actual functions
        self.tool_registry = {
            "read_csv_file": read_csv_file,
            "get_column_info": get_column_info,
            "get_data_summary": get_data_summary,
            "preview_data": preview_data,
            "filter_data": filter_data,
            "aggregate_data": aggregate_data,
            "perform_math_operations": perform_math_operations,
            "create_dummy_variables": create_dummy_variables,
            "handle_missing_values": handle_missing_values,
            "train_random_forest_model": train_random_forest_model,
            "train_regression_model": train_regression_model,
            "evaluate_model": evaluate_model,
            "calculate_survival_rate_by_group": calculate_survival_rate_by_group,
            "get_statistics_for_profile": get_statistics_for_profile,
            "calculate_survival_probability_by_features": calculate_survival_probability_by_features,
            "get_fare_estimate_by_profile": get_fare_estimate_by_profile,
            "count_passengers_by_criteria": count_passengers_by_criteria
        }

    def execute_question(self, question_id: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool sequence for a specific question."""

        print(f"\\n{'='*60}")
        print(f"Executing Question {question_id}")
        print(f"Text: {question_data['text']}")
        print(f"Complexity: {question_data['complexity']}")
        print(f"{'='*60}")

        result = {
            "question_id": question_id,
            "question_text": question_data["text"],
            "complexity": question_data["complexity"],
            "tools_used": [],
            "tool_results": [],
            "final_answer": None,
            "execution_success": True,
            "error_message": None
        }

        try:
            # Execute each tool in sequence
            for i, tool_name in enumerate(question_data["tools_sequence"]):
                print(f"\\nStep {i+1}: Executing {tool_name}")

                if tool_name not in self.tool_registry:
                    raise Exception(f"Tool '{tool_name}' not found in registry")

                tool_function = self.tool_registry[tool_name]

                # Execute tool with appropriate parameters based on question
                tool_result = self._execute_tool_with_params(tool_name, question_id, question_data)

                result["tools_used"].append(tool_name)
                result["tool_results"].append({
                    "tool": tool_name,
                    "result": tool_result
                })

                print(f"Result: {tool_result[:200]}...")

            # Extract final answer based on question type
            result["final_answer"] = self._extract_final_answer(question_id, result["tool_results"])

        except Exception as e:
            print(f"ERROR executing question {question_id}: {str(e)}")
            result["execution_success"] = False
            result["error_message"] = str(e)

        return result

    def _execute_tool_with_params(self, tool_name: str, question_id: str, question_data: Dict[str, Any]) -> str:
        """Execute tool with appropriate parameters based on the question."""

        tool_function = self.tool_registry[tool_name]

        # Question-specific parameter mapping using invoke method
        if question_id == "question_1":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "get_data_summary":
                return tool_function.invoke({"file_path": self.data_file_path})

        elif question_id == "question_2":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "get_column_info":
                return tool_function.invoke({"file_path": self.data_file_path, "column_name": "age"})

        elif question_id == "question_3":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "filter_data":
                return tool_function.invoke({"file_path": self.data_file_path, "column_name": "survived", "condition": "equals", "value": 1})
            elif tool_name == "get_data_summary":
                filtered_path = self.data_file_path.replace('.csv', '_filtered.csv')
                return tool_function.invoke({"file_path": filtered_path})

        elif question_id == "question_4":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "aggregate_data":
                return tool_function.invoke({"file_path": self.data_file_path, "group_by_columns": "pclass", "agg_column": "survived", "agg_function": "mean"})

        elif question_id == "question_5":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "aggregate_data":
                return tool_function.invoke({"file_path": self.data_file_path, "group_by_columns": "sex", "agg_column": "fare", "agg_function": "mean"})

        elif question_id == "question_6":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "filter_data":
                return tool_function.invoke({"file_path": self.data_file_path, "column_name": "age", "condition": "less_than", "value": 12})
            elif tool_name == "aggregate_data":
                filtered_path = self.data_file_path.replace('.csv', '_filtered.csv')
                return tool_function.invoke({"file_path": filtered_path, "group_by_columns": "pclass", "agg_column": "survived", "agg_function": "sum"})

        elif question_id == "question_7":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "get_fare_estimate_by_profile":
                return get_fare_estimate_by_profile.invoke({"file_path": self.data_file_path, "sex": "female", "pclass": 3, "age": 20})

        elif question_id == "question_8":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "calculate_survival_probability_by_features":
                return calculate_survival_probability_by_features.invoke({"file_path": self.data_file_path, "sex": "male", "pclass": 1, "age_range": "30-60"})

        elif question_id == "question_9":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "create_dummy_variables":
                # Create dummies for sex
                result1 = tool_function.invoke({"file_path": self.data_file_path, "column_name": "sex"})
                # Create dummies for embarked
                dummy_path = self.data_file_path.replace('.csv', '_with_dummies.csv')
                return tool_function.invoke({"file_path": dummy_path, "column_name": "embarked"})
            elif tool_name == "train_random_forest_model":
                dummy_path = self.data_file_path.replace('.csv', '_with_dummies.csv')
                features = "pclass,age,sibsp,parch,fare,sex_female,sex_male"
                return tool_function.invoke({"file_path": dummy_path, "target_column": "survived", "feature_columns": features, "task_type": "classification"})

        elif question_id == "question_10":
            if tool_name == "read_csv_file":
                return tool_function.invoke({"file_path": self.data_file_path})
            elif tool_name == "create_dummy_variables":
                # Create dummies for sex
                result1 = tool_function.invoke({"file_path": self.data_file_path, "column_name": "sex"})
                # Create dummies for embarked
                dummy_path = self.data_file_path.replace('.csv', '_with_dummies.csv')
                return tool_function.invoke({"file_path": dummy_path, "column_name": "embarked"})
            elif tool_name == "train_regression_model":
                dummy_path = self.data_file_path.replace('.csv', '_with_dummies.csv')
                features = "pclass,age,sex_female,sex_male,embarked_C,embarked_Q,embarked_S"
                return tool_function.invoke({"file_path": dummy_path, "target_column": "fare", "feature_columns": features, "model_type": "random_forest"})
            elif tool_name == "evaluate_model":
                model_path = dummy_path.replace('.csv', '_random_forest_regression_model.joblib')
                features = "pclass,age,sex_female,sex_male,embarked_C,embarked_Q,embarked_S"
                return tool_function.invoke({"model_path": model_path, "test_data_path": dummy_path, "target_column": "fare", "feature_columns": features})

        # Default execution with file path - use invoke method for LangChain tools
        try:
            return tool_function.invoke({"file_path": self.data_file_path})
        except:
            # Fallback to direct call if not a LangChain tool
            return tool_function(self.data_file_path)

    def _extract_final_answer(self, question_id: str, tool_results: List[Dict]) -> str:
        """Extract the final answer from tool results."""

        if not tool_results:
            return "No results available"

        last_result = tool_results[-1]["result"]

        # Extract key information based on question type
        if question_id == "question_1":
            # Extract total passenger count
            if "shape:" in last_result:
                import re
                match = re.search(r"shape: \((\d+),", last_result)
                if match:
                    return f"Il dataset contiene {match.group(1)} passeggeri in totale"

        elif question_id == "question_2":
            # Extract average age
            if "mean:" in last_result:
                import re
                match = re.search(r"'mean': ([0-9.]+)", last_result)
                if match:
                    return f"L'età media dei passeggeri è {float(match.group(1)):.1f} anni"

        elif question_id in ["question_3", "question_6"]:
            # Extract survival counts
            return f"Risultato dell'analisi: {last_result}"

        elif question_id in ["question_4", "question_5"]:
            # Aggregation results
            return f"Risultati dell'aggregazione: {last_result}"

        elif question_id == "question_7":
            # Fare estimate
            if "estimated_fare" in last_result:
                import re
                match = re.search(r"'estimated_fare': ([0-9.]+)", last_result)
                if match:
                    return f"Una ragazza di 20 anni in terza classe ha probabilmente pagato circa ${float(match.group(1)):.2f} per il biglietto"

        elif question_id == "question_8":
            # Survival probability
            if "survival_probability" in last_result:
                import re
                match = re.search(r"'survival_probability': '([0-9.%]+)'", last_result)
                if match:
                    return f"Un signore ricco in prima classe aveva una probabilità di sopravvivenza del {match.group(1)}"

        elif question_id == "question_9":
            # Feature importance
            if "feature_importance" in last_result:
                return f"Importanza delle caratteristiche per la sopravvivenza: {last_result}"

        elif question_id == "question_10":
            # Model performance
            return f"Risultati del modello di predizione del prezzo: {last_result}"

        return last_result

    def execute_all_questions(self, questions_file: str = "titanic_questions.json") -> Dict[str, Any]:
        """Execute all questions and return results."""

        print("Loading questions from:", questions_file)

        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        except FileNotFoundError:
            print(f"Questions file {questions_file} not found!")
            return {}

        execution_results = {}

        for question_id, question_data in questions.items():
            try:
                result = self.execute_question(question_id, question_data)
                execution_results[question_id] = result

                print(f"\\n✅ Question {question_id} completed successfully")
                print(f"Final Answer: {result['final_answer']}")

            except Exception as e:
                print(f"\\n❌ Question {question_id} failed: {str(e)}")
                execution_results[question_id] = {
                    "question_id": question_id,
                    "execution_success": False,
                    "error_message": str(e)
                }

        # Save results
        results_file = "titanic_executor_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(execution_results, f, indent=2, ensure_ascii=False)

        print(f"\\n📊 Results saved to: {results_file}")
        return execution_results

    def print_summary(self, results: Dict[str, Any]):
        """Print execution summary."""

        total_questions = len(results)
        successful = sum(1 for r in results.values() if r.get("execution_success", False))
        failed = total_questions - successful

        print(f"\\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Questions: {total_questions}")
        print(f"Successfully Executed: {successful}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {successful/total_questions*100:.1f}%")

        if failed > 0:
            print(f"\\nFailed Questions:")
            for qid, result in results.items():
                if not result.get("execution_success", False):
                    print(f"  - {qid}: {result.get('error_message', 'Unknown error')}")


def main():
    """Main execution function."""

    print("🚢 Titanic Dataset Questions Executor")
    print("=====================================")

    executor = TitanicExecutor()

    # Check if data file exists
    if not os.path.exists(executor.data_file_path):
        print(f"❌ Data file not found: {executor.data_file_path}")
        print("Please ensure the Titanic dataset is available at the specified path.")
        return

    # Execute all questions
    results = executor.execute_all_questions()

    # Print summary
    executor.print_summary(results)

    return results


if __name__ == "__main__":
    main()