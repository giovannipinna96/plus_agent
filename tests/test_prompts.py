"""Test prompts for the multi-agent system."""

from typing import List, Dict, Any


class TestPrompts:
    """Collection of test prompts for different complexity levels."""
    
    SIMPLE_PROMPTS = [
        "Show me the basic information about this dataset",
        "What are the column names and data types?",
        "How many rows and columns are in the dataset?",
        "Preview the first 5 rows of data",
        "What columns have missing values?",
        "Show me basic statistics for numeric columns",
        "What are the unique values in the 'sex' column?",
        "Tell me about the 'age' column"
    ]
    
    MEDIUM_PROMPTS = [
        "Calculate the average age by gender",
        "Show survival rates by passenger class", 
        "Find the correlation between age and fare",
        "Group passengers by embark_town and show counts",
        "Filter passengers older than 30 and show their survival rate",
        "Create dummy variables for the 'embarked' column",
        "Handle missing values in the 'age' column using mean imputation",
        "Convert the 'fare' column to integer type",
        "Show the distribution of passengers by class and gender",
        "Calculate the survival rate for each passenger class"
    ]
    
    COMPLEX_PROMPTS = [
        "Train a random forest model to predict survival and show feature importance",
        "Compare the performance of different ML models for survival prediction",
        "Perform a complete analysis: data exploration, preprocessing, and predictive modeling",
        "Build a classification model using age, sex, pclass, and fare as features",
        "Train multiple models (SVM, Random Forest, KNN) and compare their accuracy",
        "Create age groups (child, adult, senior) and analyze survival patterns by group",
        "Prepare the data for machine learning: handle missing values, encode categorical variables, and train a model",
        "Analyze the relationship between fare and survival, then build a predictive model",
        "Perform feature engineering: create family_size from sibsp and parch, then train a model",
        "Build a comprehensive predictive model and evaluate its performance with cross-validation"
    ]
    
    COMPREHENSIVE_PROMPTS = [
        """Perform a complete data science workflow on this dataset:
        1. Load and examine the data structure
        2. Handle missing values appropriately
        3. Create meaningful features from existing data
        4. Analyze patterns and relationships
        5. Train multiple machine learning models
        6. Compare model performance and select the best one
        7. Provide insights and recommendations""",
        
        """I want to understand what factors contributed to survival on the Titanic:
        1. First, show me the dataset overview
        2. Analyze survival rates by different passenger characteristics
        3. Handle any data quality issues
        4. Build a predictive model to identify key survival factors
        5. Explain which features are most important for survival prediction""",
        
        """Help me build the best possible survival prediction model:
        1. Explore the dataset and identify important patterns
        2. Preprocess the data (missing values, encoding, scaling)
        3. Engineer new features that might improve prediction
        4. Train and compare multiple ML algorithms
        5. Evaluate models using appropriate metrics
        6. Provide final recommendations on the best approach"""
    ]
    
    @classmethod
    def get_all_prompts(cls) -> Dict[str, List[str]]:
        """Get all test prompts organized by complexity level."""
        return {
            "simple": cls.SIMPLE_PROMPTS,
            "medium": cls.MEDIUM_PROMPTS,
            "complex": cls.COMPLEX_PROMPTS,
            "comprehensive": cls.COMPREHENSIVE_PROMPTS
        }
    
    @classmethod
    def get_random_prompt(cls, complexity: str = "medium") -> str:
        """Get a random prompt of specified complexity."""
        import random
        
        prompts_by_complexity = cls.get_all_prompts()
        if complexity in prompts_by_complexity:
            return random.choice(prompts_by_complexity[complexity])
        else:
            return random.choice(cls.MEDIUM_PROMPTS)
    
    @classmethod
    def get_prompts_for_testing(cls) -> List[Dict[str, Any]]:
        """Get a curated set of prompts for systematic testing."""
        return [
            {
                "prompt": "Show me the basic information about this dataset",
                "complexity": "simple",
                "expected_agents": ["data_reader"],
                "description": "Basic dataset exploration"
            },
            {
                "prompt": "Calculate the average age by gender and handle missing values",
                "complexity": "medium", 
                "expected_agents": ["data_reader", "data_operations", "data_manipulation"],
                "description": "Data operations with preprocessing"
            },
            {
                "prompt": "Train a random forest model to predict survival",
                "complexity": "complex",
                "expected_agents": ["data_reader", "data_manipulation", "ml_prediction"],
                "description": "Machine learning model training"
            },
            {
                "prompt": "Perform complete analysis: exploration, preprocessing, and modeling",
                "complexity": "comprehensive",
                "expected_agents": ["data_reader", "data_manipulation", "data_operations", "ml_prediction"],
                "description": "End-to-end data science workflow"
            }
        ]


def run_test_suite():
    """Run a test suite with different prompts."""
    # This would be implemented to actually test the system
    # For now, it just prints the available test prompts
    
    print("=== Multi-Agent System Test Prompts ===\n")
    
    all_prompts = TestPrompts.get_all_prompts()
    
    for complexity, prompts in all_prompts.items():
        print(f"## {complexity.upper()} PROMPTS:")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. {prompt}")
        print()
    
    print("## CURATED TEST SET:")
    test_prompts = TestPrompts.get_prompts_for_testing()
    for i, test in enumerate(test_prompts, 1):
        print(f"  {i}. [{test['complexity']}] {test['prompt']}")
        print(f"     Expected agents: {', '.join(test['expected_agents'])}")
        print(f"     Description: {test['description']}")
        print()


if __name__ == "__main__":
    run_test_suite()