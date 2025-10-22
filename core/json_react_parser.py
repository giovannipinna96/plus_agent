"""Custom ReAct output parser that handles JSON Action Input."""

import json
import re
from typing import Union
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class JSONReActOutputParser(ReActSingleInputOutputParser):
    """
    Custom ReAct parser that automatically converts JSON string Action Input to dict.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse LLM output and convert JSON strings in Action Input to dicts.
        """
        # Check for Final Answer first
        if "Final Answer:" in text:
            final_answer_pattern = r"Final Answer:\s*(.*)"
            match = re.search(final_answer_pattern, text, re.DOTALL)
            if match:
                return AgentFinish(
                    return_values={"output": match.group(1).strip()},
                    log=text
                )
            else:
                raise OutputParserException(
                    f"Could not parse Final Answer from text: {text}\n"
                    f"Expected format: 'Final Answer: <answer>'"
                )

        # Check for Thought/Action/Action Input format
        thought_pattern = r"Thought:\s*(.*?)\nAction:\s*(.*?)\nAction Input:\s*(.*?)(?:\n|$)"
        match = re.search(thought_pattern, text, re.DOTALL)

        if not match:
            # Detailed error message
            error_msg = f"Output does not match required ReAct format.\n\n"
            error_msg += f"Expected format:\n"
            error_msg += f"Thought: <your reasoning>\n"
            error_msg += f"Action: <action_name>\n"
            error_msg += f"Action Input: <action_input>\n\n"
            error_msg += f"Received text:\n{text}\n\n"

            if "Thought:" not in text:
                error_msg += "ERROR: Missing 'Thought:'\n"
            if "Action:" not in text:
                error_msg += "ERROR: Missing 'Action:'\n"
            if "Action Input:" not in text:
                error_msg += "ERROR: Missing 'Action Input:'\n"

            raise OutputParserException(error_msg)

        thought = match.group(1).strip()
        action = match.group(2).strip()
        action_input_str = match.group(3).strip()

        # Validate that action and action_input are not empty
        if not action:
            raise OutputParserException(f"Action cannot be empty. Text: {text}")
        if not action_input_str:
            raise OutputParserException(f"Action Input cannot be empty. Text: {text}")

        # Try to parse Action Input as JSON
        tool_input = action_input_str
        if action_input_str.startswith('{') and action_input_str.endswith('}'):
            try:
                # Try to parse as JSON
                tool_input = json.loads(action_input_str)
                print(f"[JSON PARSER]: Successfully parsed JSON input with {len(tool_input)} keys")
            except json.JSONDecodeError as e:
                # If JSON parsing fails, keep it as string and let the tool handle it
                print(f"[JSON PARSER]: Failed to parse as JSON: {e}. Using as string.")
                tool_input = action_input_str

        return AgentAction(tool=action, tool_input=tool_input, log=text)
