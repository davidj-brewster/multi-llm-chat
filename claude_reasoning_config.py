"""
Configuration module for Claude 3.7 reasoning parameters.

This module provides enhanced configuration for Claude 3.7 reasoning capabilities,
including control over reasoning verbosity, token limits, and template selection.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

# Define reasoning level type
ReasoningLevel = Literal["none", "low", "medium", "high", "auto"]

@dataclass
class ClaudeReasoningConfig:
    """
    Configuration for Claude 3.7 reasoning capabilities.
    
    This class provides fine-grained control over how Claude 3.7's reasoning
    features behave, including token allocation and reasoning style.
    
    Attributes:
        level (ReasoningLevel): Reasoning verbosity level (none, low, medium, high, auto).
            Defaults to "auto".
        
        max_reasoning_tokens (Optional[int]): Maximum tokens to allocate for reasoning
            in the visible response. This is implemented as a system prompt instruction
            since the API doesn't directly support token limiting for visible reasoning.
            Defaults to None (no limit).
        
        reasoning_format (Optional[str]): Format instructions for reasoning.
            Defaults to None (use Claude's default format).
            
        show_working (Optional[bool]): Whether to explicitly ask Claude to show its working.
            Defaults to True for high/medium reasoning levels, False for low/none.
            
        extended_thinking (bool): Whether to enable extended thinking mode, which allows
            Claude to perform more extensive internal reasoning before generating a response.
            Defaults to False.
            
        budget_tokens (Optional[int]): Maximum tokens Claude is allowed to use for its
            internal reasoning process when extended_thinking is enabled. Larger budgets
            can improve response quality for complex problems. Must be less than max_tokens.
            Defaults to None (let the API choose an appropriate budget).
            
        format (Optional[str]): Alias for reasoning_format for compatibility with templates.
            Will be stored in reasoning_format.
    """
    level: ReasoningLevel = "auto"
    max_reasoning_tokens: Optional[int] = None
    reasoning_format: Optional[str] = None
    show_working: Optional[bool] = None
    extended_thinking: bool = False
    budget_tokens: Optional[int] = None
    format: Optional[str] = None
    
    def __post_init__(self):
        # Default show_working based on reasoning level if not explicitly set
        if self.show_working is None:
            self.show_working = self.level in ["high", "medium", "auto"]
            
        # Handle format alias for reasoning_format
        if self.format is not None and self.reasoning_format is None:
            self.reasoning_format = self.format
            self.format = None
    
    def to_api_params(self) -> Dict[str, Any]:
        """
        Convert the configuration to API parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of parameters to add to the Claude API request.
        """
        params = {
            "reasoning": self.level
        }
        
        # Add extended thinking parameters if enabled
        if self.extended_thinking:
            params["thinking"] = True
            
            if self.budget_tokens is not None:
                params["budget_tokens"] = self.budget_tokens
                
        return params
    
    def to_system_instruction(self) -> str:
        """
        Generate system instructions to implement reasoning configuration.
        
        This creates instructions for aspects of reasoning configuration that
        aren't directly supported by the API parameter.
        
        Returns:
            str: System instructions to control reasoning behavior.
        """
        instructions = []
        
        # Add token limit instruction if specified
        if self.max_reasoning_tokens:
            instructions.append(
                f"When showing your reasoning, limit it to approximately {self.max_reasoning_tokens} tokens "
                f"(roughly {self.max_reasoning_tokens // 4} words) to be concise."
            )
        
        # Add format instruction if specified
        if self.reasoning_format:
            instructions.append(f"Format your reasoning as follows: {self.reasoning_format}")
        
        # Add show working instruction based on configuration
        if self.show_working and self.level not in ["none", "low"]:
            instructions.append(
                "Show your detailed step-by-step thinking before giving your final answer."
            )
        elif not self.show_working and self.level not in ["none"]:
            instructions.append(
                "Think through the problem carefully but only show your final answer, not your full reasoning process."
            )
        
        # Join all instructions
        if instructions:
            return "REASONING INSTRUCTIONS:\n" + "\n".join(instructions)
        else:
            return ""

# Example reasoning templates
REASONING_TEMPLATES = {
    "step_by_step": {
        "level": "high",
        "format": """
        Step 1: [First step of reasoning]
        Step 2: [Second step of reasoning]
        ...
        Conclusion: [Final answer]
        """
    },
    "concise": {
        "level": "medium",
        "max_reasoning_tokens": 200,
        "format": """
        Thoughts: [Brief reasoning process]
        Answer: [Final answer]
        """
    },
    "mathematical": {
        "level": "high",
        "format": """
        Given:
        - [List of given information]
        
        Approach:
        [Explanation of method]
        
        Calculation:
        [Step-by-step calculation]
        
        Result:
        [Final answer with units if applicable]
        """
    },
    "direct": {
        "level": "none",
        "show_working": False
    },
    "extended_thinking": {
        "level": "medium",
        "extended_thinking": True,
        "budget_tokens": 8000,
        "format": """
        REASONING PROCESS:
        [Summary of key insights from extended thinking]
        
        ANSWER:
        [Final answer with confidence level]
        """
    },
    "deep_analysis": {
        "level": "high",
        "extended_thinking": True,
        "budget_tokens": 16000,
        "format": """
        ANALYSIS SUMMARY:
        [Key insights from extended thinking]
        
        DETAILED REASONING:
        [Step-by-step explanation]
        
        CONCLUSION:
        [Final answer with confidence level and limitations]
        """
    }
}

def get_reasoning_config(template_name: str = None, **kwargs) -> ClaudeReasoningConfig:
    """
    Get a reasoning configuration, optionally based on a template.
    
    Args:
        template_name (str, optional): Name of the template to use.
            Must be one of the keys in REASONING_TEMPLATES.
        **kwargs: Override any template parameters.
    
    Returns:
        ClaudeReasoningConfig: A reasoning configuration.
    
    Raises:
        ValueError: If the template name is unknown.
    """
    if template_name:
        if template_name not in REASONING_TEMPLATES:
            raise ValueError(f"Unknown reasoning template: {template_name}. "
                            f"Valid templates: {list(REASONING_TEMPLATES.keys())}")
        
        # Start with template and override with kwargs
        config_dict = REASONING_TEMPLATES[template_name].copy()
        config_dict.update(kwargs)
        
        return ClaudeReasoningConfig(**config_dict)
    
    # No template, just use kwargs
    return ClaudeReasoningConfig(**kwargs)