# Tool Use Extension Plan: Enhancing AI Battle with Tool Use and Model Collaboration

This extension plan focuses on implementing tool use capabilities and creating a collaborative system where smaller local models can be paired with larger models to overcome challenges. This approach can significantly enhance the framework's capabilities while optimizing for both performance and cost.

## 1. Tool Use Architecture

### 1.1 Current Limitations

The current framework doesn't have a structured approach for models to use external tools or APIs. Models are primarily used for generating text responses based on prompts and conversation history, without the ability to:

- Execute code
- Search the web
- Access external databases
- Manipulate files
- Use specialized tools for specific domains

### 1.2 Proposed Tool Use Architecture

```mermaid
graph TD
    CM[Conversation Manager] --> TM[Tool Manager]
    TM --> TR[Tool Registry]
    TR --> T1[Code Execution Tool]
    TR --> T2[Web Search Tool]
    TR --> T3[File Manipulation Tool]
    TR --> T4[Database Query Tool]
    TR --> T5[Domain-Specific Tools]
    
    CM --> MM[Model Manager]
    MM --> M1[Model 1]
    MM --> M2[Model 2]
    
    M1 --> TM: Tool Request
    TM --> M1: Tool Response
```

### 1.3 Tool Registry and Tool Interface

Create a standardized interface for tools:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class Tool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the tool."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the tool."""
        pass
        
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters schema for the tool."""
        pass
        
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        pass
```

Implement a tool registry to manage available tools:

```python
class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools = {}
        
    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools with their descriptions and parameters."""
        return {
            name: {
                "description": tool.description,
                "parameters": tool.parameters
            }
            for name, tool in self._tools.items()
        }
```

### 1.4 Example Tool Implementations

#### Code Execution Tool

```python
import subprocess
from typing import Dict, Any

class CodeExecutionTool(Tool):
    """Tool for executing code in various languages."""
    
    @property
    def name(self) -> str:
        return "code_execution"
        
    @property
    def description(self) -> str:
        return "Execute code in various programming languages."
        
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "language": {
                "type": "string",
                "description": "Programming language (python, javascript, bash)",
                "enum": ["python", "javascript", "bash"]
            },
            "code": {
                "type": "string",
                "description": "Code to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "default": 10
            }
        }
        
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code and return the result."""
        language = parameters.get("language", "python")
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", 10)
        
        if language == "python":
            return self._execute_python(code, timeout)
        elif language == "javascript":
            return self._execute_javascript(code, timeout)
        elif language == "bash":
            return self._execute_bash(code, timeout)
        else:
            return {"error": f"Unsupported language: {language}"}
            
    def _execute_python(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute Python code."""
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"error": "Execution timed out"}
        except Exception as e:
            return {"error": str(e)}
```

#### Web Search Tool

```python
import aiohttp
from typing import Dict, Any

class WebSearchTool(Tool):
    """Tool for searching the web."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    @property
    def name(self) -> str:
        return "web_search"
        
    @property
    def description(self) -> str:
        return "Search the web for information."
        
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        }
        
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search and return results."""
        query = parameters.get("query", "")
        num_results = parameters.get("num_results", 5)
        
        if not query:
            return {"error": "Query is required"}
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.search.com/search",
                    params={
                        "q": query,
                        "num": num_results,
                        "key": self.api_key
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "results": data.get("results", []),
                            "total": data.get("total", 0)
                        }
                    else:
                        return {"error": f"Search API error: {response.status}"}
        except Exception as e:
            return {"error": str(e)}
```

### 1.5 Tool Manager

Create a tool manager to handle tool requests and responses:

```python
class ToolManager:
    """Manager for handling tool requests and responses."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        
    async def handle_tool_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool request and return the response."""
        tool_name = request.get("tool")
        parameters = request.get("parameters", {})
        
        if not tool_name:
            return {"error": "Tool name is required"}
            
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool not found: {tool_name}"}
            
        try:
            return await tool.execute(parameters)
        except Exception as e:
            return {"error": f"Tool execution error: {str(e)}"}
            
    def get_tool_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get descriptions of all available tools."""
        return self.registry.list_tools()
```

## 2. Model Collaboration Architecture

### 2.1 Current Limitations

The current framework primarily uses models independently, without a mechanism for models to collaborate or assist each other when facing challenges. This limits the potential for:

- Cost optimization by using smaller models for most tasks
- Performance optimization by using larger models only when needed
- Handling complex tasks that require different model strengths

### 2.2 Proposed Model Collaboration Architecture

```mermaid
graph TD
    CM[Conversation Manager] --> MCP[Model Collaboration Protocol]
    MCP --> PM[Primary Model]
    MCP --> SM[Secondary Model]
    
    PM --> SD[Stuck Detection]
    SD --> MCP: Request Help
    MCP --> SM: Delegate Task
    SM --> MCP: Provide Solution
    MCP --> PM: Continue with Solution
```

### 2.3 Stuck Detection Mechanism

Create a mechanism to detect when a model is "stuck" and needs assistance:

```python
class StuckDetector:
    """Detector for identifying when a model is stuck."""
    
    def __init__(self, threshold: float = 0.7, max_retries: int = 3):
        self.threshold = threshold
        self.max_retries = max_retries
        self.retry_count = 0
        
    def is_stuck(self, response: str, prompt: str) -> bool:
        """Determine if the model is stuck based on its response."""
        # Check for explicit indicators of being stuck
        stuck_phrases = [
            "I'm not sure how to",
            "I don't know how to",
            "I'm unable to",
            "I can't perform this task",
            "This is beyond my capabilities"
        ]
        
        if any(phrase in response.lower() for phrase in stuck_phrases):
            return True
            
        # Check for response quality
        quality_score = self._calculate_quality_score(response, prompt)
        if quality_score < self.threshold:
            self.retry_count += 1
            return self.retry_count >= self.max_retries
            
        # Reset retry count if response is good
        self.retry_count = 0
        return False
        
    def _calculate_quality_score(self, response: str, prompt: str) -> float:
        """Calculate a quality score for the response."""
        # Implement quality metrics:
        # 1. Response length relative to prompt complexity
        # 2. Semantic relevance to the prompt
        # 3. Presence of actionable content
        # 4. Absence of hedging or uncertainty
        
        # Simplified implementation for illustration
        if len(response) < 20:
            return 0.3
            
        relevance = self._calculate_relevance(response, prompt)
        actionable = self._contains_actionable_content(response)
        certainty = 1.0 - self._uncertainty_level(response)
        
        return (relevance + actionable + certainty) / 3.0
        
    def _calculate_relevance(self, response: str, prompt: str) -> float:
        """Calculate semantic relevance between response and prompt."""
        # Implement semantic similarity calculation
        # For illustration, return a placeholder value
        return 0.8
        
    def _contains_actionable_content(self, response: str) -> float:
        """Determine if the response contains actionable content."""
        # Check for code blocks, step-by-step instructions, etc.
        if "```" in response or "Step 1:" in response:
            return 1.0
        return 0.5
        
    def _uncertainty_level(self, response: str) -> float:
        """Calculate the level of uncertainty in the response."""
        uncertainty_phrases = [
            "I think",
            "perhaps",
            "maybe",
            "might be",
            "not sure",
            "could be"
        ]
        
        count = sum(1 for phrase in uncertainty_phrases if phrase in response.lower())
        return min(count / 10.0, 1.0)
```

### 2.4 Model Collaboration Protocol

Create a protocol for models to collaborate:

```python
class ModelCollaborationProtocol:
    """Protocol for enabling collaboration between models."""
    
    def __init__(self, 
                 primary_model: BaseClient, 
                 secondary_model: BaseClient,
                 stuck_detector: StuckDetector):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.stuck_detector = stuck_detector
        
    async def generate_response(self, 
                              prompt: str, 
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              file_data: Dict[str, Any] = None) -> str:
        """Generate a response using model collaboration."""
        # First, try with the primary model
        primary_response = await self.primary_model.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            history=history,
            file_data=file_data
        )
        
        # Check if the primary model is stuck
        if self.stuck_detector.is_stuck(primary_response, prompt):
            # Create a meta-prompt for the secondary model
            meta_prompt = self._create_meta_prompt(prompt, primary_response, history)
            
            # Get help from the secondary model
            secondary_response = await self.secondary_model.generate_response(
                prompt=meta_prompt,
                system_instruction=self._create_helper_instruction(),
                history=history,
                file_data=file_data
            )
            
            # Extract the solution from the secondary model's response
            solution = self._extract_solution(secondary_response)
            
            # Create a new prompt for the primary model with the solution
            enhanced_prompt = self._create_enhanced_prompt(prompt, solution)
            
            # Try again with the primary model
            return await self.primary_model.generate_response(
                prompt=enhanced_prompt,
                system_instruction=system_instruction,
                history=history,
                file_data=file_data
            )
        
        return primary_response
        
    def _create_meta_prompt(self, 
                          prompt: str, 
                          primary_response: str,
                          history: List[Dict[str, str]]) -> str:
        """Create a meta-prompt for the secondary model."""
        return f"""
        I need your help to solve a problem that a smaller model is struggling with.
        
        Original prompt:
        {prompt}
        
        The model's response:
        {primary_response}
        
        The model appears to be stuck or unable to complete this task effectively.
        Please provide a clear, step-by-step solution that the smaller model can use.
        Focus on breaking down complex concepts and providing concrete examples.
        """
        
    def _create_helper_instruction(self) -> str:
        """Create a system instruction for the helper model."""
        return """
        You are an expert AI assistant helping another AI model that is stuck on a task.
        Your goal is to provide clear, detailed guidance that helps the other model complete its task.
        
        Guidelines:
        1. Break down complex problems into simple steps
        2. Provide concrete examples and templates
        3. Explain concepts clearly and concisely
        4. Focus on actionable advice rather than general explanations
        5. If code is needed, provide complete, working examples
        
        Your response should be structured in a way that's easy for another AI to understand and implement.
        """
        
    def _extract_solution(self, secondary_response: str) -> str:
        """Extract the solution from the secondary model's response."""
        # For simplicity, use the entire response
        # In a real implementation, this could parse and extract the most relevant parts
        return secondary_response
        
    def _create_enhanced_prompt(self, original_prompt: str, solution: str) -> str:
        """Create an enhanced prompt with the solution for the primary model."""
        return f"""
        {original_prompt}
        
        Here's a helpful approach to solve this:
        {solution}
        
        Please use this guidance to complete the task. You can adapt this approach as needed.
        """
```

### 2.5 Integration with Conversation Manager

Integrate the model collaboration protocol with the conversation manager:

```python
class CollaborativeConversationManager(ConversationManager):
    """Enhanced conversation manager with model collaboration capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collaboration_protocols = {}
        
    def setup_collaboration(self, 
                          primary_model: str, 
                          secondary_model: str,
                          threshold: float = 0.7,
                          max_retries: int = 3) -> None:
        """Set up collaboration between primary and secondary models."""
        primary_client = self._get_client(primary_model)
        secondary_client = self._get_client(secondary_model)
        
        if not primary_client or not secondary_client:
            raise ValueError(f"Could not initialize required clients: {primary_model}, {secondary_model}")
            
        stuck_detector = StuckDetector(threshold=threshold, max_retries=max_retries)
        protocol = ModelCollaborationProtocol(
            primary_model=primary_client,
            secondary_model=secondary_client,
            stuck_detector=stuck_detector
        )
        
        self.collaboration_protocols[primary_model] = protocol
        
    async def run_conversation_turn_collaborative(self,
                                               prompt: str,
                                               model_type: str,
                                               mode: str,
                                               role: str,
                                               file_data: Dict[str, Any] = None,
                                               system_instruction: str = None) -> str:
        """Run a conversation turn with model collaboration if available."""
        # Check if we have a collaboration protocol for this model
        if model_type in self.collaboration_protocols:
            protocol = self.collaboration_protocols[model_type]
            
            # Use the collaboration protocol to generate a response
            response = await protocol.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                history=self.conversation_history.copy(),
                file_data=file_data
            )
            
            # Add the response to the conversation history
            self.conversation_history.append({"role": role, "content": response})
            
            return response
        else:
            # Fall back to the standard method if no collaboration is set up
            return await self.run_conversation_turn(
                prompt=prompt,
                model_type=model_type,
                client=self._get_client(model_type),
                mode=mode,
                role=role,
                file_data=file_data,
                system_instruction=system_instruction
            )
```

## 3. Tool Use by Models

### 3.1 Tool Use Protocol

Create a protocol for models to use tools:

```python
class ToolUseProtocol:
    """Protocol for enabling models to use tools."""
    
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        
    def create_tool_use_instruction(self) -> str:
        """Create a system instruction for tool use."""
        tools = self.tool_manager.get_tool_descriptions()
        
        instruction = """
        You have access to the following tools:
        
        """
        
        for name, tool in tools.items():
            instruction += f"""
            {name}: {tool['description']}
            Parameters:
            """
            
            for param_name, param in tool['parameters'].items():
                required = "required" if param.get("required", False) else "optional"
                default = f", default: {param.get('default')}" if "default" in param else ""
                instruction += f"  - {param_name} ({param['type']}, {required}{default}): {param['description']}\n"
                
        instruction += """
        To use a tool, respond with:
        
        ```tool
        {
            "tool": "tool_name",
            "parameters": {
                "param1": "value1",
                "param2": "value2"
            }
        }
        ```
        
        The tool will be executed, and the result will be provided to you.
        """
        
        return instruction
        
    async def process_response(self, response: str) -> str:
        """Process a response to handle tool use requests."""
        # Check if the response contains a tool use request
        tool_match = re.search(r"```tool\s+(.*?)\s+```", response, re.DOTALL)
        if not tool_match:
            return response
            
        try:
            # Extract and parse the tool request
            tool_request = json.loads(tool_match.group(1))
            
            # Execute the tool
            tool_response = await self.tool_manager.handle_tool_request(tool_request)
            
            # Format the tool response
            formatted_response = f"""
            I used the {tool_request.get('tool')} tool with the provided parameters.
            
            Tool Response:
            ```
            {json.dumps(tool_response, indent=2)}
            ```
            
            Based on this information, 
            """
            
            # Replace the tool request with the formatted response
            return response.replace(tool_match.group(0), formatted_response)
        except json.JSONDecodeError:
            return response + "\n\nNote: I tried to use a tool, but the request format was invalid."
        except Exception as e:
            return response + f"\n\nNote: I tried to use a tool, but an error occurred: {str(e)}"
```

### 3.2 Integration with Model Clients

Enhance model clients to support tool use:

```python
class ToolCapableClient(BaseClient):
    """Base client with tool use capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_protocol = None
        
    def set_tool_protocol(self, protocol: ToolUseProtocol) -> None:
        """Set the tool use protocol for this client."""
        self.tool_protocol = protocol
        
    async def generate_response(self,
                              prompt: str,
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              file_data: Dict[str, Any] = None) -> str:
        """Generate a response with tool use capabilities."""
        # Enhance system instruction with tool use instructions if available
        enhanced_instruction = system_instruction
        if self.tool_protocol:
            tool_instruction = self.tool_protocol.create_tool_use_instruction()
            if system_instruction:
                enhanced_instruction = f"{system_instruction}\n\n{tool_instruction}"
            else:
                enhanced_instruction = tool_instruction
                
        # Generate the initial response
        response = await super().generate_response(
            prompt=prompt,
            system_instruction=enhanced_instruction,
            history=history,
            file_data=file_data
        )
        
        # Process the response for tool use if protocol is available
        if self.tool_protocol:
            return await self.tool_protocol.process_response(response)
            
        return response
```

### 3.3 Tool Use Configuration

Extend the configuration system to support tool use:

```yaml
discussion:
  # Existing fields...
  
  # Tool configuration
  tools:
    enabled: true
    registry:
      - name: "code_execution"
        enabled: true
        parameters:
          timeout: 15
      - name: "web_search"
        enabled: true
        parameters:
          api_key: "${WEB_SEARCH_API_KEY}"
      - name: "file_manipulation"
        enabled: false
  
  # Model collaboration configuration
  collaboration:
    enabled: true
    pairs:
      - primary: "ollama-gemma3-1b"
        secondary: "claude-3-sonnet"
        threshold: 0.7
        max_retries: 2
      - primary: "ollama-phi4"
        secondary: "gpt-4o"
        threshold: 0.6
        max_retries: 3
```

## 4. Example Use Cases

### 4.1 Code Generation and Debugging

```python
# Example configuration for code generation and debugging
config = {
    "discussion": {
        "turns": 5,
        "models": {
            "model1": {
                "type": "ollama-phi4",
                "role": "human"
            },
            "model2": {
                "type": "claude-3-sonnet",
                "role": "assistant"
            }
        },
        "tools": {
            "enabled": True,
            "registry": [
                {
                    "name": "code_execution",
                    "enabled": True,
                    "parameters": {
                        "timeout": 15
                    }
                }
            ]
        },
        "collaboration": {
            "enabled": True,
            "pairs": [
                {
                    "primary": "ollama-phi4",
                    "secondary": "claude-3-sonnet",
                    "threshold": 0.7,
                    "max_retries": 2
                }
            ]
        },
        "goal": "Write a Python script to analyze a CSV file and generate a summary report."
    }
}

# Initialize manager with the configuration
manager = CollaborativeConversationManager.from_config(config)

# Run the conversation
result = await manager.run_discussion()
```

### 4.2 Research Assistant with Web Search

```python
# Example configuration for research assistant
config = {
    "discussion": {
        "turns": 8,
        "models": {
            "model1": {
                "type": "ollama-gemma3-1b",
                "role": "human"
            },
            "model2": {
                "type": "gpt-4o",
                "role": "assistant"
            }
        },
        "tools": {
            "enabled": True,
            "registry": [
                {
                    "name": "web_search",
                    "enabled": True,
                    "parameters": {
                        "api_key": "${WEB_SEARCH_API_KEY}"
                    }
                }
            ]
        },
        "collaboration": {
            "enabled": True,
            "pairs": [
                {
                    "primary": "ollama-gemma3-1b",
                    "secondary": "gpt-4o",
                    "threshold": 0.6,
                    "max_retries": 3
                }
            ]
        },
        "goal": "Research the latest advancements in quantum computing and summarize the key findings."
    }
}

# Initialize manager with the configuration
manager = CollaborativeConversationManager.from_config(config)

# Run the conversation
result = await manager.run_discussion()
```

## 5. Implementation Plan

### 5.1 Phase 1: Tool Use Framework

1. Implement the `Tool` abstract base class and `ToolRegistry`
2. Create basic tools (code execution, web search)
3. Implement the `ToolManager` for handling tool requests
4. Enhance model clients with tool use capabilities
5. Extend the configuration system to support tool configuration

### 5.2 Phase 2: Model Collaboration

1. Implement the `StuckDetector` for identifying when models are stuck
2. Create the `ModelCollaborationProtocol` for model collaboration
3. Enhance the conversation manager with collaboration capabilities
4. Extend the configuration system to support collaboration configuration
5. Implement the collaborative conversation turn method

### 5.3 Phase 3: Integration and Testing

1. Integrate tool use and model collaboration
2. Create comprehensive tests for different scenarios
3. Implement example use cases
4. Update documentation and examples

## 6. Conclusion

This extension plan addresses the need for tool use capabilities and model collaboration in the AI Battle framework. By implementing these features, the framework will be able to:

1. **Enable Tool Use**: Allow models to use external tools and APIs to enhance their capabilities
2. **Optimize Performance and Cost**: Use smaller local models for most tasks and larger models only when needed
3. **Handle Complex Tasks**: Combine the strengths of different models to handle complex tasks more effectively

The proposed architecture maintains the strengths of the current framework while adding powerful new capabilities that will significantly enhance its usefulness and flexibility.