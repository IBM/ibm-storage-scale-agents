"""Common utility functions for agent configuration and MCP setup."""

import asyncio
import base64
import configparser
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import httpx
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.utils.constants import SEPARATOR_LINE, TOOL_CONFIGS

# Null placeholder strings that LLMs may generate for optional parameters
NULL_PLACEHOLDERS = ('<nil>', 'null', 'NULL', '<null>', 'nil', '')

# Global semaphore to enforce sequential tool execution
# This ensures only one tool executes at a time across all tool instances
_TOOL_EXECUTION_SEMAPHORE = None


def get_tool_execution_semaphore():
    """Get or create the global tool execution semaphore.

    This semaphore ensures that only one tool can execute at a time,
    preventing concurrent tool calls that would cause SSE connection issues.
    """
    global _TOOL_EXECUTION_SEMAPHORE
    if _TOOL_EXECUTION_SEMAPHORE is None:
        _TOOL_EXECUTION_SEMAPHORE = asyncio.Semaphore(1)
    return _TOOL_EXECUTION_SEMAPHORE


def decode_policy_contents(base64_content: str) -> str:
    """Decode base64-encoded policy contents to plain text.

    Args:
        base64_content: Base64-encoded policy content

    Returns:
        Decoded plain text policy content
 
    Raises:
        ValueError: If content is not valid base64 or UTF-8
    """
    try:
        decoded_bytes = base64.b64decode(base64_content, validate=True)
        return decoded_bytes.decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to decode policy contents: {str(e)}") from e


def process_policy_contents(tool_name: str, filtered_kwargs: dict, logger: logging.Logger) -> None:
    """Process policy_contents parameter: normalize to base64 encoding for MCP transmission.
    
    This is a shared helper for both test_policy and update_policy tools.
    Modifies filtered_kwargs in place.
    
    The decode/encode cycle serves two purposes:
    1. Normalizes input - accepts both plain text and base64-encoded content
    2. Ensures consistent base64 encoding for MCP server transmission
    
    Args:
        tool_name: Name of the tool being called
        filtered_kwargs: Dictionary of tool arguments (modified in place)
        logger: Logger instance for output
    """
    if "policy_contents" not in filtered_kwargs:
        return
        
    raw = filtered_kwargs["policy_contents"]
    if not isinstance(raw, str):
        return
    
    # Normalize input: if already base64, decode to plain text first
    # This allows the function to accept both plain text and base64-encoded input
    try:
        decoded_bytes = base64.b64decode(raw, validate=True)
        raw = decoded_bytes.decode("utf-8")
    except Exception:
        # Input is not base64-encoded, treating as plain text
        logger.debug(f"Input for {tool_name} is not base64-encoded, treating as plain text")
    
    # Log the decoded policy content before encoding
    logger.debug(f"Raw policy content for {tool_name} (decoded):")
    logger.debug(raw)
    
    # Always encode to base64 for consistent MCP transmission
    encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
    filtered_kwargs["policy_contents"] = encoded


def filter_tool_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out None and LLM-generated placeholder strings for optional parameters.
    
    Some LLMs may generate '<nil>', 'null', etc. instead of omitting optional params.
    
    Args:
        kwargs: Dictionary of tool arguments
        
    Returns:
        Filtered dictionary with None and placeholder values removed
    """
    filtered = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        # Skip string placeholders that represent "no value"
        if isinstance(v, str) and v in NULL_PLACEHOLDERS:
            continue
        filtered[k] = v
    return filtered


def create_tool_input_model(tool_name: str, args_config: Dict[str, Any]) -> Type[BaseModel]:
    """Dynamically create a Pydantic model for a tool's input schema.
    
    Args:
        tool_name: Name of the tool
        args_config: Dictionary of argument configurations
        
    Returns:
        Dynamically created Pydantic BaseModel class
    """
    fields = {}
    annotations = {}
    
    for arg_name, arg_config in args_config.items():
        arg_type = arg_config["type"]
        arg_desc = arg_config["description"]
        is_optional = arg_config.get("optional", False)

        if is_optional:
            annotations[arg_name] = Optional[arg_type]
            fields[arg_name] = Field(default=None, description=arg_desc)
        else:
            annotations[arg_name] = arg_type
            fields[arg_name] = Field(description=arg_desc)
    
    # Create dynamic Pydantic model with proper annotations and defaults
    return type(f"{tool_name}_input", (BaseModel,), {"__annotations__": annotations, **fields})


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/agent.log",
    log_format: str = "json",
    max_bytes: int = 10485760,
    backup_count: int = 5,
    console_level: Optional[str] = None,
) -> logging.Logger:
    """Setup logging based on configuration.
    
    This function is idempotent - calling it multiple times with the same log_file
    will not create duplicate handlers.

    Args:
        log_level: Logging level for file handler (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Format type ('json' or 'text')
        max_bytes: Maximum size of log file before rotation in bytes
        backup_count: Number of backup files to keep
        console_level: Logging level for console handler (defaults to INFO to reduce clutter)

    Returns:
        Configured logger instance
    """
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)

    # Console defaults to INFO to avoid DEBUG clutter, unless explicitly set
    if console_level is None:
        console_level_value = logging.INFO
    else:
        console_level_value = getattr(logging, console_level.upper(), logging.INFO)

    if log_format == "json":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root_logger = logging.getLogger()
    # Set root logger to the lowest level (DEBUG) to capture everything
    # Individual handlers will filter based on their own levels
    root_logger.setLevel(logging.DEBUG)
    
    # Only setup handlers once (on first call)
    if not root_logger.handlers:
        # Suppress noisy third-party loggers: use WARNING level
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        # Console handler: defaults to INFO to reduce clutter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level_value)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler: uses configured log_level (typically DEBUG for full details)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setLevel(log_level_value)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    return root_logger


def setup_agent_logging(config: configparser.ConfigParser, default_log_path: str) -> None:
    """Setup logging for an agent from configuration.
    
    Uses a shared log file (log_path) for all agents.

    Args:
        config: ConfigParser instance with agent configuration
        default_log_path: Default log file path if log_path not found in config
    """
    logging_config = config["logging"] if "logging" in config else {}
    # Use shared log_path for all agents, fall back to default_log_path if not found
    shared_log_path = logging_config.get("log_path", default_log_path)

    # Get console_level from config, default to INFO to reduce console clutter
    console_level = logging_config.get("console_level", "INFO")

    setup_logging(
        log_level=logging_config.get("level", "DEBUG"),  # File gets DEBUG for full details
        log_file=shared_log_path,
        log_format=logging_config.get("format", "json"),
        max_bytes=int(logging_config.get("max_bytes", "10485760")),
        backup_count=int(logging_config.get("backup_count", "5")),
        console_level=console_level,  # Console gets INFO by default
    )


class MCPClient:
    """Client for interacting with MCP server via HTTP."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize MCP client.

        Args:
            base_url: Base URL of the MCP server
            timeout: Timeout for HTTP requests in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=timeout, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        )
        self._tools_cache = None
        self._session_id = None
        self.logger = logging.getLogger(__name__)

    def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response format.

        Args:
            text: Raw SSE response text

        Returns:
            Parsed JSON data from the SSE message
        """
        import json

        self.logger.debug(f"Parsing SSE response, length: {len(text)}")

        lines = text.strip().split("\n")
        parsed_data = []

        for line in lines:
            if line.startswith("data: "):
                json_str = line[6:]
                parsed = json.loads(json_str)
                parsed_data.append(parsed)

        if parsed_data:
            self.logger.debug(f"Found {len(parsed_data)} data lines")
            for idx, parsed in enumerate(parsed_data):
                self.logger.debug(
                    f"Data line {idx}: method={parsed.get('method', 'N/A')}, has result={('result' in parsed)}"
                )

            return parsed_data[-1]

        self.logger.debug("No data lines found, parsing entire response")
        return json.loads(text)

    async def _ensure_session(self):
        """Ensure we have a valid session ID."""
        if self._session_id is not None:
            return

        # Initialize session with the server
        response = await self.client.post(
            f"{self.base_url}",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "scale-agents", "version": "1.0.0"},
                },
            },
        )
        response.raise_for_status()

        # Extract session ID from response headers
        self._session_id = response.headers.get("mcp-session-id")
        if not self._session_id:
            raise Exception("Server did not provide session ID")

        # Update client headers with session ID
        self.client.headers["mcp-session-id"] = self._session_id

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from MCP server.

        Returns:
            List of tool definitions
        """
        if self._tools_cache is not None:
            return self._tools_cache

        await self._ensure_session()

        response = await self.client.post(
            f"{self.base_url}", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        response.raise_for_status()

        # Parse SSE response
        result = self._parse_sse_response(response.text)

        if "result" in result and "tools" in result["result"]:
            self._tools_cache = result["result"]["tools"]
            return self._tools_cache
        return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result (extracted from content)
        """
        await self._ensure_session()

        response = await self.client.post(
            f"{self.base_url}",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        response.raise_for_status()

        result = self._parse_sse_response(response.text)

        if "error" in result:
            raise Exception(f"MCP tool error: {result['error']}")

        if "result" in result:
            mcp_result = result["result"]
            self.logger.debug(f"Raw MCP result keys: {mcp_result.keys()}")

            if "content" in mcp_result and isinstance(mcp_result["content"], list):
                self.logger.debug(f"Found content array with {len(mcp_result['content'])} items")
                for idx, content_item in enumerate(mcp_result["content"]):
                    self.logger.debug(f"Content item {idx}: type={content_item.get('type')}")
                    if content_item.get("type") == "text" and "text" in content_item:
                        import json

                        text_content = content_item["text"]
                        self.logger.debug(f"Text content (first 200 chars): {text_content[:200]}")

                        try:
                            parsed = json.loads(text_content)
                            self.logger.debug("Successfully parsed text content as JSON")
                            return parsed
                        except json.JSONDecodeError:
                            self.logger.debug("Text content is not JSON, returning as plain text")
                            return {"text": text_content}

            if "structuredContent" in mcp_result:
                self.logger.debug("Using structuredContent")
                return mcp_result["structuredContent"]

            self.logger.debug("Fallback: returning entire result")
            return mcp_result

        return result

    async def cleanup(self):
        """Cleanup client resources."""
        await self.client.aclose()


def load_agent_config(config_path: Path, default_log_path: str = "logs/agents.log"):
    """Load and validate agent configuration from INI file and setup logging.

    Args:
        config_path: Path to the configuration file
        default_log_path: Default log file path if log_path not found in config

    Returns:
        Tuple of (config, llm, mcp_client)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required sections or keys are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\nPlease create the file with [llm] and [mcp] sections."
        )

    config = configparser.ConfigParser()
    config.read(config_path)

    # Validate required sections
    required_sections = ["llm", "mcp"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section [{section}] in config file")

    # Validate required keys
    if "model_name" not in config["llm"]:
        raise ValueError("Missing 'model_name' in [llm] section")

    # Validate MCP configuration based on transport
    transport = config["mcp"].get("transport", "http").lower()
    if transport in ("sse", "http") and "url" not in config["mcp"]:
        raise ValueError(f"Missing 'url' in [mcp] section for {transport.upper()} transport")

    # Setup logging using shared log_path from config
    setup_agent_logging(config, default_log_path)
    
    llm_config = config["llm"]
    mcp_config = config["mcp"]
    
    model_name = llm_config["model_name"].replace("ollama_chat/", "")
    llm_timeout = float(llm_config.get("timeout", "60.0"))
    
    llm = ChatOllama(model=model_name, temperature=0, verbose=True, timeout=llm_timeout)
    mcp_client = create_mcp_client(mcp_config)
    
    return config, llm, mcp_client


def create_mcp_client(mcp_config: configparser.SectionProxy) -> MCPClient:
    """Create an MCP client based on configuration.

    Args:
        mcp_config: MCP section from configuration

    Returns:
        Configured MCPClient instance

    Raises:
        ValueError: If transport type is unsupported
    """
    transport = mcp_config.get("transport", "http").lower()

    if transport in ("sse", "http"):
        # HTTP/SSE transport
        mcp_url = mcp_config["url"].strip('"')  # Remove quotes if present
        timeout = float(mcp_config.get("timeout", "30.0"))
        return MCPClient(base_url=mcp_url, timeout=timeout)
    else:
        raise ValueError(
            f"Unsupported transport type: {transport}. Currently only 'http' and 'sse' are supported for LangChain integration."
        )


async def _execute_tool_with_semaphore(
    tool_name: str,
    mcp_client: MCPClient,
    filtered_kwargs: Dict[str, Any],
    logger: logging.Logger,
    require_confirmation: bool = True
) -> str:
    """Execute MCP tool with semaphore lock and optional confirmation.
    
    This helper function encapsulates the common execution logic for both
    confirmation and no-confirmation tool variants.
    
    Args:
        tool_name: Name of the MCP tool to execute
        mcp_client: MCP client instance
        filtered_kwargs: Filtered tool arguments
        logger: Logger instance
        require_confirmation: Whether to require human confirmation before execution
        
    Returns:
        JSON string with tool execution result or error
    """
    if mcp_client is None:
        return f"Error: MCP client not initialized for tool {tool_name}"
    
    # Acquire semaphore to ensure sequential execution
    semaphore = get_tool_execution_semaphore()
    async with semaphore:
        logger.debug(f"[SEQUENTIAL] Acquired execution lock for {tool_name}")
        
        if require_confirmation:
            print(f"\n{SEPARATOR_LINE}")
            print(f"CONFIRMATION REQUIRED: {tool_name}")
            print(f"{SEPARATOR_LINE}")
            print("Arguments:")
            for key, value in filtered_kwargs.items():
                # For policy_contents, decode and display the actual policy text
                if key == "policy_contents" and isinstance(value, str):
                    try:
                        decoded_policy = decode_policy_contents(value)
                        print(f"  {key}:")
                        print("    --- Policy Content (decoded) ---")
                        for line in decoded_policy.split('\n'):
                            print(f"    {line}")
                        print("    --- End Policy Content ---")
                        
                        # Generic duplicate detection: check for multiple rules
                        import re
                        rules = re.findall(r"RULE\s+'([^']+)'", decoded_policy, re.IGNORECASE)
                        if len(rules) > 1:
                            # Check for duplicate rule names
                            unique_rules = set(rules)
                            if len(rules) != len(unique_rules):
                                print("\n    WARNING: Duplicate rule names detected")
                                print("    TIP: If updating an existing rule, remove the old version")
                            else:
                                # Multiple rules with unique names - inform user
                                print(f"\n    INFO: Policy contains {len(rules)} rules")
                                print("    TIP: Review carefully if any rules have similar intent")
                        
                    except Exception:
                        # If decoding fails, show the base64 string
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
            print(f"{SEPARATOR_LINE}")

            approval = interrupt(
                {
                    "type": "human_confirmation",
                    "tool_name": tool_name,
                    "arguments": filtered_kwargs,
                    "message": f"Do you approve execution of {tool_name}?",
                }
            )

            if approval is None or not approval.get("approved", False):
                logger.debug(f"[SEQUENTIAL] Released execution lock for {tool_name} (cancelled)")
                return json.dumps({
                    "status": "error",
                    "isError": True,
                    "message": f"Operation {tool_name} cancelled by user",
                    "cancelled": True
                })
        
        logger.debug(f"{'Approved. ' if require_confirmation else ''}Calling {tool_name} with args: {filtered_kwargs}")
        try:
            result = await mcp_client.call_tool(tool_name, filtered_kwargs)
            logger.debug(f"Result from {tool_name} (type: {type(result)})")
            logger.debug(f"Full result: {json.dumps(result, indent=2)}")
            logger.debug(f"[SEQUENTIAL] Released execution lock for {tool_name} (success)")
            return json.dumps(result, indent=2)
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"[SEQUENTIAL] Released execution lock for {tool_name} (error)")
            return json.dumps({"status": "error", "message": error_msg})


def create_langchain_tool(tool_name: str, mcp_client: MCPClient, require_confirmation: bool = True):
    """Create a LangChain tool wrapper with optional human confirmation.

    This unified function replaces the separate confirmation and no-confirmation variants,
    reducing code duplication while maintaining the same functionality.

    Args:
        tool_name: Name of the MCP tool
        mcp_client: MCP client for execution
        require_confirmation: Whether to require human approval before execution (default: True)

    Returns:
        LangChain StructuredTool instance
        
    Raises:
        ValueError: If tool_name is not found in TOOL_CONFIGS
    """
    # Validate tool exists in configuration
    if tool_name not in TOOL_CONFIGS:
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {', '.join(TOOL_CONFIGS.keys())}"
        )
    
    config = TOOL_CONFIGS[tool_name]
    confirmation_suffix = " (requires confirmation)" if require_confirmation else ""
    tool_description = config.get("description", f"Execute {tool_name} operation{confirmation_suffix}")

    # Create Pydantic model using shared helper
    InputModel = create_tool_input_model(tool_name, config.get("args", {}))

    logger = logging.getLogger(__name__)

    async def tool_func(**kwargs) -> str:
        """Execute MCP tool with optional confirmation."""
        # Filter kwargs using shared helper
        filtered_kwargs = filter_tool_kwargs(kwargs)

        # Process policy_contents for relevant tools
        if tool_name in ("update_policy", "test_policy"):
            process_policy_contents(tool_name, filtered_kwargs, logger)

        result_str = await _execute_tool_with_semaphore(
            tool_name, mcp_client, filtered_kwargs, logger, require_confirmation=require_confirmation
        )
        
        # Auto-decode policy_contents for get_policy tool
        if tool_name == "get_policy":
            try:
                result = json.loads(result_str)
                if isinstance(result, dict) and "policy_contents" in result and result["policy_contents"]:
                    logger.debug("Decoding base64 policy_contents from get_policy response")
                    result["policy_contents"] = decode_policy_contents(result["policy_contents"])
                    result["decoded"] = True
                    logger.debug(f"Decoded policy for {tool_name}:\n{result['policy_contents']}")
                    return json.dumps(result, indent=2)
            except Exception as e:
                logger.warning(f"Failed to decode policy_contents: {e}")
        
        return result_str

    tool = StructuredTool(name=tool_name, description=tool_description, args_schema=InputModel, coroutine=tool_func)

    return tool


def create_langchain_tool_with_confirmation_simple(tool_name: str, mcp_client: MCPClient):
    """Create a LangChain tool wrapper with human confirmation requirement.
    
    DEPRECATED: Use create_langchain_tool(tool_name, mcp_client, require_confirmation=True) instead.
    This function is kept for backward compatibility.

    Args:
        tool_name: Name of the MCP tool
        mcp_client: MCP client for execution

    Returns:
        LangChain StructuredTool instance with confirmation requirement
    """
    return create_langchain_tool(tool_name, mcp_client, require_confirmation=True)


def create_langchain_tool_no_confirmation_simple(tool_name: str, mcp_client: MCPClient):
    """Create a LangChain tool wrapper without confirmation requirement.
    
    DEPRECATED: Use create_langchain_tool(tool_name, mcp_client, require_confirmation=False) instead.
    This function is kept for backward compatibility.

    Args:
        tool_name: Name of the MCP tool
        mcp_client: MCP client for execution

    Returns:
        LangChain StructuredTool instance
    """
    return create_langchain_tool(tool_name, mcp_client, require_confirmation=False)
