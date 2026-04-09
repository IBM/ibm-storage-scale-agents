"""Custom LangGraph workflow for ILM policy operations with enforced sequencing.

Workflow State Machine:
=======================
This workflow implements a strict sequence for ILM policy operations:

    initial ──┬──> get_policy ──> generate_rule ──> verify_pools ──> validate_rule ──> test_policy ──┬──> update_policy ──┬──> apply_policy ──> completed
              │                                                                                      │                    │
              │                                                                                      └──> completed       └──> completed
              │                                                                                           (test only)           (update only)
              └──> cancelled (user cancellation)

State Transitions:
- initial: Neutral state, workflow activates on modification tools or rule generation keywords
- get_policy: Retrieves existing policy (REQUIRED before any modifications)
- generate_rule: LLM generates policy rule with correct IBM Storage Scale syntax
- verify_pools: Validates storage pools exist (REQUIRED after rule generation)
- validate_rule: Uses LLM to check if the newly generated rule duplicates or conflicts with existing rules' intent
- test_policy: Validates policy syntax (REQUIRED before update)
- update_policy: Saves policy to system (optional - depends on user intent)
- apply_policy: Executes policy rules (optional - depends on user intent)
- completed: Terminal state, workflow can be reset for new requests
- cancelled: Terminal state when user cancels operation

Intent Detection:
- Testing intent: Keywords like "test", "validate", "check" → stop after test_policy
- Update intent: Default behavior → stop after update_policy
- Apply intent: Keywords like "apply", "run", "execute" → continue to apply_policy
"""

import json
import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.ilm_agent.rule_generator import generate_ilm_rule_with_llm

logger = logging.getLogger(__name__)

# Schema version for state migrations
STATE_SCHEMA_VERSION = 1

# Constants for error detection in tool responses
ERROR_KEYWORDS = ["error", "failed", "failure", "exception", "no such file", "cancelled"]
ERROR_CHECK_FIELDS = ["status", "text", "result", "message", "output"]

# Constants for workflow activation - tools and keywords that trigger workflow
MODIFICATION_TOOLS = {"update_policy", "apply_policy"}
TESTING_KEYWORDS = ["test", "validate", "check", "verify", "dry run", "dry-run", "simulation"]
RULE_GENERATION_KEYWORDS = ["migrate", "move", "delete", "files older than", "files larger than"]

# Question keywords that indicate informational queries (not action requests)
QUESTION_KEYWORDS = ["how", "what", "why", "when", "where", "can i", "should i", "could i", "would i"]

# Action phrases that indicate actual modification requests
ACTION_PHRASES = ["create a rule", "set up policy", "configure policy", "add a rule", "make a rule"]

# UI constants for status messages
WORKFLOW_STATUS_PREFIX = "[Workflow Status]"

# Input sanitization limits to prevent excessive input
MAX_USER_INPUT_LENGTH = 10000


class ILMWorkflowState(TypedDict):
    """State for ILM workflow with enforced sequencing."""
    
    # Standard message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Workflow tracking
    workflow_step: str  # current step: "initial", "get_policy", "generate_rule", "verify_pools", "validate_rule", "test_policy", "update_policy", "apply_policy", "completed", "cancelled"
    workflow_active: bool  # True when workflow is actively managing a modification sequence
    filesystem: Optional[str]
    
    # Data collected during workflow
    existing_policy: Optional[Dict[str, Any]]
    existing_policy_text: Optional[str]
    storage_pools: Optional[List[str]]
    
    # Rule generation data
    user_request: Optional[str]  # Original user request for rule generation
    target_pool: Optional[str]  # Target pool for migration
    source_pool: Optional[str]  # Source pool (optional)
    generated_rule: Optional[Dict[str, Any]]  # Generated rule with metadata
    rule_generation_error: Optional[str]  # Error during rule generation

    # Rule validation data
    rule_validation_result: Optional[Dict[str, Any]]  # Result of LLM-based rule validation
    has_duplicate_intent: bool  # True if rule duplicates existing rule intent
    has_conflicting_intent: bool  # True if rule conflicts with existing rules

    # Validation flags
    policy_retrieved: bool
    pools_verified: bool
    rule_generated: bool
    rule_validated: bool
    policy_tested: bool
    test_passed: bool
    policy_updated: bool
    
    # Migration results
    migration_stats: Optional[Dict[str, Any]]  # Statistics from apply_policy execution

    # Error tracking
    error_message: Optional[str]
    
    # Performance cache - last user message to avoid repeated traversal
    last_user_message: Optional[str]


def create_initial_state() -> ILMWorkflowState:
    """Create initial workflow state."""
    return ILMWorkflowState(
        messages=[],
        workflow_step="initial",  # Start in neutral state - workflow activates only for actual modifications
        workflow_active=False,  # Workflow not active until modification requested
        filesystem=None,
        existing_policy=None,
        existing_policy_text=None,
        storage_pools=None,
        user_request=None,
        target_pool=None,
        source_pool=None,
        generated_rule=None,
        rule_generation_error=None,
        rule_validation_result=None,
        has_duplicate_intent=False,
        has_conflicting_intent=False,
        policy_retrieved=False,
        pools_verified=False,
        rule_generated=False,
        rule_validated=False,
        policy_tested=False,
        test_passed=False,
        policy_updated=False,
        migration_stats=None,
        error_message=None,
        last_user_message=None,
    )


def _combine_policies(existing_policy_text: str, new_rule_content: str) -> str:
    """Combine existing policy with new rule.
    
    Args:
        existing_policy_text: Existing policy content (decoded)
        new_rule_content: New rule to add
        
    Returns:
        Combined policy with all rules
    """
    if not existing_policy_text:
        return new_rule_content
    
    if not new_rule_content:
        return existing_policy_text
    
    # Ensure both parts end with newline for clean combination
    existing = existing_policy_text.rstrip('\n')
    new_rule = new_rule_content.rstrip('\n')
    
    return f"{existing}\n{new_rule}"


def _sanitize_user_input(text: str) -> str:
    """Sanitize user input for safe inclusion in messages.
    
    Args:
        text: User input text to sanitize
        
    Returns:
        Sanitized text with length limits and escaped special characters
    """
    if not text:
        return ""
    
    # Truncate to maximum length
    if len(text) > MAX_USER_INPUT_LENGTH:
        text = text[:MAX_USER_INPUT_LENGTH] + "... [truncated]"
    
    # Escape control characters and excessive whitespace
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _find_user_request(messages: List[BaseMessage]) -> Optional[str]:
    """Extract the most recent user request from messages.
    
    This is the canonical implementation used throughout the workflow.
    Filters out system-generated messages (those starting with '[').
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Most recent user message content, or None if not found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not msg.content.startswith("["):
            return msg.content
    return None


def _is_question_not_action(user_request: str) -> bool:
    """Determine if user request is an informational question vs action request.
    
    Args:
        user_request: User's message text
        
    Returns:
        True if this appears to be a question, False if it's an action request
    """
    request_lower = user_request.lower().strip()
    
    # Check for question keywords at start
    if any(request_lower.startswith(kw) for kw in QUESTION_KEYWORDS):
        return True
    
    # Check for explicit action phrases (override question detection)
    if any(phrase in request_lower for phrase in ACTION_PHRASES):
        return False
    
    # Check if it ends with question mark
    if request_lower.endswith("?"):
        return True
    
    return False


def _has_intent_keywords(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the intent keywords using word boundary matching.
    
    Uses regex word boundaries to avoid false positives from substrings.
    
    Args:
        text: Text to search
        keywords: List of keywords to search for
        
    Returns:
        True if any keyword found with word boundaries
    """
    if not text:
        return False
    
    text_lower = text.lower()
    return any(re.search(rf'\b{re.escape(kw)}\b', text_lower) for kw in keywords)


def _get_workflow_guidance(state: ILMWorkflowState) -> str:
    """Generate workflow guidance for the LLM agent based on current state.
    
    This guidance is sent to the LLM agent to instruct it on what
    actions to take next and what to communicate to the user.
    
    Note: Workflow reset is handled in validate_tool_call, not here.
    If workflow is in completed/cancelled state, we return empty guidance
    to allow the reset to happen naturally when tool calls are made.
    """
    # If workflow is completed/cancelled, return empty guidance
    # The reset will happen in validate_tool_call when the next tool is called
    if state["workflow_step"] in ["completed", "cancelled"]:
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            # If last message is from user (not system-generated), it's a new request
            if isinstance(last_msg, HumanMessage) and not last_msg.content.startswith("["):
                logger.debug(f"Workflow in {state['workflow_step']} state with new user message - returning empty guidance")
                return ""
        
        # If we're still in cancelled state and it's not a new request, show cancellation message
        if state["workflow_step"] == "cancelled":
            return (
                "AGENT INSTRUCTION: The workflow was cancelled by the user.\n"
                "The policy update operation was not completed.\n"
                "Inform the user that the operation was cancelled and the policy remains unchanged.\n"
                "Do NOT retry test_policy or update_policy unless the user explicitly requests it."
            )
        
        # Completed state with no new request - show completion message
        return (
            "AGENT INSTRUCTION: Workflow completed successfully. "
            "Present the results to the user and DO NOT call any more tools. "
            "The task is finished."
        )

    error = state.get("error_message")
    
    # If there's an error, provide context about it
    if error:
        # Check if it's a cancellation error
        if "cancelled" in error.lower():
            return (
                f"AGENT INSTRUCTION: Operation was cancelled by user: {error}\n"
                "The workflow has been stopped. The user chose not to proceed with this operation.\n"
                "Inform the user that the operation was cancelled.\n"
                "Do NOT retry the same operation unless the user explicitly requests it with different parameters."
            )
        return (
            f"Previous operation encountered an error: {error}\n"
            "You can:\n"
            "1. Retry the operation with corrected parameters\n"
            "2. Try a different approach\n"
            "3. Ask the user for clarification"
        )
    
    # Workflow step guidance mapping
    step = state["workflow_step"]

    # Initial state: no guidance needed for read-only operations (get_policy, list_storage_pools, test_policy)
    # Workflow only activates when actual modification tools (update_policy, apply_policy) are attempted
    if step == "initial":
        return ""

    # CRITICAL: For get_policy step, ALWAYS guide to get_policy first
    if step == "get_policy" and not state["policy_retrieved"]:
        return (
            "AGENT INSTRUCTION - CRITICAL FIRST STEP: You MUST call get_policy to retrieve the existing policy "
            "before making ANY changes. This ensures all existing rules are preserved. "
            "Do NOT call update_policy, apply_policy, or any other modification tool "
            "until you have successfully retrieved the existing policy with get_policy."
        )

    # After get_policy succeeds, generate rule
    if step == "generate_rule" and not state["rule_generated"]:
        if state.get("generated_rule"):
            rule_info = state["generated_rule"]
            return (
                f"Rule generated successfully: {rule_info.get('rule_name')}\n"
                f"Rule content:\n{rule_info.get('rule_content')}\n\n"
                "Next step: verify storage pools with list_storage_pools."
            )
        return "Generating policy rule based on user request..."

    # After rule generation, verify pools
    if step == "verify_pools" and not state["pools_verified"]:
        return (
            f"AGENT INSTRUCTION: Rule generated for filesystem '{state['filesystem']}'. "
            "CRITICAL NEXT STEP: You MUST call list_storage_pools to verify available storage pools. "
            "After pools are verified, the rule will be validated for duplicates and conflicts."
        )

    # After pools verified, validate rule (this happens automatically via validate_rule node)
    if step == "validate_rule" and not state["rule_validated"]:
        return "Validating generated rule for duplicate or conflicting intent with existing rules..."

    if step == "test_policy" and not state["policy_tested"]:
        generated_rule = state.get("generated_rule", {})
        rule_content = generated_rule.get("rule_content", "")
        existing_policy_text = state.get("existing_policy_text", "")
        
        # Check if rule validation found issues
        if state.get("has_duplicate_intent"):
            return (
                "AGENT INSTRUCTION: WARNING - The generated rule has duplicate intent with an existing rule. "
                "You should inform the user and ask if they want to proceed anyway or modify the request."
            )
        if state.get("has_conflicting_intent"):
            return (
                "AGENT INSTRUCTION: WARNING - The generated rule conflicts with existing rules. "
                "You should inform the user about the conflict and ask if they want to proceed anyway or modify the request."
            )
        
        # Combine existing policy with new rule
        combined_policy = _combine_policies(existing_policy_text, rule_content)
        
        return (
            f"AGENT INSTRUCTION: Rule validated successfully. Next step: test the complete policy with test_policy.\n"
            f"CRITICAL: Use this EXACT combined policy (existing rules + new rule):\n\n"
            f"{combined_policy}\n\n"
            f"Do NOT modify the syntax - use it exactly as shown above."
        )

    if step == "update_policy" and state["test_passed"] and not state["policy_updated"]:
        generated_rule = state.get("generated_rule", {})
        rule_content = generated_rule.get("rule_content", "")
        existing_policy_text = state.get("existing_policy_text", "")
        
        # Combine existing policy with new rule
        combined_policy = _combine_policies(existing_policy_text, rule_content)
        
        return (
            f"AGENT INSTRUCTION: Policy test passed. Ready to update with update_policy.\n"
            f"CRITICAL: Use this EXACT combined policy (all rules):\n\n"
            f"{combined_policy}\n\n"
            f"Do NOT modify - use exactly as shown."
        )

    if step == "apply_policy" and state["policy_updated"]:
        return (
            "AGENT INSTRUCTION: Policy updated successfully. Final step: apply the policy with apply_policy "
            "to execute the policy rules on the filesystem."
        )
    
    # Workflow completed - inform user and stop
    if step == "completed":
        # Check if we have migration statistics to report
        migration_stats = state.get("migration_stats")
        if migration_stats:
            stats_summary = _format_migration_stats(migration_stats)
            return (
                f"AGENT INSTRUCTION: Workflow completed successfully.\n\n"
                f"MIGRATION RESULTS:\n{stats_summary}\n\n"
                f"Present these results clearly to the user. DO NOT call any more tools. "
                f"The task is finished."
            )

        return (
            "AGENT INSTRUCTION: Workflow completed successfully. "
            "Present the results to the user and DO NOT call any more tools. "
            "The task is finished."
        )
    
    return ""


def _format_migration_stats(stats: Dict[str, Any]) -> str:
    """Format migration statistics for user-friendly display."""
    lines = []

    if "files_processed" in stats:
        files_count = stats["files_processed"]
        if files_count > 0:
            lines.append(f"✓ Successfully processed {files_count} file(s)")
        else:
            lines.append("⚠ No files were processed (0 files matched the policy criteria)")

    if "files_chosen" in stats and "total_candidates" in stats:
        lines.append(f"  - {stats['files_chosen']} of {stats['total_candidates']} candidate files were selected")

    if "files_dispatched" in stats:
        lines.append(f"  - {stats['files_dispatched']} file(s) dispatched for processing")

    if "files_skipped" in stats and stats["files_skipped"] > 0:
        lines.append(f"⚠ {stats['files_skipped']} file(s) skipped or encountered errors")

    return "\n".join(lines) if lines else "Policy applied successfully"


def _parse_tool_content(tool_content: Any) -> Dict[str, Any]:
    """Parse tool content into a dictionary.

    Args:
        tool_content: Tool response content (string or dict)

    Returns:
        Parsed dictionary

    Raises:
        ValueError: If content is a string but not valid JSON
    """
    if isinstance(tool_content, str):
        try:
            return json.loads(tool_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Tool content is not valid JSON: {str(e)}") from e
    return tool_content


def _check_tool_error(tool_result: Dict[str, Any], tool_name: str) -> tuple[bool, Optional[str]]:
    """Check if tool result contains an error.

    Detects errors from multiple sources by checking for common error keywords
    in various fields of the response. Also detects cancellation as an error.

    IMPORTANT: Respects explicit success indicators (status="success", exit_code=0)
    before checking for error keywords in output text.
    """
    if not isinstance(tool_result, dict):
        return False, None

    # Check for cancellation first (user said "no" to confirmation)
    if tool_result.get("cancelled") is True:
        error_message = tool_result.get("message") or f"Operation {tool_name} cancelled by user"
        logger.warning(f"Tool {tool_name} was cancelled by user")
        return True, str(error_message)

    # Check isError flag first (explicit error indicator)
    if tool_result.get("isError") is True:
        error_message = tool_result.get("message") or tool_result.get("text") or f"Error in {tool_name}"
        logger.warning(f"Tool {tool_name} returned isError=true: {error_message}")
        return True, str(error_message)

    # CRITICAL: Check for explicit success indicators before checking error keywords
    # If status is "success" or exit_code is 0, don't treat informational messages as errors
    status = tool_result.get("status", "").lower()
    exit_code = tool_result.get("exit_code")

    if status == "success" or exit_code == 0:
        logger.debug(f"Tool {tool_name} has explicit success indicator (status={status}, exit_code={exit_code})")
        return False, None

    # Check all text fields for error keywords (only if no explicit success)
    for field in ERROR_CHECK_FIELDS:
        field_value = str(tool_result.get(field, "")).lower()
        if field_value and any(keyword in field_value for keyword in ERROR_KEYWORDS):
            error_message = tool_result.get(field, f"Error in {tool_name}")
            
            # Provide more specific context for policy validation errors
            if tool_name in ["test_policy", "update_policy"] and "400 bad request" in field_value.lower():
                logger.warning(f"Policy validation failed for {tool_name}: {str(error_message)[:200]}")
                logger.debug("This typically indicates incorrect policy syntax. Check the raw policy content logged above.")
            else:
                logger.warning(f"Tool {tool_name} returned error in {field}: {str(error_message)[:200]}")
            
            return True, str(error_message)
    
    return False, None


def _check_success_status(tool_result: Any) -> bool:
    """Check if tool result indicates success.

    Success is determined by the absence of error indicators.
    Uses the same error detection logic as _check_tool_error.
    """
    if isinstance(tool_result, dict):
        # Success = no error detected
        is_error, _ = _check_tool_error(tool_result, "check_success")
        logger.debug(f"Tool check_success has explicit success indicator (status={tool_result.get('status')}, exit_code={tool_result.get('exit_code')})")
        return not is_error

    # For non-dict results, check string representation for error keywords
    result_str = str(tool_result).lower()
    return not any(keyword in result_str for keyword in ERROR_KEYWORDS)


def _extract_filesystem_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """Extract filesystem parameter from tool calls in messages."""
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "get_policy":
                    return tc.get("args", {}).get("filesystem")
    return None


def _handle_tool_error(tool_name: str, error_message: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool execution errors."""
    if "cancelled" in str(error_message).lower():
        logger.warning(f"User cancelled {tool_name} - marking workflow as cancelled")
        updates["error_message"] = error_message
        updates["user_cancelled"] = True
        updates["workflow_step"] = "cancelled"
    else:
        updates["error_message"] = error_message
        logger.debug("Workflow staying at current step due to error - agent can retry")
    return updates


def _process_get_policy_result(state: ILMWorkflowState, tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """Process get_policy tool result."""
    updates = {
        "policy_retrieved": True,
        "existing_policy": tool_result,
        "filesystem": _extract_filesystem_from_messages(state["messages"]),
    }
    
    if isinstance(tool_result, dict) and "policy_contents" in tool_result:
        updates["existing_policy_text"] = tool_result["policy_contents"]
    
    # Only advance workflow if in modification mode - go directly to generate_rule
    if state["workflow_step"] != "initial":
        updates["workflow_step"] = "generate_rule"
    
    logger.info(f"Policy retrieved for filesystem: {updates.get('filesystem')}")
    return updates


def _process_list_storage_pools_result(state: ILMWorkflowState, tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """Process list_storage_pools tool result."""
    updates = {}
    
    if isinstance(tool_result, dict) and "storage_pools" in tool_result:
        pools = tool_result["storage_pools"]
        if isinstance(pools, list):
            pool_names = [p.get("name") for p in pools if isinstance(p, dict) and p.get("name")]
            updates["storage_pools"] = pool_names
            updates["pools_verified"] = True
            
            # Only advance workflow if in modification mode - move to validate_rule after pools verified
            if state["workflow_step"] != "initial":
                updates["workflow_step"] = "validate_rule"
                logger.info(f"Storage pools verified: {pool_names}. Moving to rule validation.")
            else:
                logger.debug(f"Storage pools retrieved: {pool_names} (read-only request)")
        else:
            logger.warning("Pools field is not a list - may need retry")
    else:
        logger.warning("No pools data in response - workflow staying at current step")
    
    return updates


def _process_test_policy_result(state: ILMWorkflowState, tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """Process test_policy tool result."""
    test_passed = _check_success_status(tool_result)
    updates = {
        "policy_tested": True,
        "test_passed": test_passed,
    }
    
    # Check if user only wanted to test (not update)
    user_request = state.get("user_request", "")
    user_request_lower = user_request.lower() if user_request else ""
    has_testing_intent = any(kw in user_request_lower for kw in TESTING_KEYWORDS)
    
    # Only advance workflow if in modification mode and test passed
    if test_passed and state["workflow_step"] != "initial":
        # If user explicitly requested testing only, complete the workflow
        if has_testing_intent:
            updates["workflow_step"] = "completed"
            logger.info("Policy test PASSED - completing workflow (testing intent detected)")
        else:
            updates["workflow_step"] = "update_policy"
            logger.info("Policy test PASSED - ready for update")
    elif not test_passed:
        updates["error_message"] = "Policy test failed - please fix errors before updating"
        logger.warning("Policy test FAILED")
    else:
        logger.debug("Policy test PASSED (read-only validation)")
    
    return updates


def _process_update_policy_result(state: ILMWorkflowState, tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """Process update_policy tool result.
    
    By default, after updating a policy, the workflow proceeds to apply it.
    Users can opt out by using keywords like "only", "just", or "don't apply".
    """
    if _check_success_status(tool_result):
        user_request = state.get("user_request", "")
        user_request_lower = user_request.lower() if user_request else ""
        
        # Keywords that indicate user wants to skip apply
        skip_apply_keywords = ["only update", "just update", "don't apply", "do not apply", "skip apply"]
        wants_to_skip_apply = any(kw in user_request_lower for kw in skip_apply_keywords)
        
        if wants_to_skip_apply:
            logger.info("Policy updated successfully - completing workflow (skip apply intent detected)")
            return {
                "policy_updated": True,
                "workflow_step": "completed",
            }
        else:
            logger.info("Policy updated successfully - proceeding to apply (default behavior)")
            return {
                "policy_updated": True,
                "workflow_step": "apply_policy",
            }
    else:
        logger.warning("update_policy failed or was cancelled - workflow will not advance")
        return {}


def _process_apply_policy_result(tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """Process apply_policy tool result and extract migration statistics."""
    if _check_success_status(tool_result):
        logger.info("Policy applied successfully - workflow complete")

        # Extract migration statistics from the output
        migration_stats = _extract_migration_stats(tool_result)

        # Add migration statistics to the result if available
        # IMPORTANT: Clear any previous error message on successful completion
        result = {
            "workflow_step": "completed",
            "error_message": None  # Clear any previous errors
        }
        if migration_stats:
            result["migration_stats"] = migration_stats
            logger.info(f"Migration statistics: {migration_stats}")

        return result
    return {}


def _extract_migration_stats(tool_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract migration statistics from apply_policy output.

    Parses the mmapplypolicy output to find:
    - Number of files migrated/processed
    - Number of files chosen for migration
    - Any errors or skipped files
    """
    import re

    # Get the output text from various possible fields
    output_text = ""
    for field in ["text", "output", "result", "message"]:
        if field in tool_result:
            output_text = str(tool_result[field])
            break

    if not output_text:
        return None

    stats = {}

    # Pattern: "A total of X files have been migrated, deleted or processed"
    total_pattern = r"A total of (\d+) files? have been migrated, deleted or processed"
    total_match = re.search(total_pattern, output_text, re.IGNORECASE)
    if total_match:
        stats["files_processed"] = int(total_match.group(1))

    # Pattern: "Chose to migrate XKB: Y of Z candidates"
    chosen_pattern = r"Chose to migrate \d+KB: (\d+) of (\d+) candidates"
    chosen_match = re.search(chosen_pattern, output_text, re.IGNORECASE)
    if chosen_match:
        stats["files_chosen"] = int(chosen_match.group(1))
        stats["total_candidates"] = int(chosen_match.group(2))

    # Pattern: "X 'skipped' files and/or errors"
    skipped_pattern = r"(\d+) 'skipped' files and/or errors"
    skipped_match = re.search(skipped_pattern, output_text, re.IGNORECASE)
    if skipped_match:
        stats["files_skipped"] = int(skipped_match.group(1))

    # Pattern: "X files dispatched" (last occurrence is the final count)
    dispatched_pattern = r"(\d+) files dispatched"
    dispatched_matches = re.findall(dispatched_pattern, output_text, re.IGNORECASE)
    if dispatched_matches:
        stats["files_dispatched"] = int(dispatched_matches[-1])

    return stats if stats else None


def _process_tool_results(state: ILMWorkflowState, result: Dict[str, Any]) -> Dict[str, Any]:
    """Process tool execution results and update workflow state.

    Handles both successful results and errors, allowing the agent to see errors
    and decide how to recover (retry, adjust parameters, ask user, etc.).
    """
    updates = {"messages": result.get("messages", [])}

    # Extract the last tool message
    messages = result.get("messages", [])
    if not messages or not isinstance(messages[-1], ToolMessage):
        return updates
    
    last_message = messages[-1]
    tool_name = last_message.name
    tool_result = _parse_tool_content(last_message.content)
    
    logger.debug(f"Processing tool result for: {tool_name}")

    # Check for errors in tool result
    is_error, error_message = _check_tool_error(tool_result, tool_name)
    
    if is_error:
        return {**updates, **_handle_tool_error(tool_name, error_message, updates)}

    # Clear any previous error on success
    if state.get("error_message"):
        logger.debug("Clearing previous error - operation succeeded")
        updates["error_message"] = None

    # Tool-specific processing
    tool_processors = {
        "get_policy": lambda: _process_get_policy_result(state, tool_result),
        "list_storage_pools": lambda: _process_list_storage_pools_result(state, tool_result),
        "test_policy": lambda: _process_test_policy_result(state, tool_result),
        "update_policy": lambda: _process_update_policy_result(state, tool_result),
        "apply_policy": lambda: _process_apply_policy_result(tool_result),
    }
    
    processor = tool_processors.get(tool_name)
    if processor:
        updates.update(processor())
    
    return updates


def should_continue(state: ILMWorkflowState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    # If workflow was cancelled, end immediately
    if state.get("workflow_step") == "cancelled":
        logger.debug("Workflow cancelled - ending execution")
        return "end"

    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise end
    return "end"


def _get_validation_error(tool_name: str, state: ILMWorkflowState) -> Optional[str]:
    """Get validation error message for a tool call based on workflow state.
    
    Only update_policy and apply_policy trigger the modification workflow.
    Other tools (get_policy, list_storage_pools, test_policy) can be used standalone.
    """
    validation_rules = {
        "update_policy": [
            (not state["policy_retrieved"],
             "Cannot update policy without first retrieving existing policy. CRITICAL: Call get_policy first to preserve all existing rules."),
            (not state["policy_tested"],
             "Cannot update policy without testing it first. Please call test_policy to validate the policy."),
            (not state["test_passed"],
             "Cannot update policy because test_policy did not pass. Please fix the policy errors before updating."),
        ],
        "apply_policy": (
            not state["policy_updated"],
            "Cannot apply policy without updating it first. Please call update_policy to save the policy before applying."
        ),
    }
    
    rules = validation_rules.get(tool_name)
    if not rules:
        return None

    # Handle single rule (tuple) or multiple rules (list)
    if isinstance(rules, tuple):
        condition, message = rules
        return message if condition else None

    # Multiple rules - return first matching error
    for condition, message in rules:
        if condition:
            return message
    
    return None


def _create_error_response(tool_call: Dict[str, Any], tool_name: str, error: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Create an error response for a tool call."""
    error_msg = ToolMessage(
        content=json.dumps({"status": "error", "message": error}),
        tool_call_id=tool_call.get("id", "error"),
        name=tool_name,
    )
    return {"messages": [error_msg], "error_message": error, **updates}


def _reset_workflow_if_needed(state: ILMWorkflowState) -> Dict[str, Any]:
    """Reset workflow state if it was completed/cancelled and a new request comes in.
    
    Returns:
        Dictionary of state updates for reset, or empty dict if no reset needed
    """
    if state["workflow_step"] not in ["completed", "cancelled"]:
        return {}
    
    logger.debug(f"Resetting workflow from {state['workflow_step']} to initial for new request")
    return {
        "workflow_step": "initial",
        "workflow_active": False,
        "policy_retrieved": False,
        "pools_verified": False,
        "rule_generated": False,
        "rule_validated": False,
        "policy_tested": False,
        "test_passed": False,
        "policy_updated": False,
        "error_message": None,
        "user_request": None,
        "generated_rule": None,
        "rule_validation_result": None,
        "has_duplicate_intent": False,
        "has_conflicting_intent": False,
        "last_user_message": None,
    }


def _activate_workflow_for_rule_generation(
    tool_call: Dict[str, Any],
    tool_name: str,
    user_request: str,
    updates: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Activate workflow when rule generation keywords detected.
    
    Returns:
        Error response dict if tool should be blocked, None if tool can proceed
    """
    has_rule_keywords = _has_intent_keywords(user_request, RULE_GENERATION_KEYWORDS)
    
    if not has_rule_keywords:
        return None
    
    # If rule keywords found and NOT calling get_policy, activate workflow and block
    if tool_name != "get_policy":
        logger.debug("Rule generation intent detected - activating workflow, must start with get_policy")
        updates["workflow_step"] = "get_policy"
        updates["workflow_active"] = True
        updates["user_request"] = user_request
        updates["last_user_message"] = user_request
        error = f"Cannot call {tool_name} without first retrieving existing policy. CRITICAL: Call get_policy first to retrieve the existing policy, then the workflow will guide you through: list_storage_pools → generate_rule → test_policy → update_policy."
        return _create_error_response(tool_call, tool_name, error, updates)
    
    # Activate workflow but allow get_policy to proceed
    logger.debug("Rule generation intent detected - activating workflow")
    updates["workflow_step"] = "get_policy"
    updates["workflow_active"] = True
    updates["user_request"] = user_request
    updates["last_user_message"] = user_request
    return None


def _activate_workflow_for_modification(
    tool_name: str,
    messages: List[BaseMessage],
    updates: Dict[str, Any]
) -> None:
    """Activate workflow when modification tools are called.
    
    Modifies updates dict in place.
    """
    logger.debug(f"Modification tool {tool_name} called - activating workflow")
    updates["workflow_step"] = "get_policy"
    updates["workflow_active"] = True
    user_request = _find_user_request(messages)
    if user_request:
        updates["user_request"] = user_request
        updates["last_user_message"] = user_request


def _validate_test_policy_from_initial(
    tool_call: Dict[str, Any],
    tool_name: str,
    user_request: Optional[str],
    updates: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Validate test_policy call from initial state.
    
    Returns:
        Error response dict if tool should be blocked, None if tool can proceed
    """
    if not user_request:
        return None
    
    has_rule_keywords = _has_intent_keywords(user_request, RULE_GENERATION_KEYWORDS)
    
    # If rule keywords found, ALWAYS activate workflow for proper rule generation
    if has_rule_keywords:
        logger.debug("Rule generation intent detected - activating workflow to use LLM rule generator")
        updates["workflow_step"] = "get_policy"
        updates["workflow_active"] = True
        updates["user_request"] = user_request
        updates["last_user_message"] = user_request
        error = "Cannot test policy without first retrieving existing policy and generating the new rule using the LLM. CRITICAL: Call get_policy first to retrieve the existing policy, then list_storage_pools to verify pools, and the rule will be generated automatically with correct IBM Storage Scale syntax."
        return _create_error_response(tool_call, tool_name, error, updates)
    
    return None


def validate_tool_call(state: ILMWorkflowState) -> Dict[str, Any]:
    """Validate that tool calls follow the workflow sequence.
    
    When modification tools are called from 'initial' state, activate the workflow.
    """
    messages = state["messages"]
    if not messages:
        logger.debug("validate_tool_call: No messages in state")
        return {}
    
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.debug("validate_tool_call: No tool calls in last message")
        return {}

    # Reset workflow if needed
    updates = _reset_workflow_if_needed(state)
    
    # Get current workflow step (use updated value if reset occurred)
    current_workflow_step = updates.get("workflow_step", state["workflow_step"])
    
    logger.debug(f"validate_tool_call: Checking tool calls, workflow_step={current_workflow_step}")
    
    # Check each tool call
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        logger.debug(f"validate_tool_call: Processing tool '{tool_name}'")
        
        # Check if workflow should be activated from initial state
        if current_workflow_step == "initial":
            user_request = _find_user_request(messages)
            if user_request:
                # Check if this is a question (not an action request)
                if _is_question_not_action(user_request):
                    logger.debug("User request appears to be a question, not activating workflow")
                    continue
                
                # Try to activate workflow for rule generation
                error_response = _activate_workflow_for_rule_generation(
                    tool_call, tool_name, user_request, updates
                )
                if error_response:
                    return error_response
        
        # If a modification tool is called from initial state, activate workflow and block
        if tool_name in MODIFICATION_TOOLS and current_workflow_step == "initial":
            _activate_workflow_for_modification(tool_name, messages, updates)
            # Block the modification tool - must go through workflow
            error = f"Cannot call {tool_name} without first retrieving existing policy. CRITICAL: Call get_policy first to retrieve the existing policy, then the workflow will guide you through the proper sequence."
            return _create_error_response(tool_call, tool_name, error, updates)
        
        # Handle test_policy from initial state
        if tool_name == "test_policy" and current_workflow_step == "initial":
            user_request = _find_user_request(messages)
            error_response = _validate_test_policy_from_initial(
                tool_call, tool_name, user_request, updates
            )
            if error_response:
                return error_response
        
        # Check for validation errors
        error = _get_validation_error(tool_name, state)
        if error:
            return _create_error_response(tool_call, tool_name, error, updates)
    
    return updates


def create_ilm_workflow_graph(llm, tools):
    """Create the ILM workflow graph with enforced sequencing.
    
    Args:
        llm: Language model instance for agent and rule generation
        tools: List of tool instances for the workflow

    Returns:
        StateGraph ready to be compiled
        
    Raises:
        ValueError: If tools list is empty or missing required tools
    """
    # Validate inputs
    if not tools:
        raise ValueError("Tools list cannot be empty. At least one tool must be provided.")
    
    # Log available tools for debugging
    tool_names = [getattr(tool, 'name', str(tool)) for tool in tools]
    logger.debug(f"Creating workflow graph with {len(tools)} tools: {tool_names}")
    
    # Create closures that capture llm and tools
    async def agent_node_with_context(state: ILMWorkflowState) -> Dict[str, Any]:
        """Agent node with captured LLM and tools."""
        # Build context-aware system message based on workflow state
        workflow_guidance = _get_workflow_guidance(state)

        # Create enhanced messages with workflow context
        messages = state["messages"].copy()

        # ALWAYS add workflow guidance when available (including initial step)
        if workflow_guidance:
            logger.debug(f"Workflow Status: {workflow_guidance}")
            # Insert guidance as a user instruction to prompt the agent to act
            # Note: workflow_guidance is system-generated but may include sanitized user content excerpts
            sanitized_guidance = _sanitize_user_input(workflow_guidance)
            guidance_msg = HumanMessage(content=f"{WORKFLOW_STATUS_PREFIX}\n{sanitized_guidance}\n\nPlease proceed with the next step.")
            messages.append(guidance_msg)

        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)

        # Get LLM response
        response = await llm_with_tools.ainvoke(messages)
        
        return {"messages": [response]}
    
    async def tool_execution_node_with_context(state: ILMWorkflowState) -> Dict[str, Any]:
        """Tool execution node with captured tools."""
        tool_node = ToolNode(tools)

        # Execute the tool asynchronously
        result = await tool_node.ainvoke(state)

        # Update workflow state based on tool execution
        updates = _process_tool_results(state, result)
        
        return updates
    
    async def generate_rule_node_with_llm(state: ILMWorkflowState) -> Dict[str, Any]:
        """Node that generates ILM policy rules using LLM."""
        # Validate prerequisites
        if not state.get("policy_retrieved"):
            logger.error("Policy not retrieved before rule generation")
            return {
                "rule_generation_error": "Policy not retrieved",
                "error_message": "Cannot generate rule without retrieving existing policy first",
                "workflow_step": "get_policy",
            }
        
        # Extract user request using canonical function
        user_request = state.get("user_request") or _find_user_request(state["messages"])

        if not user_request:
            logger.error("No user request found for rule generation")
            return {
                "rule_generation_error": "No user request found",
                "error_message": "Cannot generate rule without user request",
            }

        # Get context - storage_pools may not be available yet (will be verified after generation)
        storage_pools = state.get("storage_pools", [])
        existing_policy_text = state.get("existing_policy_text", "")
        filesystem = state.get("filesystem", "unknown")

        # If storage pools not available, use empty list - LLM will generate rule without pool validation
        # Pools will be verified in the next step
        if not storage_pools:
            logger.info("Storage pools not yet verified - generating rule without pool validation")
            storage_pools = []

        # Call LLM-based rule generator
        result = await generate_ilm_rule_with_llm(
            llm=llm,
            user_request=user_request,
            storage_pools=storage_pools,
            existing_policy_text=existing_policy_text or "",
            filesystem=filesystem or "unknown",
        )

        if "error" in result:
            return {
                "rule_generation_error": result["error"],
                "error_message": result["error"],
            }

        # Create info message
        info_message = HumanMessage(
            content=f"[Rule Generated by LLM]\nRule Name: {result['rule_name']}\n\nGenerated Rule:\n{result['rule_content']}"
        )

        return {
            "messages": [info_message],
            "generated_rule": result,
            "rule_generated": True,
            "workflow_step": "verify_pools",
            "user_request": user_request,
        }
    
    async def validate_rule_node_with_llm(state: ILMWorkflowState) -> Dict[str, Any]:
        """Node that validates the generated rule for duplicate or conflicting intent using LLM."""
        # Validate prerequisites
        if not state.get("rule_generated"):
            logger.error("No rule generated to validate")
            return {
                "error_message": "Cannot validate rule - no rule has been generated",
                "workflow_step": "generate_rule",
            }
        
        generated_rule = state.get("generated_rule", {})
        rule_content = generated_rule.get("rule_content", "")
        existing_policy_text = state.get("existing_policy_text", "")
        
        if not rule_content:
            logger.error("Generated rule has no content")
            return {
                "error_message": "Generated rule is empty",
                "rule_validated": True,
                "workflow_step": "test_policy",
            }
        
        if not existing_policy_text:
            logger.info("No existing policy to compare against - skipping validation")
            return {
                "rule_validated": True,
                "has_duplicate_intent": False,
                "has_conflicting_intent": False,
                "workflow_step": "test_policy",
            }
        
        # Create prompt for LLM to analyze rule intent
        validation_prompt = f"""Analyze the following IBM Storage Scale ILM policy rules and determine if there are duplicates or conflicts.

CRITICAL DUPLICATE DETECTION RULES:

1. EXACT SEMANTIC DUPLICATE: Two rules are duplicates if they have:
   - Same ACTION (MIGRATE/DELETE/REPLICATE)
   - Same FROM POOL and TO POOL (for MIGRATE/REPLICATE)
   - Same FROM POOL (for DELETE)
   - Same or overlapping WHERE conditions (time thresholds, file patterns, etc.)
   
   Examples of DUPLICATES:
   - MIGRATE FROM 'system' TO 'silver' WHERE ... > 180 days (rule A)
   - MIGRATE FROM 'system' TO 'silver' WHERE ... > 180 days (rule B)
   → These are DUPLICATES even with different rule names
   
   - MIGRATE FROM 'system' TO 'silver' WHERE ... > 90 days (rule A)
   - MIGRATE FROM 'system' TO 'silver' WHERE ... > 180 days (rule B)
   → These are DUPLICATE INTENT (same pools, different thresholds)

2. CONFLICT: Rules contradict each other
   - Example: One rule migrates files TO pool X, another migrates FROM pool X

ANALYZE THESE RULES:

NEW RULE:
{rule_content}

EXISTING RULES:
{existing_policy_text}

IMPORTANT: Compare the SEMANTIC MEANING, not just the rule names. Two rules with different names but identical actions, pools, and conditions are DUPLICATES.

Respond in JSON format:
{{
  "has_duplicate_intent": true/false,
  "duplicate_explanation": "If duplicate: explain which existing rule(s) match and why (include rule names, pools, and conditions)",
  "has_conflicting_intent": true/false,
  "conflict_explanation": "If conflict: explain the contradiction"
}}"""

        logger.debug("Invoking LLM for rule validation...")
        
        try:
            response = await llm.ainvoke([HumanMessage(content=validation_prompt)])
            validation_text = response.content.strip()
            
            # Parse JSON response
            import json
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in validation_text:
                validation_text = validation_text.split("```json")[1].split("```")[0].strip()
            elif "```" in validation_text:
                validation_text = validation_text.split("```")[1].split("```")[0].strip()
            
            validation_result = json.loads(validation_text)
            
            has_duplicate = validation_result.get("has_duplicate_intent", False)
            has_conflict = validation_result.get("has_conflicting_intent", False)
            
            logger.info(f"Rule validation complete - Duplicate: {has_duplicate}, Conflict: {has_conflict}")
            
            # Create info message about validation
            validation_msg_parts = ["[Rule Validation Complete]"]
            if has_duplicate:
                dup_explanation = validation_result.get('duplicate_explanation', 'Rule has same intent as existing rule')
                validation_msg_parts.append(f"WARNING - DUPLICATE INTENT DETECTED: {dup_explanation}")
                validation_msg_parts.append("\nNOTE: The combined policy includes BOTH the existing and new rules.")
                validation_msg_parts.append("During confirmation, you can review and modify the policy if needed.")
                validation_msg_parts.append("To REPLACE an existing rule, remove the old rule line from the policy.")
            if has_conflict:
                validation_msg_parts.append(f"ERROR - CONFLICTING INTENT: {validation_result.get('conflict_explanation', 'Rule conflicts with existing rules')}")
                validation_msg_parts.append("WARNING: This may cause unexpected behavior. Review carefully before proceeding.")
            if not has_duplicate and not has_conflict:
                validation_msg_parts.append("PASSED: No duplicates or conflicts detected")
            
            info_message = HumanMessage(content="\n".join(validation_msg_parts))
            
            return {
                "messages": [info_message],
                "rule_validation_result": validation_result,
                "rule_validated": True,
                "has_duplicate_intent": has_duplicate,
                "has_conflicting_intent": has_conflict,
                "workflow_step": "test_policy",
            }
            
        except Exception as e:
            logger.error(f"Rule validation failed: {str(e)}")
            # On validation error, proceed anyway but log the issue
            return {
                "rule_validated": True,
                "has_duplicate_intent": False,
                "has_conflicting_intent": False,
                "error_message": f"Rule validation failed: {str(e)}",
                "workflow_step": "test_policy",
            }
    
    def should_proceed_to_tools(state: ILMWorkflowState) -> Literal["tools", "agent"]:
        """Determine if we should proceed to tools or return to agent after validation.
        
        If validation added an error message, return to agent to show the error.
        Otherwise, proceed to tools.
        """
        messages = state["messages"]
        if not messages:
            return "tools"
        
        last_message = messages[-1]
        # If last message is a ToolMessage with an error, return to agent
        if isinstance(last_message, ToolMessage):
            try:
                content = json.loads(last_message.content) if isinstance(last_message.content, str) else last_message.content
                if isinstance(content, dict) and content.get("status") == "error":
                    logger.debug("Validation returned error - returning to agent")
                    return "agent"
            except Exception:
                pass
        
        return "tools"
    
    def should_generate_or_validate_rule(state: ILMWorkflowState) -> Literal["generate_rule", "validate_rule", "agent"]:
        """Determine if we should generate a rule, validate a rule, or continue with agent."""
        # Check if we need to generate a rule
        if state["workflow_step"] == "generate_rule" and not state["rule_generated"]:
            return "generate_rule"
        # Check if we need to validate a rule
        if state["workflow_step"] == "validate_rule" and not state["rule_validated"]:
            return "validate_rule"
        # Otherwise continue with agent
        return "agent"

    # Create the graph
    workflow = StateGraph(ILMWorkflowState)

    # Add nodes with context
    workflow.add_node("agent", agent_node_with_context)
    workflow.add_node("tools", tool_execution_node_with_context)
    workflow.add_node("validate", validate_tool_call)
    workflow.add_node("generate_rule", generate_rule_node_with_llm)
    workflow.add_node("validate_rule", validate_rule_node_with_llm)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "validate",
            "end": END,
        },
    )

    # After validation, conditionally go to tools or back to agent (if validation failed)
    workflow.add_conditional_edges(
        "validate",
        should_proceed_to_tools,
        {
            "tools": "tools",
            "agent": "agent",
        },
    )

    # After tools, check if we need to generate a rule, validate a rule, or go back to agent
    workflow.add_conditional_edges(
        "tools",
        should_generate_or_validate_rule,
        {
            "generate_rule": "generate_rule",
            "validate_rule": "validate_rule",
            "agent": "agent",
        },
    )

    # After rule generation, go back to agent
    workflow.add_edge("generate_rule", "agent")
    
    # After rule validation, go back to agent
    workflow.add_edge("validate_rule", "agent")
    
    return workflow
