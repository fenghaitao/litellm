# Requirements Document

## Introduction

This document outlines the requirements for integrating the proven Anthropic-to-OpenAI transformation logic from examples/anthropic/server into the LiteLLM proxy. The goal is to enable Claude Code and other Anthropic clients to work seamlessly with the LiteLLM proxy without any client-side code changes, while routing requests to any OpenAI-compatible backend provider.

**Current State:**
- LiteLLM has `/v1/messages` endpoint in `litellm/proxy/anthropic_endpoints/endpoints.py`
- Current implementation uses pass-through mode that calls `litellm.anthropic_messages()`
- Existing transformation in `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`

**Target State:**
- Integrate server's transformation logic from `examples/anthropic/server/src/openai_client.py`
- Accept Anthropic API format at `/v1/messages` endpoint
- Transform to OpenAI format internally
- Route to any OpenAI-compatible provider (OpenAI, Azure, Anthropic via OpenAI format, etc.)
- Transform responses back to Anthropic format
- Maintain full backward compatibility with existing LiteLLM functionality

**Key Principle:**
- **Zero client changes required** - Anthropic SDK clients work as-is
- **Universal provider support** - Route to any provider LiteLLM supports
- **Proven transformation** - Use battle-tested logic from server implementation

## Glossary

- **LiteLLM Proxy**: The proxy server component of LiteLLM that provides a unified API gateway for multiple LLM providers
- **Anthropic Adapter**: The component that translates between Anthropic API format and OpenAI format
- **Pass-Through Mode**: A mode where the proxy accepts Anthropic-formatted requests and returns Anthropic-formatted responses
- **Claude Code**: Anthropic's IDE integration that uses the Anthropic Messages API
- **Server Implementation**: The reference implementation in examples/anthropic/server that provides proven transformation logic
- **Content Block**: A structured piece of content in Anthropic's API (text, tool_use, tool_result, image)
- **SSE (Server-Sent Events)**: The streaming protocol used by Anthropic's API
- **Model Mapping**: The system that maps Anthropic model names to provider-specific models

## Requirements

### Requirement 1: Anthropic Request Acceptance

**User Story:** As a developer using Claude Code or the Anthropic SDK, I want to send requests in Anthropic format to the LiteLLM proxy, so that I can use any backend provider without changing my client code.

#### Acceptance Criteria

1. WHEN a client sends a POST request to `/v1/messages` with Anthropic format, THE LiteLLM Proxy SHALL accept the request
2. WHEN the request includes Anthropic-specific fields (system, tools, tool_choice, max_tokens), THE LiteLLM Proxy SHALL process them correctly
3. WHEN the request includes Anthropic message format with content blocks, THE LiteLLM Proxy SHALL parse them correctly
4. WHEN the request includes tool_use and tool_result blocks, THE LiteLLM Proxy SHALL handle them appropriately
5. WHEN the request includes streaming (stream: true), THE LiteLLM Proxy SHALL enable streaming mode

### Requirement 2: Anthropic to OpenAI Message Transformation

**User Story:** As a system integrator, I want Anthropic message format to be transformed to OpenAI format, so that requests can be routed to any OpenAI-compatible provider.

#### Acceptance Criteria

1. WHEN Anthropic messages contain text content blocks, THE LiteLLM Proxy SHALL convert them to OpenAI message content strings
2. WHEN Anthropic messages contain tool_use blocks, THE LiteLLM Proxy SHALL convert them to OpenAI tool_calls format
3. WHEN Anthropic messages contain tool_result blocks, THE LiteLLM Proxy SHALL convert them to OpenAI tool messages with correct tool_call_id
4. WHEN Anthropic messages contain image blocks, THE LiteLLM Proxy SHALL convert them to OpenAI image_url format
5. WHEN Anthropic system parameter is provided, THE LiteLLM Proxy SHALL convert it to OpenAI system message

### Requirement 3: Tool Definition Transformation

**User Story:** As a developer using function calling, I want Anthropic tool definitions to be transformed to OpenAI format, so that tool calling works with any provider.

#### Acceptance Criteria

1. WHEN Anthropic tools with input_schema are provided, THE LiteLLM Proxy SHALL convert them to OpenAI tools with parameters
2. WHEN Anthropic tool_choice is "auto", THE LiteLLM Proxy SHALL convert it to OpenAI "auto"
3. WHEN Anthropic tool_choice is "any", THE LiteLLM Proxy SHALL convert it to OpenAI "required"
4. WHEN Anthropic tool_choice specifies a specific tool, THE LiteLLM Proxy SHALL convert it to OpenAI function choice format
5. WHEN tools are already in OpenAI format, THE LiteLLM Proxy SHALL pass them through unchanged

### Requirement 4: OpenAI to Anthropic Response Transformation

**User Story:** As a Claude Code user, I want responses from any provider to be transformed back to Anthropic format, so that my client can parse them correctly.

#### Acceptance Criteria

1. WHEN OpenAI response contains text content, THE LiteLLM Proxy SHALL convert it to Anthropic text content block
2. WHEN OpenAI response contains tool_calls, THE LiteLLM Proxy SHALL convert them to Anthropic tool_use content blocks
3. WHEN OpenAI response has finish_reason "stop", THE LiteLLM Proxy SHALL convert it to Anthropic stop_reason "end_turn"
4. WHEN OpenAI response has finish_reason "length", THE LiteLLM Proxy SHALL convert it to Anthropic stop_reason "max_tokens"
5. WHEN OpenAI response has finish_reason "tool_calls", THE LiteLLM Proxy SHALL convert it to Anthropic stop_reason "tool_use"

### Requirement 5: Streaming Response Transformation

**User Story:** As a Claude Code user, I want streaming responses to follow Anthropic's SSE event format, so that I can see incremental results as they are generated.

#### Acceptance Criteria

1. WHEN streaming is enabled, THE LiteLLM Proxy SHALL emit message_start event with message metadata
2. WHEN content begins, THE LiteLLM Proxy SHALL emit content_block_start event for each content block
3. WHEN content is generated, THE LiteLLM Proxy SHALL emit content_block_delta events with text_delta or input_json_delta
4. WHEN content blocks complete, THE LiteLLM Proxy SHALL emit content_block_stop events
5. WHEN the message completes, THE LiteLLM Proxy SHALL emit message_delta with stop_reason and usage, followed by message_stop

### Requirement 6: Request Validation

**User Story:** As a developer, I want invalid requests to be rejected with clear error messages, so that I can quickly identify and fix issues.

#### Acceptance Criteria

1. WHEN required fields (model, messages) are missing, THE LiteLLM Proxy SHALL return a 400 error with Anthropic-compatible error format
2. WHEN message structure is invalid, THE LiteLLM Proxy SHALL return a 400 error describing the validation failure
3. WHEN tool definitions are malformed, THE LiteLLM Proxy SHALL return a 400 error identifying the problematic tool
4. WHEN tool_result references non-existent tool_use_id, THE LiteLLM Proxy SHALL return a 400 error with valid IDs listed
5. WHEN content blocks have invalid structure, THE LiteLLM Proxy SHALL return a 400 error with specific details

### Requirement 7: Provider Routing Integration

**User Story:** As a proxy administrator, I want Anthropic requests to be routed through LiteLLM's existing routing logic, so that I can use load balancing, fallbacks, and all other LiteLLM features.

#### Acceptance Criteria

1. WHEN a model is configured in the router, THE LiteLLM Proxy SHALL route the transformed request through the router
2. WHEN load balancing is configured, THE LiteLLM Proxy SHALL distribute requests across deployments
3. WHEN fallbacks are configured, THE LiteLLM Proxy SHALL retry failed requests with fallback models
4. WHEN rate limiting is configured, THE LiteLLM Proxy SHALL enforce rate limits on Anthropic requests
5. WHEN logging is configured, THE LiteLLM Proxy SHALL log Anthropic requests with all metadata

### Requirement 8: Backward Compatibility

**User Story:** As an existing LiteLLM user, I want the new Anthropic support to not break my existing workflows, so that I can adopt it incrementally.

#### Acceptance Criteria

1. WHEN existing OpenAI-format endpoints are used, THE LiteLLM Proxy SHALL continue to work as before
2. WHEN the Anthropic adapter is not explicitly enabled, THE LiteLLM Proxy SHALL not interfere with existing request handling
3. WHEN configuration is missing Anthropic-specific settings, THE LiteLLM Proxy SHALL use sensible defaults
4. WHEN both OpenAI and Anthropic formats are used, THE LiteLLM Proxy SHALL handle them independently
5. WHEN existing integrations are tested, THE LiteLLM Proxy SHALL pass all existing test suites
