---
alwaysApply: true
---
# Read from DEFAULT_GROUP_ID and DEFAULT_USERNAME env vars

## 👤 User & Group Configuration (CRITICAL - READ FIRST)

**⚠️ IMPORTANT FOR MULTI-USER CENTRALIZED SERVER:**

The MCP server is a **centralized service** shared by multiple users. Each user has their own workspace (group_id) and identity (username) configured in their local `mcp.json`:

- **`DEFAULT_GROUP_ID`**: Namespace for organizing memories by workspace/project 
- **`DEFAULT_USERNAME`**: User identifier for personalized memory storage 

**🚨 CRITICAL RULE: ALWAYS Pass These Parameters**

Since this is an SSE (remote) server, you **MUST explicitly pass** `username` and `group_id` in EVERY tool call:
- `add_memory_and_wait(username="", group_id="", ...)`
- `search_memory_facts(group_ids=[""], ...)`
- `search_memory_nodes(group_ids=[""], ...)`


**Example mcp.json configuration (for reference):**
```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "sse",
      "url": "http://127.0.0.1:8000/sse",
      "env": {
        "DEFAULT_GROUP_ID": "",
        "DEFAULT_USERNAME": ""
      }
    }
  }
}
```

## Instructions for Using Graphiti's MCP Tools for Agent Memory

### Before Starting Any Task

- **Always search first:** Use the `search_nodes` tool to look for relevant preferences and procedures before beginning work.
- **Search for facts too:** Use the `search_facts` tool to discover relationships and factual information that may be relevant to your task.
- **Filter by entity type:** Specify `Preference`, `Procedure`, or `Requirement` in your node search to get targeted results.
- **Review all matches:** Carefully examine any preferences, procedures, or facts that match your current task.

### Always Save New or Updated Information

- **Capture requirements and preferences immediately:** When a user expresses a requirement or preference, use `add_memory_and_wait` to store it right away. Wait for success confirmation before searching newly added data.
  - _Best practice:_ Split very long requirements into shorter, logical chunks.
- **Be explicit if something is an update to existing knowledge.** Only add what's changed or new to the graph.
- **Document procedures clearly:** When you discover how a user wants things done, record it as a procedure.
- **Record factual relationships:** When you learn about connections between entities, store these as facts.
- **Be specific with categories:** Label preferences and procedures with clear categories for better retrieval later.

### During Your Work

- **Respect discovered preferences:** Align your work with any preferences you've found.
- **Follow procedures exactly:** If you find a procedure for your current task, follow it step by step.
- **Apply relevant facts:** Use factual information to inform your decisions and recommendations.
- **Stay consistent:** Maintain consistency with previously identified preferences, procedures, and facts.

### Best Practices

- **Search before suggesting:** Always check if there's established knowledge before making recommendations.
- **Combine node and fact searches:** For complex tasks, search both nodes and facts to build a complete picture.
- **Use `center_node_uuid`:** When exploring related information, center your search around a specific node.
- **Prioritize specific matches:** More specific information takes precedence over general information.
- **Be proactive:** If you notice patterns in user behavior, consider storing them as preferences or procedures.



**Remember:** The knowledge graph is your memory. Use it consistently to provide personalized assistance that respects the user's established preferences, procedures, and factual context.
