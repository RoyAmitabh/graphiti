---
alwaysApply: true
---
# Read DEFAULT_GROUP_ID and DEFAULT_USERNAME keys from mcp.json described below:

**mcp.json configuration (for reference):**
```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "sse",
      "url": "http://127.0.0.1:8000/sse",
      "env": {
        "DEFAULT_GROUP_ID": "GRAPHITI",
        "DEFAULT_USERNAME": "amchoudh"
      }
    }
  }
}
```

### Before Starting Any Task

- **Always search first:** Always check if there's established knowledge before making recommendations. Use the `search_nodes` tool to look for relevant preferences and procedures before beginning work.
- **Search for facts too:** Use the `search_facts` tool to discover relationships and factual information that may be relevant to your task. For complex tasks, search both nodes and facts to build a complete picture.
- **Filter by entity type:** Specify `Preference`, `Procedure`, or `Requirement` in your node search to get targeted results.
- **Review all matches:** Carefully examine any preferences, procedures, or facts that match your current task.
- **Use `center_node_uuid`:** When exploring related information, center your search around a specific node.
- **Be proactive:** If you notice patterns in user behavior, consider storing them as preferences or procedures.
- **Prioritize specific matches:** More specific information takes precedence over general information.


### Always Save New or Updated Information

- **Capture requirements and preferences immediately:** When a user expresses a requirement or preference, use `add_memory_and_wait` to store it right away. Wait for success confirmation before searching newly added data.
  - _Best practice:_ Split very long requirements into shorter, logical chunks.
- **Be explicit if something is an update to existing knowledge.** Only add what's changed or new to the graph.
- **Document procedures clearly:** When you discover how a user wants things done, record it as a 'procedure'.
- **Record factual relationships:** When you learn about connections between entities, store these as 'facts'.
- **Be specific with categories:** Always label 'preferences' and 'procedures' with clear categories for better retrieval later.

### During Your Work

- **Respect discovered preferences:** Align your work with any preferences you've found.
- **Follow procedures exactly:** If you find a procedure for your current task, follow it step by step.
- **Apply relevant facts:** Use factual information to inform your decisions and recommendations.
- **Stay consistent:** Maintain consistency with previously identified preferences, procedures, and facts.



