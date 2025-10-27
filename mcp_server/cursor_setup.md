# Setting Up Graphiti MCP Server with Cursor

This guide explains how to connect and use the Graphiti MCP Server with Cursor IDE.

## 📋 **Step 1: Configure Cursor**

### **Option A: SSE Transport (Recommended)**

1. **Start the MCP server** (keep it running):
   ```bash
   cd mcp_server
   python3 graphiti_mcp_server.py --transport sse
   ```

2. **Add to Cursor's `mcp.json`**:

   ```json
   {
  "mcpServers": 
   {
      "graphiti-memory": {
        "transport": "sse",
        "url": "http://127.0.0.1:8000/sse",
        "env": {
        "DEFAULT_GROUP_ID": "<project repo name>",
        "DEFAULT_USERNAME": "<current username>"
      }
      }
    }  
   }
   ```


**🎉 You're all set!** Cursor can now remember and recall information from your conversations.

