#!/bin/bash
# Complete startup script with full debug logging for another terminal

echo "🚀 Starting Enhanced Anthropic Proxy with FULL DEBUG LOGGING"
echo "==========================================================="

# Set ALL debug environment variables
echo "🔧 Setting debug environment variables..."
export LITELLM_DEBUG=true
export LITELLM_LOG=DEBUG
export IFLOW_API_KEY="sk-904074de7f5049f3e828ec88a1a8fa7d"

echo "✅ Environment variables set:"
echo "   LITELLM_DEBUG=$LITELLM_DEBUG"
echo "   LITELLM_LOG=$LITELLM_LOG"
echo "   IFLOW_API_KEY=${IFLOW_API_KEY:0:10}..."

# Kill any existing proxy
echo "🔄 Stopping any existing proxy..."
pkill -f "litellm.*anthropic_iflow" 2>/dev/null || true
sleep 2

# Start with maximum debugging
echo "🚀 Starting proxy with MAXIMUM debug logging..."
echo "Command: .venv/bin/litellm --config anthropic_iflow_qwen3_coder_config.yaml --port 4000 --debug"
echo ""
echo "🔍 WATCH FOR THESE CRITICAL LOGS:"
echo "================================"
echo "✅ '🔍 CONFIG DEBUG: configure_from_general_settings called'"
echo "✅ '🔍 CONFIG DEBUG: Found 'anthropic' in general_settings'"
echo "✅ '✅ CONFIG: Updated Anthropic model tier mappings'"
echo "✅ '✅ CONFIG: Registered provider iflow (default: True)'"
echo "✅ '- Default provider: iflow'"
echo ""
echo "❌ IF YOU DON'T SEE THESE LOGS:"
echo "   The configuration loading is broken"
echo ""
echo "✅ IF YOU SEE THESE LOGS:"
echo "   Configuration is working, issue is elsewhere"
echo ""

.venv/bin/litellm --config anthropic_iflow_qwen3_coder_config.yaml --port 4000 --debug
