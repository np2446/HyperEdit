require('dotenv').config();
const axios = require("axios");

async function getLLMResponse(query, options = {}) {
  const {
    modelName = process.env.LLM_MODEL_NAME || "meta-llama/Meta-Llama-3.1-70B-Instruct",
    apiEndpoint = process.env.LLM_API_ENDPOINT || "https://api.hyperbolic.xyz/v1",
    bearerToken = process.env.LLM_BEARER_TOKEN || "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhcDc4MjlAbnl1LmVkdSIsImlhdCI6MTczOTgyNzI4OH0.LIbiKs_0ZDYMPQPNVT0UYxeV1mc81S-MCLaccxlfniU",
    systemPrompt = "You are an AI video editor",
    maxTokens = parseInt(process.env.LLM_CTX_SIZE || "16384")
  } = options;

  try {
    const response = await axios({
      method: 'post',
      url: `${apiEndpoint}/chat/completions`,
      headers: {
        'Authorization': `Bearer ${bearerToken}`,
        'accept':'application/json',
        'Content-Type': 'application/json'
      },
      data: {
        model: modelName,
        messages: [
          {
            role: "system",
            content: systemPrompt
          },
          {
            role: "user",
            content: query
          }
        ],
        max_tokens: maxTokens,
        temperature: 0.7,
        stream: false
      }
    });

    return {
      choice: response.data.choices[0].message.content,
      status: response.status,
      model: response.data.model
    };
  } catch (error) {
    const errorMessage = error.response?.data || error.message;
    console.error("Hyperbolic API error:", errorMessage);
    throw new Error(`LLM request failed: ${JSON.stringify(errorMessage)}`);
  }
}

// Helper function for truncating text if needed
function truncateText(text, maxChars) {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars);
}

module.exports = {
  getLLMResponse,
  truncateText
};
