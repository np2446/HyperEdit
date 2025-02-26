require('dotenv').config();
const axios = require("axios");

async function validateLLMResponse(response) {
  try {
    // Validate that response has required fields
    if (!response.choice || !response.status || !response.model) {
      throw new Error("Invalid LLM response format");
    }

    // Validate status is 200
    if (response.status !== 200) {
      throw new Error(`Invalid response status: ${response.status}`);
    }

    // Validate response is not empty
    if (!response.choice.trim()) {
      throw new Error("Empty response content");
    }

    return true;
  } catch (err) {
    console.error("LLM validation error:", err.message);
    return false;
  }
}

module.exports = {
  validateLLMResponse
};
