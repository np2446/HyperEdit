require('dotenv').config();
const dalService = require("./dal.service");
const llmService = require("./llm.service");

async function validate(proofOfTask) {
  try {

    // Get task result from IPFS using the CID (proofOfTask)
    const taskResult = await dalService.getIPfsTask(proofOfTask);
    
    // Validate the LLM response
    const isValid = await llmService.validateLLMResponse(taskResult);
    
    return isValid;
  } catch (err) {
    console.error("Validation error:", err?.message);
    return false;
  }
}

module.exports = {
  validate,
};
