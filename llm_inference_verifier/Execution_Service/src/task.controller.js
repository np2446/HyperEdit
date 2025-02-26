"use strict";
const { Router } = require("express")
const CustomError = require("./utils/validateError");
const CustomResponse = require("./utils/validateResponse");
const llmService = require("./llm.service");
const dalService = require("./dal.service");

const router = Router()

router.post("/execute", async (req, res) => {
    try {
        console.log("Received request body:", req.body);
        var taskDefinitionId = Number(req.body.taskDefinitionId) || 0;
        const query = req.body.query;
        if (!query) {
            throw new Error("Query is required");
        }
        
        console.log("Calling LLM service with query:", query);
        const result = await llmService.getLLMResponse(query); // provide query to api
        console.log("LLM response:", result);
        
        console.log("Publishing to IPFS...");
        const cid = await dalService.publishJSONToIpfs(result);
        console.log("IPFS CID:", cid);
        
        const data = llmService.truncateText(result.choice, 2000); // Truncate response for data field
        console.log("Truncated data:", data);
        
        console.log("Sending task...");
        await dalService.sendTask(cid, data, taskDefinitionId);
        
        return res.status(200).send(new CustomResponse({proofOfTask: cid, data: data, taskDefinitionId: taskDefinitionId}, "Task executed successfully"));
    } catch (error) {
        console.log("Full error object:", error);
        console.log("Error stack:", error.stack);
        return res.status(500).send(new CustomError(error.message || "Something went wrong", {}));
    }
})

module.exports = router
