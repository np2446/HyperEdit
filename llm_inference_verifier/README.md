# Verifiable LLM Inference AVS

This repository demonstrates how to implement verifiable inference from a Gaia node using the Othentic Stack, designed initially by MotherDAO and updated by Team EIKOAI .

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture](#usage)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Usage](#usage)

---

## Overview

The Verifiable LLM Inference AVS demonstrates how to deploy a minimal AVS using Othentic Stack to achieve verifiable AI inference. This project is developed by Mother, the founding entity of MotherDAO, as part of its mission to bring harmony to the Ethereum ecosystem through the creation and deployment of purposeful AI agents.

### About Mother

Mother is a nurturing force that emerged to foster sustainable growth in Web3 by developing agents that perform real, valuable work rather than merely generating speculative value. With a deep understanding of EVM chains (especially Base, Ethereum, Linea, Optimism, Arbitrum), Mother's mission is to deploy AI agents that perform actual jobs like DAO governance, community management, hackathon coordination, and DeFi automation.


### 🌟 EIKO AI – The Ultimate Web3 Growth Companion! 🚀🎀  

EIKO AI is a powerful marketing automation platform designed to help Web3 projects grow communities, increase engagement, and run **seamless, reward-driven campaigns**. With **multi-chain integration**, automated rewards, and influencer collaborations, it simplifies campaign creation for **airdrops, social engagement, and community-building events**! 💖✨  

---

### ✨ Key Features  

🔗 **Multi-Chain Integration**  
Supports **Ethereum, Polygon, BNB Chain, Berachain, Agoric, Archway, Nibiru, Secret Network, Hypersign, Cosmos, Omniflex, Stargaze, Osmosis & more!**  

🏡 **Community & Event Management**  
Create and manage **Web3 communities and marketing campaigns** with ease.  

🎯 **Engaging Social Tasks**  
Set up **follows, retweets, wallet check-ins, and other interactive actions** to drive engagement.  

🎁 **Automated Rewards**  
Distribute **NFTs, tokens, XP, and other incentives** effortlessly.  

🌈 **KOL & Influencer Boosting**  
Amplify campaigns through **top Web3 influencers** for maximum exposure.  

📲 **Seamless Social Integrations**  
Connect with **Twitter, Telegram, Discord, GitHub & more** to expand reach.  

---

🚀 **Level up any Web3 campaign today!** Start creating **exciting, gamified campaigns** now! 🎀✨  
👉 [EIKO AI](https://eiko.zone/)  


Sugoi ✨(≧◡≦)💖  

### Features

- **Verifiable LLM Inference:** Ensures transparent and verifiable AI responses through AVS
- **Containerised deployment:** Simplifies deployment and scaling
- **Prometheus and Grafana integration:** Enables real-time monitoring and observability
- **Integration with Gaia Nodes:** Connects to Mother's network of AI agents

## Project Structure

```mdx
📂 verifiable-llm-inference-avs
├── Execution_Service  # Implements LLM inference execution logic
├── Validation_Service # Implements inference validation logic
├── grafana           # Grafana monitoring configuration
├── docker-compose.yml # Docker setup for Operator Nodes (Performer, Attesters, Aggregator), Execution Service, Validation Service, and monitoring tools
└── README.md         # Project documentation
```

## Architecture

The architecture leverages Othentic's AVS framework to create a verifiable inference system:

1. The Performer node executes tasks using the Task Execution Service and sends the results to the p2p network.
2. Attester Nodes validate task execution through the Validation Service.

Task Execution logic:
- Receive an LLM query
- Send query to Gaia node for inference
- Store the result in IPFS
- Share the IPFS CID as proof

Validation Service logic:
- Retrieve the LLM response from IPFS using the CID
- Validate the response format and structure
- Ensure the response adheres to Mother's communication principles
- Verify the inference came from an authorized Gaia node

This architecture ensures that AI responses are:
- Verifiable through blockchain consensus
- Transparent through IPFS storage
- Aligned with Mother's principles of providing real utility
- Secured by Othentic's robust validation framework

---

## Prerequisites

- Node.js (v 22.6.0)
- Foundry
- [Yarn](https://yarnpkg.com/)
- [Docker](https://docs.docker.com/engine/install/)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MotherDAO/verifiable-llm-inference-avs.git
   cd verifiable-llm-inference-avs
   ```

2. Install Othentic CLI:

   ```bash
   npm i -g @othentic/othentic-cli
   ```

## Usage

Follow the steps in the official documentation's [Quickstart](https://docs.othentic.xyz/main/avs-framework/quick-start#steps) Guide for setup and deployment.

### Next Steps
Customize the inference parameters, adjust validation criteria, and deploy your own verifiable AI service using Mother's infrastructure.

Building sustainable Web3 intelligence together! 🌱
