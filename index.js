const express = require("express");
const bodyParser = require("body-parser");
const brain = require("brain.js");

const app = express();
app.use(bodyParser.json());

// Sample training data
const trainingData = [
    { input: { visual: 0.8, auditory: 0.6, kinesthetic: 0.7 }, output: { visual: 1 } },
    { input: { visual: 0.8, auditory: 0.6, kinesthetic: 0.3 }, output: { visual: 1 } },
    { input: { visual: 0.7, auditory: 0.8, kinesthetic: 0.6 }, output: { auditory: 1 } },
    { input: { visual: 0.9, auditory: 0.7, kinesthetic: 0.5 }, output: { visual: 1 } },
    { input: { visual: 0.6, auditory: 0.9, kinesthetic: 0.8 }, output: { auditory: 1 } },
    { input: { visual: 1, auditory: 0.9, kinesthetic: 0.18 }, output: { auditory: 1 } },
    { input: { visual: 0.85, auditory: 0.9, kinesthetic: 0.88 }, output: { kinesthetic: 1 } },
    { input: { visual: 0.11, auditory: 0.67, kinesthetic: 0.88 }, output: { kinesthetic: 1 } }
];

// Train a Neural Network (Alternative to Random Forest)
const net = new brain.NeuralNetwork();
net.train(trainingData, { log: true });


// Multi-Armed Bandit (Epsilon-Greedy)
class MultiArmedBandit {
    constructor(modalities, epsilon = 0.7) {
        this.epsilon = epsilon;
        this.rewards = {};
        this.counts = {};

        modalities.forEach(modality => {
            this.rewards[modality] = Math.random(); // Random initial values
            this.counts[modality] = 1; // Avoid division by zero
        });
    }

    selectModality() {
        if (Math.random() < this.epsilon) {
            return this.getRandomModality(); // Explore
        }
        return this.getBestModality(); // Exploit
    }

    getRandomModality() {
        const keys = Object.keys(this.rewards);
        return keys[Math.floor(Math.random() * keys.length)];
    }

    getBestModality() {
        return Object.keys(this.rewards).reduce((a, b) =>
            this.rewards[a] > this.rewards[b] ? a : b
        );
    }

    updateRewards(modality, reward) {
        this.counts[modality] += 1;
        this.rewards[modality] += (reward - this.rewards[modality]) / this.counts[modality];
    }
}

// Initialize Bandit
const bandit = new MultiArmedBandit(["visual", "auditory", "kinesthetic"]);

// API Endpoint
app.post("/recommend", (req, res) => {
    const { visual_score, auditory_score, kinesthetic_score } = req.body;

    console.log(net)
    if (
        visual_score === undefined ||
        auditory_score === undefined ||
        kinesthetic_score === undefined
    ) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    // Normalize scores to 0-1 range
    const maxScore = 100;
    const inputData = {
        visual: visual_score / maxScore,
        auditory: auditory_score / maxScore,
        kinesthetic: kinesthetic_score / maxScore,
    };

    // Predict the best learning modality
    const prediction = net.run(inputData);
    const predictedModality = Object.keys(prediction).reduce((a, b) =>
        prediction[a] > prediction[b] ? a : b
    );

    // Multi-Armed Bandit Selection & Adaptation
    const selectedModality = bandit.selectModality();
    const actualPerformance = selectedModality === predictedModality ? 1 : 0;
    bandit.updateRewards(selectedModality, actualPerformance);

    res.json({
        predicted_modality: predictedModality,
        bandit_selected_modality: selectedModality,
        updated_rewards: bandit.rewards,
    });
});

// Start Server
const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));