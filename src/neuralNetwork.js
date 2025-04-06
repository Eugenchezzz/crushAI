const brain = require('brain.js');
const fs = require('fs-extra');
const path = require('path');

class NeuralNetworkManager {
    constructor() {
        this.net = new brain.NeuralNetwork({
            hiddenLayers: [10, 10],
            learningRate: 0.3
        });
        this.modelPath = path.join(__dirname, 'model.json');
        this.trained = false;
        this.lastDataLength = 0;
        this.loadModel();
    }

    async loadModel() {
        try {
            if (await fs.pathExists(this.modelPath)) {
                const modelData = await fs.readJson(this.modelPath);
                this.net.fromJSON(modelData);
                this.trained = true;
            }
        } catch (error) {
            this.trained = false;
        }
    }

    async saveModel() {
        await fs.writeJson(this.modelPath, this.net.toJSON());
        this.trained = true;
    }

    normalize(value) {
        return value / 100;
    }

    denormalize(value) {
        return value * 100;
    }

    async train(data) {
        if (!Array.isArray(data) || data.length <= this.lastDataLength) {
            return;
        }

        try {
            const normalizedData = data.map(v => this.normalize(v));
            
            // Создаем обучающие данные с окном в 3 значения
            const trainingData = [];
            for (let i = 2; i < normalizedData.length; i++) {
                trainingData.push({
                    input: {
                        first: normalizedData[i-2],
                        second: normalizedData[i-1],
                        third: normalizedData[i]
                    },
                    output: {
                        next: normalizedData[i]
                    }
                });
            }

            await this.net.train(trainingData, {
                iterations: 1000,
                errorThresh: 0.005,
                log: false
            });

            await this.saveModel();
            this.lastDataLength = data.length;
        } catch (error) {
            console.error('Training error:', error);
        }
    }

    predict(data) {
        if (!this.trained || !Array.isArray(data) || data.length < 3) {
            return 0;
        }

        try {
            const normalizedData = data.map(v => this.normalize(v));
            const lastThree = normalizedData.slice(-3);
            
            const result = this.net.run({
                first: lastThree[0],
                second: lastThree[1],
                third: lastThree[2]
            });

            return this.denormalize(result.next);
        } catch (error) {
            console.error('Prediction error:', error);
            return 0;
        }
    }
}

module.exports = NeuralNetworkManager; 