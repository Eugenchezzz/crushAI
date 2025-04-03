const brain = require('brain.js');
const fs = require('fs-extra');
const path = require('path');
const { parse } = require('csv-parse/sync');

class NeuralNetworkManager {
    constructor() {
        this.net = new brain.NeuralNetwork({
            hiddenLayers: [10, 10],
            activation: 'sigmoid'
        });
        this.modelPath = path.join(__dirname, 'model.json');
        this.trained = false;
        this.loadModel();
    }

    async loadModel() {
        try {
            if (await fs.pathExists(this.modelPath)) {
                const modelData = await fs.readJson(this.modelPath);
                this.net.fromJSON(modelData);
                this.trained = true;
                console.log('Модель загружена');
            }
        } catch (error) {
            console.log('Модель не найдена, будет создана новая');
            this.trained = false;
        }
    }

    async saveModel() {
        await fs.writeJson(this.modelPath, this.net.toJSON());
        this.trained = true;
    }

    normalize(value) {
        return value / 100; // Нормализуем значения до диапазона 0-1
    }

    denormalize(value) {
        return value * 100; // Возвращаем к исходному масштабу
    }

    async train(data) {
        const normalizedData = data.map(v => this.normalize(v));
        const trainingData = normalizedData.map((value, index) => ({
            input: { 
                v1: normalizedData[Math.max(0, index - 1)] || 0,
                v2: normalizedData[Math.max(0, index - 2)] || 0,
                v3: normalizedData[Math.max(0, index - 3)] || 0
            },
            output: { value: value }
        })).filter(item => item.input.v1 !== 0);

        await this.net.train(trainingData, {
            iterations: 20000,
            errorThresh: 0.005,
            log: true,
            logPeriod: 100
        });

        await this.saveModel();
    }

    predict(lastValues) {
        if (!this.trained) {
            return null;
        }

        const normalizedValues = lastValues.map(v => this.normalize(v));
        const input = {
            v1: normalizedValues[normalizedValues.length - 1] || 0,
            v2: normalizedValues[normalizedValues.length - 2] || 0,
            v3: normalizedValues[normalizedValues.length - 3] || 0
        };
        
        const result = this.net.run(input);
        return this.denormalize(result.value);
    }
}

module.exports = NeuralNetworkManager; 