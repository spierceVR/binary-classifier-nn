use ndarray::Array1;
use crate::utils::math::sigmoid;

pub(crate) struct Neuron {
    weights: Array1<f32>,
    bias: f32,
}

impl Neuron {
    pub(crate) fn new(weights: &Array1<f32>, bias: f32) -> Self {
        Self {
            weights: weights.clone(),
            bias,
        }
    }

    pub(crate) fn weights(&self) -> &Array1<f32> {
        &self.weights
    }

    pub(crate) fn mut_weights(&mut self) -> &mut Array1<f32> {
        &mut self.weights
    }

    pub(crate) fn bias(&self) -> f32 {
        self.bias
    }

    pub(crate) fn mut_bias(&mut self) -> &mut f32 {
        &mut self.bias
    }

    pub(crate) fn feedforward(&self, inputs: Array1<f32>) -> f32 {
        let sum = inputs.dot(&self.weights) + self.bias;
        sigmoid(sum)
    }
}
