use ndarray::array;

use crate::neuron::Neuron;
use crate::utils::math::deriv_sigmoid;
use crate::utils::math::sigmoid;

pub(crate) struct MyNeuralNetwork {
    //   A neural network with:
    //     - 2 inputs ( weight and height )
    //     - a hidden layer with 2 neurons (h1, h2)
    //     - an output layer with 1 neuron (o1) - gender
    //   Each neuron has the same weights and bias:
    //     - w = [0, 1]
    //     - b = 0
    h1: Neuron,
    h2: Neuron,
    o: Neuron,
}

impl MyNeuralNetwork {
    pub(crate) fn new() -> Self {
        let weights = array![0.0, 1.0];
        let bias = 0.0;
        let h1: Neuron = Neuron::new(&weights, bias);
        let h2: Neuron = Neuron::new(&weights, bias);
        let o: Neuron = Neuron::new(&weights, bias);

        Self { h1, h2, o }
    }

    fn feedforward(&self, inputs: ndarray::Array1<f32>) -> f32 {
        let h1o = self.h1.feedforward(inputs.clone());
        let h2o = self.h2.feedforward(inputs.clone());
        self.o.feedforward(array![h1o, h2o])
    }

    fn mse_loss(&self, y_actual: &ndarray::Array1<f32>, y_pred: &ndarray::Array1<f32>) -> f32 {
        let diff = y_actual - y_pred;
        diff.mapv(|x: f32| -> f32 {x.powi(2)}).mean().unwrap()
    }

    pub(crate) fn predict(&self, inputs: ndarray::Array1<f32>) -> f32 {
        self.feedforward(inputs)
    }

    pub(crate) fn train(
        &mut self,
        data: Vec<(ndarray::Array1<f32>, f32)>,
        epochs: i32,
        learning_rate: f32,
    ) {
        // input shape is (2)
        assert_eq!(data[0].0.shape(), array![2].shape());
        // data is a vec of tuples (input, y_true)
        // input is an ndarray of shape (2,)
        // y_actual is a float 

        for epoch in 0..epochs {
            for (x, y_actual) in &data {
                let sum_h1 =
                    self.h1.weights()[0] * x[0] + self.h1.weights()[1] * x[1] + self.h1.bias();
                let h1 = sigmoid(sum_h1);

                let sum_h2 =
                    self.h2.weights()[0] * x[0] + self.h2.weights()[1] * x[1] + self.h2.bias();
                let h2 = sigmoid(sum_h2);

                let sum_o1 = self.o.weights()[0] * h1 + self.o.weights()[1] * h2 + self.o.bias();
                let o1 = sigmoid(sum_o1);
                let y_pred = o1;
                
            

                // Calculate partial derivatives
                // partial L by partial w1 is partial L by partial y_pred * partial y_pred by partial w1
                let partial_L_by_partial_y_pred = -2.0 * (y_actual - y_pred);
                


            }
        }
    }
}
